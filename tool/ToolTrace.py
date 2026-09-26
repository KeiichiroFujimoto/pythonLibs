from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pythonLibs.SchemaVersions import TRACE_EVENT_SCHEMA_VERSION as TRACE_SCHEMA_VERSION


def _normalize_path_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]

    normalized: List[str] = []
    seen = set()
    for item in items:
        if item is None:
            continue
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_execution_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}

    normalized: Dict[str, Any] = {}

    workdir = context.get("working_directory") or context.get("workdir")
    if workdir:
        normalized["working_directory"] = str(workdir)

    for key, aliases in {
        "input_files": ("input_files", "inputFiles"),
        "output_files": ("output_files", "outputFiles"),
        "declared_outputs": ("declared_outputs", "declaredOutputs"),
    }.items():
        values: List[str] = []
        for alias in aliases:
            if alias in context:
                values.extend(_normalize_path_list(context.get(alias)))
        if values:
            deduped: List[str] = []
            seen = set()
            for value in values:
                if value in seen:
                    continue
                seen.add(value)
                deduped.append(value)
            normalized[key] = deduped

    metadata = context.get("metadata")
    if isinstance(metadata, dict) and metadata:
        normalized["metadata"] = dict(metadata)

    return normalized


def _normalize_error(error: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(error, dict):
        return None
    normalized: Dict[str, Any] = {}
    for key in ("type", "message", "traceback"):
        value = error.get(key)
        if value:
            normalized[key] = str(value)
    return normalized or None


def _to_runtime_event_type(status: Optional[str]) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"running", "started", "start"}:
        return "action_started"
    if normalized in {"failed", "error", "aborted"}:
        return "action_failed"
    return "action_completed"


def _to_runtime_status(status: Optional[str]) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"running", "started", "start"}:
        return "started"
    if normalized in {"failed", "error", "aborted"}:
        return "failed"
    return "completed"


class ToolTrace:
    """
    Aggregate execution results from multiple toolBase instances into one JSON.
    Attach with tool._tool_trace = ToolTrace(...), or via attach().
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        *,
        chronovault_sink: Optional[Any] = None,
    ) -> None:
        self.session_id = session_id or f"trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.meta: Dict[str, Any] = dict(meta or {})
        self._events: List[Dict[str, Any]] = []
        self._tools: Dict[int, Dict[str, Any]] = {}
        self._event_counter = 0
        self._job_counter = 0
        self._chronovault_sink = chronovault_sink

    def set_chronovault_sink(self, sink: Optional[Any]) -> None:
        """Set an optional sink whose ``persist_event(event)`` stores each recorded event (None to disable)."""
        self._chronovault_sink = sink

    def attach(self, tool: Any, name: Optional[str] = None) -> None:
        """Record every ``invoke`` of ``tool`` in this trace from now on.

        Args:
            tool: A toolBaseSecured instance (any object whose invoke checks ``_tool_trace``).
            name (str): Display name; defaults to ``tool.name`` or the class name.
        """
        tool_name = name or getattr(tool, "name", None) or tool.__class__.__name__
        tool_id = id(tool)
        self._tools[tool_id] = {
            "name": tool_name,
            "class": tool.__class__.__name__,
        }
        setattr(tool, "_tool_trace", self)
        setattr(tool, "_tool_trace_name", tool_name)

    def detach(self, tool: Any) -> None:
        """Stop recording ``tool``; events recorded so far are kept."""
        tool_id = id(tool)
        if tool_id in self._tools:
            self._tools.pop(tool_id, None)
        if hasattr(tool, "_tool_trace"):
            delattr(tool, "_tool_trace")
        if hasattr(tool, "_tool_trace_name"):
            delattr(tool, "_tool_trace_name")

    def record(
        self,
        tool: Any,
        method_name: str,
        args: Any,
        kwargs: Any,
        result: Any,
        file_paths: Optional[List[str]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        *,
        event_type: str = "tool_invocation",
        status: Optional[str] = None,
        duration_ms: Optional[float] = None,
        parent_event_id: Optional[str] = None,
        parent_job_id: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        job_started_at: Optional[str] = None,
        job_finished_at: Optional[str] = None,
    ) -> None:
        """Append one event (normally called by ``invoke``, not by user code).

        Status is "failed" when ``error`` is given, else "completed" unless the tool reports
        its own status. Mission / task / node ids and the execution context (working
        directory, input / output files) are taken from the tool when available.
        """
        tool_name = getattr(tool, "_tool_trace_name", None) or getattr(tool, "name", None) or tool.__class__.__name__
        mission_id = getattr(tool, "_tool_trace_mission_id", None)
        task_id = getattr(tool, "_tool_trace_task_id", None)
        node_id = getattr(tool, "_tool_trace_node_id", None)
        metaframe_links = getattr(tool, "_tool_trace_metaframe_links", None)
        tool_status = getattr(tool, "getStatus", None)
        normalized_error = _normalize_error(error)
        status_value = status or ("failed" if normalized_error else "completed")
        if callable(tool_status):
            try:
                current_status = tool_status()
                if current_status is not None:
                    normalized_status = getattr(current_status, "name", None) or str(current_status)
                    if normalized_status and normalized_status != "UNDEFINED":
                        status_value = normalized_status
            except Exception:
                pass
        if execution_context is None:
            getter = getattr(tool, "getExecutionContext", None)
            if callable(getter):
                try:
                    execution_context = getter()
                except Exception:
                    execution_context = None
        normalized_context = _normalize_execution_context(execution_context)
        self._event_counter += 1
        if not job_id:
            self._job_counter += 1
            job_id = f"job_{self._job_counter:06d}"
        if not job_started_at:
            job_started_at = datetime.now(timezone.utc).isoformat()
        entry = {
            "event_id": f"evt_{self._event_counter:06d}",
            "schemaVersion": TRACE_SCHEMA_VERSION,
            "eventType": event_type,
            "status": status_value,
            "jobId": job_id,
            "jobStatus": status_value,
            "startedAt": str(job_started_at),
            "tool": tool_name,
            "class": tool.__class__.__name__,
            "method": method_name,
            "args": list(args) if isinstance(args, tuple) else args,
            "kwargs": dict(kwargs or {}),
            "result": result,
            "filePathListSaved": list(file_paths or []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry["finishedAt"] = str(job_finished_at or entry["timestamp"])
        if duration_ms is not None:
            entry["durationMs"] = float(duration_ms)
        if parent_event_id:
            entry["parentEventId"] = str(parent_event_id)
        if parent_job_id:
            entry["parentJobId"] = str(parent_job_id)
        if normalized_error:
            entry["error"] = normalized_error
        if mission_id is not None:
            entry["mission_id"] = mission_id
        if task_id is not None:
            entry["task_id"] = task_id
        if node_id is not None:
            entry["node_id"] = node_id
        if metaframe_links is not None:
            entry["metaframe_links"] = metaframe_links
        if normalized_context:
            entry["execution_context"] = normalized_context
            if normalized_context.get("working_directory"):
                entry["working_directory"] = normalized_context["working_directory"]
            for key in ("input_files", "output_files", "declared_outputs"):
                if normalized_context.get(key):
                    entry[key] = list(normalized_context[key])
        if self._chronovault_sink is not None:
            try:
                persisted = self._chronovault_sink.persist_event(entry)
                if persisted:
                    entry["chronovault"] = persisted
            except Exception as exc:
                entry["chronovault_error"] = str(exc)
        self._events.append(entry)

    def begin_job(
        self,
        *,
        tool: Any,
        method_name: str,
        parent_job_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Open a job context (job id, tool, method, start time) for a call about to run; ``record`` closes it."""
        tool_name = getattr(tool, "_tool_trace_name", None) or getattr(tool, "name", None) or tool.__class__.__name__
        self._job_counter += 1
        context = {
            "jobId": f"job_{self._job_counter:06d}",
            "tool": tool_name,
            "method": method_name,
            "startedAt": datetime.now(timezone.utc).isoformat(),
        }
        if parent_job_id:
            context["parentJobId"] = str(parent_job_id)
        return context

    def snapshot_results(self) -> List[Dict[str, Any]]:
        """Return the attached tools as ``[{name, class}]``."""
        snapshots: List[Dict[str, Any]] = []
        for info in self._tools.values():
            snapshots.append({
                "name": info.get("name"),
                "class": info.get("class"),
            })
        return snapshots

    def snapshot_jobs(self) -> List[Dict[str, Any]]:
        """Return one summary per job: id, tool, method, status, start / finish time, duration, error."""
        jobs: Dict[str, Dict[str, Any]] = {}
        for event in self._events:
            job_id = event.get("jobId")
            if not job_id:
                continue
            jobs[job_id] = {
                "jobId": job_id,
                "tool": event.get("tool"),
                "method": event.get("method"),
                "status": event.get("jobStatus") or event.get("status"),
                "startedAt": event.get("startedAt"),
                "finishedAt": event.get("finishedAt"),
                "durationMs": event.get("durationMs"),
                "eventId": event.get("event_id"),
                "parentJobId": event.get("parentJobId"),
                "error": dict(event.get("error") or {}),
            }
        return list(jobs.values())

    def to_dict(self) -> Dict[str, Any]:
        """Return the whole trace: ``session_id``, ``tools``, ``jobs``, ``events`` and ``meta``."""
        return {
            "session_id": self.session_id,
            "schemaVersion": TRACE_SCHEMA_VERSION,
            "tools": self.snapshot_results(),
            "jobs": self.snapshot_jobs(),
            "events": list(self._events),
            "meta": dict(self.meta),
        }

    def to_json(self) -> str:
        """Return ``to_dict()`` as indented JSON (non-JSON values are converted with ``str``)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    def toRuntimeEvents(
        self,
        *,
        runId: Optional[str] = None,
        traceId: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert the events into ``runtime-event.v1`` dicts (run / session / trace ids, input, output, error)."""
        runtime_events: List[Dict[str, Any]] = []
        resolved_run_id = runId or str(self.meta.get("runId") or self.session_id)
        resolved_trace_id = traceId or str(self.meta.get("traceId") or self.session_id)

        for event in self._events:
            status = event.get("status")
            runtime_event: Dict[str, Any] = {
                "schemaVersion": "runtime-event.v1",
                "eventId": str(event.get("event_id") or ""),
                "eventType": _to_runtime_event_type(status),
                "timestamp": str(event.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                "runId": str(event.get("runId") or resolved_run_id),
                "sessionId": str(event.get("sessionId") or self.session_id),
                "traceId": str(event.get("traceId") or resolved_trace_id),
                "actionId": str(event.get("method") or ""),
                "nodeId": str(event.get("node_id") or event.get("task_id") or ""),
                "status": _to_runtime_status(status),
                "startedAt": str(event.get("startedAt") or event.get("timestamp") or ""),
                "finishedAt": str(event.get("finishedAt") or event.get("timestamp") or ""),
                "durationMs": event.get("durationMs"),
                "input": {
                    "args": event.get("args"),
                    "kwargs": event.get("kwargs"),
                },
                "output": event.get("result") if isinstance(event.get("result"), dict) else {"result": event.get("result")},
                "refs": {
                    "artifactRefs": list(event.get("filePathListSaved") or []),
                },
                "metadata": {
                    "tool": event.get("tool"),
                    "class": event.get("class"),
                    "jobId": event.get("jobId"),
                    "parentJobId": event.get("parentJobId"),
                    "missionId": event.get("mission_id"),
                    "taskId": event.get("task_id"),
                },
            }
            error = event.get("error")
            if isinstance(error, dict) and error:
                runtime_event["error"] = {
                    "code": error.get("type"),
                    "message": str(error.get("message") or ""),
                    "phase": "tool_invocation",
                }
            runtime_events.append(runtime_event)
        return runtime_events

    def merge_from_trace_dict(self, trace_dict: Dict[str, Any]) -> None:
        """Append the tools and events of another trace (a ``to_dict()`` result) to this one."""
        if not trace_dict:
            return
        for tool in trace_dict.get("tools", []):
            key = (tool.get("name"), tool.get("class"))
            if key not in self._tools:
                self._tools[key] = tool
        for event in trace_dict.get("events", []):
            self._events.append(event)

    @staticmethod
    def merge_trace_dicts(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine several ``to_dict()`` results into one dict with de-duplicated tools and all events."""
        combined = {
            "session_id": "merged",
            "tools": [],
            "events": [],
            "meta": {},
        }
        seen = set()
        for trace in traces:
            for tool in trace.get("tools", []):
                key = (tool.get("name"), tool.get("class"))
                if key in seen:
                    continue
                seen.add(key)
                combined["tools"].append(tool)
            combined["events"].extend(trace.get("events", []))
        return combined
