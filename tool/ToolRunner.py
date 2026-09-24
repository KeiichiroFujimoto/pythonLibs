from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context


@dataclass
class ToolTask:
    name: str
    func: Callable[..., Any]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _run_task(task: ToolTask) -> Dict[str, Any]:
    result = task.func(*task.args, **task.kwargs)
    return {
        "name": task.name,
        "result": result,
    }


class ToolRunnerBase:
    def run(self, tasks: Iterable[ToolTask], collector: Optional[Any] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _ingest(collector: Optional[Any], result: Dict[str, Any]) -> None:
        if collector is None:
            return
        trace = result.get("result")
        if hasattr(collector, "merge_from_trace_dict") and isinstance(trace, dict):
            collector.merge_from_trace_dict(trace)


class ThreadRunner(ToolRunnerBase):
    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers

    def run(self, tasks: Iterable[ToolTask], collector: Optional[Any] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                self._ingest(collector, res)
        return results


class ProcessRunner(ToolRunnerBase):
    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers

    def run(self, tasks: Iterable[ToolTask], collector: Optional[Any] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                self._ingest(collector, res)
        return results
