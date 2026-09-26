# Agent Experiment Environment

This document is the shortest guide to avoid mistaking features that already exist in `pythonLibs` / `commander` for missing ones.

## 1. Fast Map (features to look at first)

- `toolBaseSecured.invoke()`
  - Updates `paramDict["executionResult"]` automatically and also calls `ToolTrace.record()` when needed.
  - See: `/Users/fujimoto/Development/pythonLibs/tool/toolBaseSecured.py`
- `ToolTrace`
  - Keeps execution events and makes them machine-readable with `to_dict()` / `to_json()`.
  - `toRuntimeEvents()` converts them to the `runtime-event.v1` format (webCore runtime-observability integration).
  - See: `/Users/fujimoto/Development/pythonLibs/tool/ToolTrace.py`
- `ReportSession`
  - Generates `Summary / Details / Comparisons / Agent Analysis` from a `ToolTrace` automatically.
  - See: `/Users/fujimoto/Development/pythonLibs/tool/ReportSession.py`
- `MissionService`
  - Returns a summary of mission states (`statusCounts` etc.).
  - See: `/Users/fujimoto/Development/pythonLibs/commander/Services/MissionService.py`
- `ToolNodeHandler`
  - Attaches the mission/node context (`mission_id`, `task_id`, `node_id`) to the `ToolTrace`.
  - See: `/Users/fujimoto/Development/pythonLibs/commander/Services/ToolNodeHandler.py`

## 1.1 What the existing assets already do

- Machine-readable execution logs
  - `ToolTrace.to_dict()` / `to_json()` give the events, jobs and metadata.
- Bridge to the webCore observability layer
  - `ToolTrace.toRuntimeEvents(runId=..., traceId=...)` produces an event sequence with `runId/sessionId/traceId`.
- Automatic execution records
  - `toolBaseSecured.invoke()` updates `executionResult` and `ToolTrace` on success and on failure.
- Execution summaries and comparisons
  - `ReportSession` generates `Summary / Details / Comparisons / Agent Analysis`.
- Mission state summary
  - `MissionService.listMissions()` returns `statusCounts`.
- Next action of a running mission
  - `MissionService.advanceMission()` returns `awaitingHuman`.
- Aggregation of parameter sweeps
  - `ParametricSweep.summary()` returns min/max/mean.
- Turning blocked / review-required items into directives
  - `projectionToDirectives()` turns action/trial/network/reapproval into Directives.
- Risk / safety decisions
  - `evaluateSafety()` returns `allow/hold/abort`.
- One vocabulary for approval and re-approval
  - `DirectiveResolution` defines `approved/rejected/deferred/escalated/...`.
- History and replay consistency
  - `HistoryStack` (undo/redo) and `evaluateReplayContract` (evidence-claim consistency) are available.

## 2. Common misconceptions

- "There is no log"
  - First check `paramDict["executionResult"]`.
  - Then check `ToolTrace.events` / `ToolTrace.to_json()`.
- "There is no summary"
  - Most likely the run did not go through `ReportSession`.
- "The failure reason is not visible"
  - Check the `error` field of `ToolTrace`.
- "There is no mission context"
  - Check that the run goes through `ToolNodeHandler`.

## 3. 30-second diagnostic checklist

1. Was `trace.attach(tool)` called?
2. Was the call made through `tool.invoke(...)` (not a direct call)?
3. Was `tool.paramDict["executionResult"]` updated?
4. Do the entries of `trace.to_dict()["events"]` have `status` / `eventType`?
5. Was the report generated with `ReportSession`?

## 4. Minimal recipe

```python
from pythonLibs.tool.ToolTrace import ToolTrace
from pythonLibs.tool.ReportSession import ReportSession

trace = ToolTrace(session_id="demo")
trace.attach(service)

with ReportSession("Demo Run", trace, review=True, outputDir="./reports") as _:
    service.invoke("yourMethod", x=1.0, y=2.0)

print(trace.to_json())
```

## 5. Search commands (rg)

```bash
rg -n "ToolTrace|ReportSession|executionResult|executionContext" /Users/fujimoto/Development/pythonLibs/tool
rg -n "mission_id|node_id|ToolNodeHandler|statusCounts" /Users/fujimoto/Development/pythonLibs/commander
```

## 6. Working rules

- When a new observation / summary feature is added, update these sections of this document together:
  - `Fast Map`
  - `Common misconceptions`
  - `Search commands`
- Before concluding that "the feature does not exist", always go through checklist items 1-5.

## 7. Next extension candidates

- Add an `AgentReadinessSmokeTest`
  - that verifies in CI that `invoke -> ToolTrace -> ReportSession` is not broken.
- Add a lightweight endpoint for `ToolTrace` summaries to `commander/api/server.py`.

## 8. Related Design Docs (webCore)

- `/Users/fujimoto/Development/webCore/docs/runtime-observability/RuntimeBaseDesign.md`
- `/Users/fujimoto/Development/webCore/docs/runtime-observability/EventSchema.md`
- `/Users/fujimoto/Development/webCore/docs/runtime-observability/RunSummaryContract.md`
- `/Users/fujimoto/Development/webCore/docs/runtime-observability/MigrationPlan.md`
