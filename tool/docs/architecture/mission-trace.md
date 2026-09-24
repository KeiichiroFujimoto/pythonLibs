# Mission Trace Architecture

Mission Trace is a history-first structure that unifies tool execution results,
assets, and dialogue into a single, queryable timeline. It projects the Mission
and Task network onto ChronoVault so that every run leaves a durable trail.

## Why it exists
- Preserve execution history as a first-class artifact.
- Make results reproducible and auditable.
- Allow LLMs to answer "What happened in Mission X?" by loading a stable summary.

## Core building blocks
- ToolTrace: event stream of tool calls (args, kwargs, results, files).
- ChronoVault: storage + revision history for summaries and assets.
- Mission/Task graph: structural index for grouping and navigation.

## Data projection
Mission/Task structure is mapped to ChronoVault folders:

```
<chronovault_root>/
  <MissionId>/
    summary.md
    manifest.json
    <TaskId>/
      summary.md
      manifest.json
      outputs/
```

## Versioning
ChronoVault revision tracking is optional:
- summary.md and manifest.json are revisioned per update.
- outputs/* can be revisioned to keep file history aligned with events.

## Query model
The "Mission summary" is the primary entry point for LLM queries:
- Resolve MissionId -> summary.md path.
- Load summary.md + manifest.json for full context.

## Integration points
- ToolTrace collects events across single-process, multi-thread, or multi-process runs.
- ToolTraceChronoVaultWriter renders summaries and copies assets into outputs/.
- ToolRunner provides execution abstraction (ThreadRunner, ProcessRunner).

## Tool × Spacetime Network
Tool events provide the temporal axis; ChronoVault assets provide the spatial axis.
Together they form a navigable network that links execution history to stored evidence.

## Trunk → Branches Navigation
Knowledge should be accessed from the trunk (overview) down to branches (details),
not the other way around. Mission summaries act as the trunk: they give the map,
then guide the user into specific Tasks, outputs, and evidence as needed.

## MetaFrame Links (Minimal Schema)
MetaFrame references are embedded as lightweight links:

```
{"type": "concept", "id": "MF:HeatTransfer", "relation": "uses"}
```

Recommended fields:
- type: concept | evidence | artifact | dataset | mission | task | agent
- relation: uses | supports | derived_from | depends_on | parallel | supersedes | refers_to
