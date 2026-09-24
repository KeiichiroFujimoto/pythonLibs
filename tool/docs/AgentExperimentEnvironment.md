# Agent Experiment Environment

この文書は、`pythonLibs` / `commander` に既にある機能を「ない」と誤認しないための最短ガイドです。

## 1. Fast Map (最初に見る機能)

- `toolBaseSecured.invoke()`
  - 自動で `paramDict["executionResult"]` を更新し、必要なら `ToolTrace.record()` も呼ぶ。
  - 参照: `/Users/fujimoto/Development/pythonLibs/tool/toolBaseSecured.py`
- `ToolTrace`
  - 実行イベントを保持し、`to_dict()` / `to_json()` で機械可読化。
  - `toRuntimeEvents()` で `runtime-event.v1` 互換へ変換可能（webCore runtime-observability 連携）。
  - 参照: `/Users/fujimoto/Development/pythonLibs/tool/ToolTrace.py`
- `ReportSession`
  - `ToolTrace` から `Summary / Details / Comparisons / Agent Analysis` を自動生成。
  - 参照: `/Users/fujimoto/Development/pythonLibs/tool/ReportSession.py`
- `MissionService`
  - mission 状態要約 (`statusCounts` など) を返す。
  - 参照: `/Users/fujimoto/Development/pythonLibs/commander/Services/MissionService.py`
- `ToolNodeHandler`
  - mission/node 文脈 (`mission_id`, `task_id`, `node_id`) を `ToolTrace` 側へ付与。
  - 参照: `/Users/fujimoto/Development/pythonLibs/commander/Services/ToolNodeHandler.py`

## 1.1 既存資産で既にできること

- 実行ログの機械可読化
  - `ToolTrace.to_dict()` / `to_json()` でイベント・ジョブ・メタデータを取得可能。
- webCore観測基盤への橋渡し
  - `ToolTrace.toRuntimeEvents(runId=..., traceId=...)` で `runId/sessionId/traceId` 付きイベント列を生成可能。
- 自動実行記録
  - `toolBaseSecured.invoke()` が成功/失敗時に `executionResult` と `ToolTrace` を更新。
- 実行要約と比較
  - `ReportSession` が `Summary / Details / Comparisons / Agent Analysis` を生成。
- mission 状態要約
  - `MissionService.listMissions()` が `statusCounts` を返す。
- 進行中 mission の次アクション把握
  - `MissionService.advanceMission()` が `awaitingHuman` を返す。
- パラメータ探索の集計
  - `ParametricSweep.summary()` が min/max/mean を返す。
- blocked/review-required の指示化
  - `projectionToDirectives()` が action/trial/network/reapproval を Directive 化。
- リスク/安全判定
  - `evaluateSafety()` が `allow/hold/abort` を返す。
- 承認・再承認語彙の統一
  - `DirectiveResolution` に `approved/rejected/deferred/escalated/...` を定義。
- 履歴と replay 一貫性
  - `HistoryStack`（undo/redo）と `evaluateReplayContract`（evidence-claim整合）を利用可能。

## 2. 誤認しやすいポイント

- 「ログがない」
  - まず `paramDict["executionResult"]` を確認。
  - 次に `ToolTrace.events` / `ToolTrace.to_json()` を確認。
- 「要約がない」
  - `ReportSession` を通していない可能性が高い。
- 「失敗理由が見えない」
  - `ToolTrace` の `error` フィールドを確認。
- 「ミッション文脈がない」
  - `ToolNodeHandler` 経由で実行しているか確認。

## 3. 30秒診断チェックリスト

1. `trace.attach(tool)` を実行したか。
2. `tool.invoke(...)` 経由で呼んだか（直callではなく）。
3. `tool.paramDict["executionResult"]` が更新されたか。
4. `trace.to_dict()["events"]` に `status` / `eventType` があるか。
5. `ReportSession` でレポート生成したか。

## 4. 最小レシピ

```python
from pythonLibs.tool.ToolTrace import ToolTrace
from pythonLibs.tool.ReportSession import ReportSession

trace = ToolTrace(session_id="demo")
trace.attach(service)

with ReportSession("Demo Run", trace, review=True, outputDir="./reports") as _:
    service.invoke("yourMethod", x=1.0, y=2.0)

print(trace.to_json())
```

## 5. 探索コマンド (rg)

```bash
rg -n "ToolTrace|ReportSession|executionResult|executionContext" /Users/fujimoto/Development/pythonLibs/tool
rg -n "mission_id|node_id|ToolNodeHandler|statusCounts" /Users/fujimoto/Development/pythonLibs/commander
```

## 6. 運用ルール

- 新しい観測/要約機能を追加したら、この文書の
  - `Fast Map`
  - `誤認しやすいポイント`
  - `探索コマンド`
  を同時更新する。
- 「機能がない」と判断する前に、チェックリスト 1-5 を必ず実施する。

## 7. 次の拡張候補

- `AgentReadinessSmokeTest` を追加し、
  - `invoke -> ToolTrace -> ReportSession` が壊れていないことをCIで検証する。
- `commander/api/server.py` に、`ToolTrace` 要約の軽量エンドポイントを追加する。

## 8. Related Design Docs (webCore)

- `/Users/fujimoto/Development/webCore/docs/runtime-observability/RuntimeBaseDesign.md`
- `/Users/fujimoto/Development/webCore/docs/runtime-observability/EventSchema.md`
- `/Users/fujimoto/Development/webCore/docs/runtime-observability/RunSummaryContract.md`
- `/Users/fujimoto/Development/webCore/docs/runtime-observability/MigrationPlan.md`
