# Phase 3: Ticket-driven causal backtrace 実装計画

## 1. 目的

本リポジトリは、2D図面から3D CadQueryモデルを再構成するR&Dパイプラインである。

```text
SemanticHypothesis -> OperationPlan -> model.py -> verification -> audit
```

Phase 3では、auditorが観察した形状・コード上の欠陥をこの構造に沿って原因まで遡り、
根本原因を修正することで精度が向上するかを検証する。

検証対象は causal backtrace である。source blockの自動編集、patch適用、検索支援などを
同時導入すると交絡するため、最小構成から段階的に実装する。

## 2. 採用する構造

各roundは常に同じ順で進む。

```text
open tickets -> semantics -> operations -> coding (+ verification) -> audit
```

基本原則は以下。

1. `SemanticHypothesis`、`OperationPlan`、`model.py`の正本は各時点で一つだけ。
   ticket内に複製しない。
2. 各stageはpatchではなく完全版artifactを返す。semantic/operation専用revision agentや
   機械的patch applicationは作らない。
3. 全stageに全open ticketを見せる。direct/upstream/irrelevantの分類はモデルが行い、
   graph側では分岐しない。
4. モデルはartifactとticket responseを返す。snapshotの生成・保存はパイプラインが行う。
5. auditごとに、各`AuditFinding`からパイプラインが新しいticketを機械生成する。
   ticketは一round内の作業単位とし、cross-roundの同一issue追跡は行わない。
6. `persists`や`regression_of`によるticket lineageは、最小パイプラインの動作確認後に
   独立ablationとして検討する。
7. 履歴の正本は、全snapshotを含む一つの`reconstruction.json`とする。
8. loggingは事実だけを保存し、routingや修正可否を判断しない。

## 3. データモデル

概念形は次のとおり。厳密なField descriptionとvalidatorはC2bで確定する。

```python
class TicketResponse(BaseModel):
    ticket_id: str
    stage: Literal["semantics", "operations", "coding"]
    summary: str

class BootstrapWork(BaseModel):
    instruction: str

class Ticket(BaseModel):
    ticket_id: str
    subject: BootstrapWork | AuditFinding
    responses: list[TicketResponse]

class AuditReport(BaseModel):
    accepted: bool
    findings: list[AuditFinding]

class ReconstructionSnapshot(BaseModel):
    round: int
    last_completed_stage: ReasoningStage | None
    semantics: SemanticHypothesis | None
    operations: OperationPlan | None
    program_source: str | None
    verification: VerifyOutputResult | None
    open_tickets: list[Ticket]

class ReconstructionRun(BaseModel):
    schema_version: int
    run_id: str
    snapshots: list[ReconstructionSnapshot]
```

`TicketResponse`に`outcome`は持たせない。変更不要なら理由を`summary`に書く。
実際にartifactが変化したかはsnapshot差分から機械的に判定する。

`BootstrapWork`は初回だけパイプラインが生成する。初回作業を診断済みdefectである
`AuditFinding`として偽装せず、revision roundと同じticket処理へ乗せるために区別する。

### Cross-round lineage

初期実装ではticketをround間で連結しない。auditorは純粋な`AuditReport`を返し、
パイプラインが各findingへ新しいticket IDを発行する。過去ticketはsnapshot履歴には残るが、
current open ticketから参照しない。

## 4. `reconstruction.json`とモデルの探索

`ReconstructionRun`全体を、run directory内の一つのJSONへUTF-8で保存する。
一roundを一snapshotとし、stage成功ごとにcurrent snapshotをatomic更新する。
codingとverificationの完了後にfreezeし、rejected audit時だけ次roundを追加する。

- semanticsとoperationsはPydantic objectをそのままsnapshotへ持たせる。
- `program_source`には`model.py`全文を保存し、単体でreplay可能にする。
- `VerifyOutputResult`もsnapshotへ持たせる。
- STEP、DXF、PNG等のbinaryはrun directory内のfileとして保持し、相対pathで参照する。

モデルへrun全体をprompt展開しない。渡すのは次だけ。

- `reconstruction.json`のpath
- current `round`
- 必要最小限の`jq`使用例
- 図面と通常のstage固有指示

モデルは`jq`や`rg`で必要なsnapshotやticketを自発的に探索する。初期実装では
`find_ticket()`、属性別helper、retrieval toolを作らない。

探索失敗が実ログで確認された場合に限り、検索支援を独立ablationとして検討する。

## 5. 一roundの処理

### Bootstrap

最初のsnapshotにはmachine-ownedなbootstrap ticketを一つ入れる。これにより初回も
revision時も同じstage sequenceを通る。

### Semantics / Operations / Coding

全stageへの共通指示は以下。

```text
現在のopen ticketをすべて確認すること。
このroundで先行stageが記録したresponseと現在の正本も確認すること。
自stageが所有する完全版artifactを、必要な場合だけ更新すること。
各open ticketへTicketResponseを一つ返すこと。
他stageのartifactは編集しないこと。
```

- Semanticsは完全版`SemanticHypothesis`を返す。
- Operationsは完全版`OperationPlan`を返す。
- Codingはworkspaceの完全版`model.py`を保つ。

stage検証成功後、パイプラインはartifactを取得し、全ticketへのresponseを対応付け、
current snapshotをatomic更新する。モデル自身はsnapshotを編集しない。

### Coding verification

verificationはcodingの一部として直ちに実行する。`model.py`を静的検査・実行し、STEP生成に
成功した場合だけrenderする。成功・失敗を問わずterminalな`VerifyOutputResult`を得た時点で
`last_completed_stage="coding"`とし、snapshotをimmutableにする。sourceやticketは変更しない。

code-operation間の必須契約は以下だけ。

```python
ret_base = ...
ret_hole = ret_base.faces(...).workplane(...).hole(...)
result = ret_hole
```

各`op_x`にmodule-level `ret_x`が一つ以上あり、未知の`ret_*`がなく、
`result`が代入されていることを検査する。operation DAGを所定のsource順へ
linearizeしない。

### Audit

auditorは図面、coding完了snapshot、run履歴を確認し、現在も残る欠陥を
`AuditReport.findings`として出力する。reportの検証後、パイプラインが各findingから
fresh ticketを生成する。

reportがself validationとcross validationを通ったら、パイプラインが次roundの
snapshotを作る。新ticketのresponsesは空。findingがゼロなら`accepted=True`として
終了し、次snapshotは作らない。nodeをまたぐ場合だけAuditReportを
`ReconstructionState`へ一時保存し、`ReconstructionRun`には保存しない。

## 6. Validationの境界

Pydantic self validationは単一object内だけで決まる規則を扱う。

- 名前、空文字、重複、`extra="forbid"`
- actionごとのtarget/proposed name規則
- backtraceの連続性と終端
- report内finding nameの一意性
- `accepted`と空finding listの一致
- `last_completed_stage="coding"`ならverificationが存在する

`check_audit_report(report, snapshot)`はread-onlyなcross validationとする。

- `sem_*`、`op_*`、`ret_*`、`result`がaudited snapshotに存在する
- `ret_x <-> op_x`が対応する
- operations間hopが`depends_on`で支持される
- operations-to-semantics hopが`Operation.semantics`で支持される
- backtrace終端がrevision targetと一致する

validatorはinvalid outputを自動修正せず、
snapshotやticketを更新しない。

## 7. ファイル責務

```text
zeroshot/pipeline/messages/contracts/audit.py
    StageOutputRef、finding、backtrace、revision、AuditReport

zeroshot/pipeline/messages/contracts/reconstruction.py
    TicketResponse、Ticket、ReconstructionSnapshot、ReconstructionRun
    audit.pyのAuditFindingを一方向にimport

zeroshot/pipeline/verification/check_audit.py
    AuditReport x snapshotのpure cross validation

zeroshot/pipeline/workflow/reconstruction_store.py
    reconstruction.jsonのload、current roundのatomic更新、next round append

zeroshot/pipeline/workflow/graph.py
    stage順、retry、終了routingのみ
```

この実験中にcontracts全体の配置換えはしない。一つの責務が実際に大きくなった場合だけ
追加分割する。

## 8. 実装checkpoint

各checkpointを一commitにし、single-processの関連unit testを通した時点でレビューする。
5 sampleの統合実行はC4まで行わない。

### 完了済み

- **C1:** code block/instruction差分/自動削除を撤廃し、
  `check_program.py`を最小`ret_*`契約へ変更済み。
- **C2a (`e4caa75`):** 現在の`AuditReport`、finding、causal path、
  revision requestのself validationを実装済み。graph未接続。

### C2b: Ticket / Snapshot / Run contracts

- 上記domain modelとbootstrapを実装する。
- `AuditReport`はC2aの純粋なfinding出力のまま維持する。
- ticket response、stage進捗、round sequenceのself validatorを書く。
- JSON round-trip testを書く。graphとfilesystemにはまだ接続しない。

レビューfixture: bootstrap、finding由来ticket、stage response、invalid response。

### C2c: Cross validation / Persistence

- pureな`check_audit_report()`を実装する。
- validated findingからfresh ticketを機械生成し、IDを発行する。
- current snapshotのatomic更新とnext round appendを実装する。
- schema version、round indexの連続性、新規ticket IDの一意性を検査する。
- invalid reportやwrite失敗でpartial stateを残さないtestを書く。

### C3: Audit dry-run

- auditorへcurrent artifacts、verification、run pathを渡す。
- 新reportをparse、self validate、cross validateする。
- raw prompt/outputとvalidation errorを事実として保存する。
- ticketを次roundの修正にはまだ接続しない。

保存済み失敗例一件で、evidence、各hop、revision rootを人手確認する。

### C4: Full revision loop

- 全stageへ同じopen-ticket viewとrun pathを渡す。
- complete artifact + TicketResponseのstage出力契約を追加する。
- stage成功ごとにcurrent snapshotを更新し、coding verification後にfreezeする。
- rejected audit時だけfresh ticketsを持つnext round snapshotを追加する。
- `targeted_redo` / `upstream_changed`の排他的routingを共通round処理へ置換する。
- validなaccepted AuditReport、round上限、retry上限だけを終了条件にする。

workflow test後、次の5 sampleを実行する。

```text
000364  000405  000775  001014  001100
```

### C5: Research evaluation

JSONとraw eventから次をoffline導出する。

- root stageの妥当性とunsupported hop
- revision targetと実際に変化したsem/op name
- roundごとのfinding数、ticket数、revision反復回数
- no-change responseと実artifact差分
- contract rejectionとCAD execution/render failure
- 最終精度、token、latency、coder attempt数

pre-backtrace baselineと比較し、backtrace自体の寄与を評価する。

## 9. 初期実装のnon-goals

以下は実ログで必要性が示されるまで入れない。

- block marker、fingerprint、old/new instruction、source自動削除
- semantic/operation patch、revision agent、patch application、graph contraction
- harness側のdirect/upstream分類
- cross-round ticket lineage（`persists` / `regression_of`）、mutable ledger
- `find_ticket()`や内部属性ごとの探索tool
- strict edit scope、drawing census、TraceCAD-style step scope

各checkpointのレビューでは、変更fileの責務、モデル入出力例、機械的に保証する規則、
モデル判断に残す規則、focused test結果を提示する。checkpointをまとめて実装しない。
