# パイプライン改善の実装プラン

更新日: 2026-09-14。前案 `outputs/interpretation_design_v3_20260913/plan_ja.md` の実装順序・対象範囲を、この計画で置き換える。今回は計画書のみを作成し、以下の未実装項目には着手しない。

## 1. 既に修正したもの

| 状態 | 変更 | 対象 |
| --- | --- | --- |
| コミット済み | resume時にoperationsの検証attemptも復元・保持する。既存の外部出力からのresume／同一出力へのresumeテストを拡張 | `9514387`、`pipeline/runner.py`、`tests/zeroshot/test_runner.py` |
| 未コミット | ファイル生成・変更の完了後に読み込みや計測を行う指示を共通化。`load_image`は既存画像用で、JSON/Python等は`run_shell`で読むと明記 | `pipeline/stages/_base/prompts/reconstruction_history.md`、`pipeline/tools/load_image.py` |
| 未コミット | role内の重複するtool説明、round/guidelines間の提出指示・DAG説明等を整理 | interpretation・operations・coding・auditの既存prompts、`tests/zeroshot/messages/test_prompts.py` |
| 未コミット | auditプロンプト中の手書きJSON Schema展開を削除。APIへ渡す構造化回答schemaは保持 | `pipeline/stages/audit/prompts/role.md`、同promptテスト |
| 未コミット | 構造化回答をToolMessageへ再掲する代わりに `Submission received.` と返す。AIMessageの回答引数と対応tool_call_idは保持 | `pipeline/workflow/components/agent.py`、`tests/zeroshot/workflow/test_agent.py` |

既報の検証はresume関連10件、prompt等の関連テスト133件、agentテスト62件成功。これらはそれぞれの変更時の実行結果で、今回の計画書作成では再実行していない。

ユーザーによるsemantics/drawings除去・operations JSON化は前提とする。`coordinate_frames.md`の第三角法の配置ヒントもユーザー変更として保持する。

### ToolMessage短縮の適用範囲

`Submission received.` はToolStrategyが扱う **TicketAnswers / AuditReport等の構造化最終回答にだけ**適用される。`load_image`は画像ブロック、`run_shell`は実行結果を返すまま。通常toolの出力をこの文言に置き換える処理はない。

文言は既存の `Submission received.` を維持する。受領の通知であり、成果物の検証合格やauditのacceptを意味しない。

```text
AIMessage: 構造化回答のtool call
ToolMessage: Submission received.
HumanMessage: 必要なら提出内容の差し戻し、または次の指示
```

通常の画像tool callには、その直後に画像を含む対応ToolMessageが返る。構造化回答と画像toolを同じagentで使うテストで、この区別も確認する。

## 2. 今回の対象から外すもの

- 仮説から未確定量を絞る局所画像計測ツール・新たな探索手順。
- BAML、PydanticPrompt、独自のclass風schema renderer。ファイル提出用の現JSON Schema表示は維持する。
- overlay・位置合わせ・drawing_diff。
- ShapeCensusへの全体bbox追加、accept候補への追加監査、初回acceptの一律禁止。
- Coderが寸法確認をmodel.pyからprintする案。
- 最終ラウンドのaudit追加、snapshotへのaudit_report追加、そのための終了・resume経路変更。

修正ラウンドを開始できない最終codingの後は、現状どおり終了する。チケット確認は**既に実行されるaudit**で行う。ラウンド上限で止まった結果を、auditがacceptした結果と同一視しない。

## 3. チケット本文はファイルから読ませる

各stageには担当ticket IDと`reconstruction.json`の参照を提示し、本文・既存responsesは現在のsnapshotから絞って読ませる。

- `audit/stage.py`の`ticket_responses=json.dumps(...)`と、`audit/prompts/round.md`の本文展開を削除する。
- auditは現在の全open_ticketsとそのresponsesを読み、それ以外は自stageの担当ticketを読むと明記する。auditが読む対象と、解決可否をTicketReviewへ書く対象は区別する。bootstrapのresponsesも読むが、bootstrapはreview対象に含めない。
- 恒常的な読取方法は既存の`_base/prompts/reconstruction_history.md`へ置く。roundには今回のID・パス等だけを置き、同じ本文を再掲しない。
- 必須なのは回答の全件被覆。特定のtoolを一度呼んだかという行動検査は追加しない。

## 4. 一回のauditで既存ticketと現在の欠陥を扱う

### 契約

**AuditFindingのbacktraceとrevision_requestは残す。** 前案のコメントは既存fieldの省略表記であり、削除案ではなかった。変更後のfield一覧は以下。

```python
class TicketReview(BaseModel):
    ticket_id: str
    summary: str
    solved: bool


class AuditFinding(BaseModel):
    name: str
    observation: str
    evidence: list[str]
    backtrace: list[CausalHop]
    revision_request: RevisionRequest
    related_ticket_ids: list[str]  # 追加


class AuditReport(BaseModel):
    accepted: bool
    ticket_reviews: list[TicketReview]  # 追加
    findings: list[AuditFinding]
```

この一覧ではFieldのdescription・default・既存validatorを省略している。実装時は既存の制約を保持する。TicketReviewは`audit/contracts.py`へ置く。auditであることは格納場所から明らかなためstageは持たせず、TicketResponseの継承やReasoningStageの変更はしない。

### 推論と次ラウンドへの引き継ぎ

既存ticketの不具合が現在の成果物で解消したかを確認し、残った不具合と新規欠陥をまとめて現在の成果物からbacktraceする。一回のagent実行の最終AuditReportにreviewとfindingを出す。

reviewは確認結果と理由、findingは現在の観測・根拠・backtrace・修正要求を記す。未解決reviewに別のRevisionRequestを追加しない。findingの`related_ticket_ids`は同じ未解決問題との対応であり、因果backtraceの辺ではない。

- 前回と根本原因のstageが変わった場合も、今回のbacktraceとrevision_requestを採用する。
- 複数ticketが同じ現欠陥を指す場合は一つのfindingへまとめる。一ticketから複数findingへの対応も許可する。
- 旧欠陥が解消し、別の欠陥が発生した場合は旧ticketをsolvedにし、新findingを作る。
- bootstrapは初回の作業依頼として扱い、TicketReviewとfinding.related_ticket_idsの対象に含めない。初回の不具合は既存ticketとの関連IDを持たないfindingとして報告する。

次のticketは既存の`open_next_round`で今回findingから作る。古い未解決ticketと新findingを別々にticket化しない。

### bootstrapは初回限定の作業依頼

現実装も`open_next_round`ではfindingから新ticketだけを作り、bootstrapを持ち越さない。ReconstructionRunの検証もBootstrapWorkをround 0だけに制限している。この既存仕様を維持する。過去snapshotに初回ticketが残ることは、現在のopen ticketとして持ち越すことではない。

- reasoning stageは初回の`ticket_initial`へのTicketResponseを従来どおり必須とする。
- 初回auditはそのresponsesと成果物を確認するが、`ticket_reviews=[]`とし、各findingの`related_ticket_ids=[]`とする。bootstrapのsolved可否は出力させない。
- 初回のacceptedは既存の成果物検証と欠陥の有無で判定する。review対象がゼロであることをacceptの根拠にはしない。
- reject後は、今回finding由来の具体的な不具合ticketを次roundへ発行する。bootstrapを再発行したり、新ticketの継続元IDとして残したりしない。
- review対象の判定には既存のsubject型（AuditFindingかBootstrapWorkか）を使い、新しいticket種別fieldや名前による特別判定は追加しない。

### 検証・記録・互換性

1. 現在の不具合ticket（subjectがAuditFindingのticket）に、重複なく一つずつreviewがある。BootstrapWorkは除外し、bootstrapや不明なIDへのreviewを拒否する。
2. solved=FalseのID集合と、全finding.related_ticket_idsの和集合が完全一致する。
3. 一finding内のID重複は禁止。異なるfindingから同じticketを参照することは許可する。
4. 既存のbacktrace、根本原因、修正対象の検査を維持する。
5. acceptedとfindingsの整合条件を維持する。全旧ticketが解決しても、新findingがあればrejectとなる。

局所条件は`audit/contracts.py`、現在のticket集合・成果物との照合は`audit/validate.py`で検査する。エラーには不足しているticket/finding等を示し、solvedをtrueへ書き換えるよう誘導しない。solved=Trueという意味上の誤判断を機械保証できるとは扱わない。

既存の`events.jsonl`のauditイベントはAuditReport全体を記録している。reviewもそこへ記録し、既存のvalidationイベントと合わせて採否を追えることをテストする。AuditReport用の新しい保存ファイルやsnapshot fieldは追加しない。

旧ticket.subject内のAuditFindingを読めるよう、追加参照リストは既定値を空にする等の読込互換を保つ。新たに提出されるAuditReportには、現在の不具合ticketに対する全件reviewを必須とする。初回の空reviewリストもこの条件を満たす。履歴の読み込みだけで過去の回答を新規約により拒否しない。

## 5. ステージ全体の補足と寸法確認をTicketAnswersへ持たせる

TicketResponseは特定ticketへの回答に限定し、`summary: str`を維持する。全寸法をticketへ分配する前案と、無関係な寸法を先頭ticketへ寄せる規約は撤回する。

### 上流への疑義を伝える現経路

JSON化で経路が消えたわけではない。現行の`operations/prompts/guidelines.md`は暫定判断と影響するsem_をticket responseへ、`coding/prompts/guidelines.md`は上流の誤り・不足と自分の対応をticket responseへ書くよう指示している。`TicketResponse.summary`のdescriptionも上流への疑義を許容し、auditはそれを読む。

問題は、実行中のstageが担当ticketの範囲外の問題を発見しても、そのticketのsummaryへ載せることになる点。これをステージ全体の補足で受ける。一方、担当ticketの対処不能や暫定解釈を説明するために必要な上流への疑義は、引き続きそのsummaryへ記載する。担当ticketがなくagentを呼ばないstageは、現状どおり呼ばない。

### 出力契約

`remark: str | CodingSummary`にはせず、役割の異なる二つのfieldを分ける。CodingSummaryというラッパーは不要になる。

```python
class StageReport(BaseModel):
    remark: str = ""
    dimension_checks: dict[str, str] | None = None


class TicketAnswers(StageReport):
    responses: list[TicketResponse]
```

StageReportはこの二つのfieldを提出とsnapshot保存で共用するための型。JSONは平坦なままで、TicketAnswersのfieldはresponses・remark・dimension_checksの三つとなる。既存のTicketAnswersという名前は維持する。

| 項目 | 内容 | 書かないもの |
| --- | --- | --- |
| `responses[].summary` | 個別の担当ticketに対して変更したこと・変更不要の理由・残件。その回答に必要な上流の不備や、継続のために採った暫定解釈も含む | 全寸法の検査表、無関係な問題 |
| `remark` | 担当ticketへの回答に収まらない追加の指摘。対象、理由、自分がどう扱ったかを短く述べる | 通常の完了報告、summaryの詳しい説明の再掲、全寸法の確認表 |
| `dimension_checks` | Coderが全dimの実現先と成立根拠、または未成立・未確認の理由を一度だけ記載 | ticketごとの重複表、定型的な換算過程 |

補足がなければremarkは空文字にする。Coderもremarkとdimension_checksを同時に書ける。寸法欄に既に記した疑義や、interpretation.questionsに保存済みの論点は、必要なら参照するだけにして繰り返さない。

書き分けの基準は「その担当ticketへの回答・対処不能の理由・暫定解釈として必要か」。上流の話か自stageの話かでは分けない。担当外のticketにだけ関係する発見はremarkへ記し、既存ticketが分かればIDを添える。担当外ticketへのTicketResponseは作らない。

複数ticketに共通する長い説明はremarkに一度置いてもよいが、各summaryには当該ticketへの影響と対応結果を残す。どちらに置くか迷う追加の指摘もremarkへ残してよい。これによって個別ticketへの回答が省略可能になるわけではない。

```json
{
  "responses": [
    {
      "ticket_id": "ticket_001_profile",
      "stage": "coding",
      "summary": "sem_profileの接点と指定Rが両立しないため、ret_profileではRと接線連続を優先して接点を暫定的に再計算した。上流の接点定義の修正は未完了。"
    }
  ],
  "remark": "op_recessの開口方向はview_rightの隠線と合わない可能性がある。今回は計画どおり実装したため、auditで確認してほしい。",
  "dimension_checks": {
    "dim_height": "Established: ret_profileで全高を構築。最終resultのZ方向両端間距離は84.000 mmで、指定値に一致。",
    "dim_inner_radius": "Established: ret_profileでは指定Rの円弧と接線連続を実現。暫定的な接点変更の経緯はticket_001_profileのsummaryを参照。"
  }
}
```

これはprofile修正ticketと無関係なrecessの疑義も発見した場合の書式例であり、今回の形状の検証結果ではない。各寸法の説明には実現先と根拠を記し、測定した場合は最終形状のどこを測ったかと結果を示す。変数への値の代入だけで形状上の成立確認とはしない。直径から半径への定型的な計算過程は要求しない。

### Coderの入力と検証

現在のDrawingInterpretationの`views[].dimensions`から、全dimのID・印字値・種類・quantityを短い一覧として機械生成し、Coderへ提示する。OperationPlanが引用した寸法だけを対象にしない。

同値の寸法を自動で重複排除しない。84 mmの例が関係するのは、featureから未引用のdimがあっても、別IDで同じ拘束が渡っている場合があり、未引用だけでは形状違反と判定できない点。同じ拘束なら同じ実現先を参照してよいが、登録された各dim IDについて確認結果を残す。

- 新規coding提出ではdimension_checksを必須とし、キー集合が現在の全dim IDと完全一致することを検査する。
- 寸法ゼロ件なら`{}`、それ以外のstageでは`None`とする。`None`は未提出、`{}`は対象寸法ゼロとして区別する。
- 名目値nullのdimも対象とし、読取不能等の理由を記す。未成立・未確認という回答自体は拒否しない。
- 空の説明・不明なID・欠落IDを拒否する。辞書内で全dimを一度ずつ扱うため、ticket間の分配・重複集計は不要。
- 既存ticketの全件回答検査、各成果物の検証は維持する。remarkによって未検証の成果物の提出や、上流成果物の無断編集を許可しない。
- 説明の真偽や寸法の幾何学的成立をこの被覆検査が証明するとは扱わない。auditorが図面・コード・最終形状に照らして確認する。

構造の検査は`messages/tickets.py`、codingのdim被覆は`coding/validate.py`、stageによるfield利用の条件は既存の提出検証経路で確認する。TicketResponse.summaryの型と文字列検査は変更しない。

### 保存とauditへの配線

**TicketAnswersへ追加するだけでは不十分。** 現在の`build_snapshot_update()`はsubmission.responsesだけを保存側へ渡すため、追加fieldを明示的に統合する。

```python
class ReconstructionSnapshot(BaseModel):
    # 既存fieldを維持
    stage_reports: dict[ReasoningStage, StageReport] = Field(default_factory=dict)
```

- `snapshot_update.py`で、既存の参照解決を行ったsubmissionからStageReport部分を取り出す。
- `workflow/lifecycle.py`で、成果物・ticket responsesと同じsnapshot更新に`stage_reports[実行stage]`を含める。stage名はpipelineが決定する。
- 個別responsesは引き続きticket内だけに保存し、stage_reportsへ重複保存しない。追加のJSONファイルも作らない。
- 現roundで完了したstageのreportだけを許可し、後続stageは既存の上流reportを変更しない。失敗した提出はsnapshotへ反映しない。
- 新roundのstage_reportsは空から始め、過去の補足を現在の未解決問題として自動コピーしない。過去分は過去snapshotで参照可能なままにする。
- 旧snapshotはstage_reportsの既定値`{}`で読み込む。旧報告の欠落を「補足なしと確認済み」「寸法確認済み」と解釈しない。新規coding提出の必須検査と履歴読込を分ける。
- 担当ticketがなく自動通過するstageに、remark生成のための追加agent呼出しを設けない。

保存先は`.snapshots[-1].stage_reports`。これでshare_threadや古い会話の保持に依存せず、resume後もauditが読める。既存のresolverはBaseModel・dict値を再帰処理するため、このための別resolverは追加しない。

### プロンプトとauditの責務

operations/codingの上流疑義の指示を、担当ticketへの回答に必要ならsummary、それ以外の追加指摘ならremarkという規則へ変更する。一律にremarkへ移す指示にはしない。TicketResponse.summaryとStageReport.remarkのField descriptionも同じ境界に揃える。

共通promptへ置く指示案:

```text
In each assigned ticket's summary, explain your response, including any upstream issue that prevented resolution or any provisional interpretation you used to continue.
Use remark for additional concerns not covered by those ticket responses; leave it empty if there are none.
Do not repeat detailed explanations. If several tickets share a concern, you may explain it once in remark, but each summary must still state its own outcome and the concern's effect on it.
```

全stageのTicketAnswers説明・roundの「ticket responsesだけ」という文言を新契約に合わせる。恒常的な書き分け規則を各stageのrole/round/guidelinesへ重複記載しない。interpretation.questionsは形状解釈の未確定事項を保持する役割のままにする。

共通historyとauditの読取指示にstage_reportsを追加する。本文をroundへ再展開せず、現在のticket responses・stage reports・成果物を絞って読ませる。auditorはsummaryとremarkの両方にある疑義を確認し、指摘が正しければ現在の成果物からfindingを作る。新規問題ならrelated_ticket_idsは空、既存の不具合ticketの残件ならそのIDへ対応付ける。記載場所だけで新規／既存や根本原因を決めない。

auditの恒常的な指示には、summaryとremarkの書き分け、dimension_checksがCoderによる確認の主張であること、bootstrapを読みつつreview対象から除外することを明記する。report契約のField descriptionと同じ基準に揃え、roundには読取先を示す。

summaryとremarkの意味上の振り分けはvalidation errorにしない。どちらに書かれてもauditへ届くことを優先し、誤配置だけによる再提出を要求しない。一方、既存のticket回答の全件被覆・割当・空欄検査は維持し、remarkだけでticketへ回答済みとは扱わない。remark自体から自動でticketを作ったり、無条件にrejectしたりしない。

## 6. 実装・レビューの単位と検証順序

| 単位 | 内容 | 主な変更・検証対象 |
| --- | --- | --- |
| A | 既存のコンテクスト削減差分を確定。`Submission received.`と通常toolへの非適用を維持 | 1節の未コミット差分、既存prompt/agent/load_imageテスト |
| B | チケット本文の展開撤去とTicketReview導入 | `audit/{contracts,validate,stage}.py`・prompts、messages/audit・workflow・event_loggingの既存テスト |
| C | TicketAnswers.remarkとステージ全体の補足の保存・読取 | `messages/tickets.py`、`stages/{contracts,snapshot_update}.py`、`workflow/lifecycle.py`、共通・各stageのprompts、既存契約・提出・保存・resumeテスト |
| D | TicketAnswers.dimension_checks、全dim一覧と被覆検査 | `messages/tickets.py`、`coding/{stage,validate}.py`、`stages/validate.py`、coding/audit prompts、既存契約・提出・参照解決テスト |

Bでは原因stage変更、複数ticketの統合、一ticketの複数finding化、新規欠陥、accept/未解決の矛盾、旧履歴の読込を確認する。初回のTicketResponse必須・auditのreview対象ゼロ・bootstrapへのreview/関連ID禁止・初回accept/reject・次roundがfinding由来ticketだけになることも確認する。ラウンド上限で追加auditが起動しない既存挙動は維持する。

Cではticketに関係する上流不備・暫定解釈がsummaryに残る指示、ticket外の疑義がremarkへ行く指示、両方をauditが読む指示を確認する。意味上の振り分けは機械判定しないが、remarkによって未回答ticketの検査を回避できないことをテストする。補足なし、既存reportの保持、未完stageへのreport記入禁止、新roundへの持越しなし、旧snapshotの読込、担当ticketゼロ時の追加agent呼出しなしも確認し、補足が保存・resume・auditの読取りまで欠けずに届くことを検証する。

Dでは複数ticketでも全寸法がstage全体に一表だけあること、欠落・未知dim、寸法ゼロ件、名目値null、同値の別dim、未成立の明示、他stageのdimension_checks禁止、旧報告を寸法確認済みとしないことを確認する。remarkとdimension_checksの同時提出・参照解決・保存も検証する。

単体・結合テストの後、既存の不良STEPを固定したaudit試験で誤acceptとbacktraceを確認する。続いて001100をCoderから実行し直し、全dimの実現状況が提出されauditへ届くか、修正が形状へ反映されるかを確認する。最後に同じ4サンプル・モデル設定でF1/IoU、誤accept、修正後の欠陥、所要時間・tokensを比較する。

これらは開発時の評価であり、本番パイプラインに追加auditを組み込む計画ではない。幾何計測・schema表示等の保留項目を混ぜず、各変更の効果とコストを切り分ける。
