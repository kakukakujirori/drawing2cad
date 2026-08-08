# Staged reconstruction workflow 実装計画（v3）

この文書は `implementation_plan_v2.md` の未実装部分を置き換える。
v2 に記録されている既完了作業の履歴は残すが、今後の設計判断には本書を使う。

本実装は挙動維持だけを目的としたリファクタではなく、単一 coder workflow を
段階的な推論・検証 workflow へ更新する作業である。過去の単一 agent baseline は
以前の commit から参照できるため、現行 `graph.py` の出力を厳密に固定する
characterization step は設けない。

---

## 1. 目標と非目標

最終的に次の二重ループを持つ workflow を作る。

1. **agent 内ループ**: model が tool call を続ける間、現在の
   `create_agent_subgraph` が ReAct loop を実行する。
2. **stage 間ループ**: verifier / critic の型付き判定に基づき、親 graph が
   semantic hypothesis、operation plan、code のいずれかへ差し戻す。

初期の逐次構成は次の通りとする。

```text
input
  -> semantic hypothesis（cross-view correspondenceの推論はtranscriptに保持）
  -> verify hypothesis
  -> operation plan
  -> verify operations
  -> code generation + self verification
  -> final render / output critic
  -> accept or revise one of the upstream stages
```

現段階の非目標:

- graph topology を YAML DSL にすること
- 汎用 `Stage` class / stage registry / generic executor を先に作ること
- 全stageへ terminal submission tool を追加すること
- fan-out のための package や抽象化を先回りして作ること
- hidden Chain-of-Thought を取得・保存すること

---

## 2. 確定した設計判断

### 2.1 `workflow/agent.py` に単一agentの責務を集約する

現在の `subgraphs/agent.py` は `workflow/agent.py` へ移動し、`spec.py` の
`AgentSpec` も同じファイルへ統合する。

```text
workflow/
├── agent.py       AgentSpec, create_agent_subgraph, budget notice
├── state.py       AgentState, ReconstructionState, stage output schemas
├── graph.py       domain topology と薄いstage adapter
└── fan_out_reduce.py  # fan-outを実装する時点で初めて追加
```

理由:

- `AgentSpec` と `create_agent_subgraph` は、現状どちらも「1体のagentを構成して
  実行可能にする」という同じ変更理由を持つ。
- `spec.py` は1 classだけを隔離しており、独立moduleにするほど別の関心事ではない。
- 再利用可能なsubgraph templateは現時点では単一agent loopしかない。
  fan-outは性質の異なる合成処理であり、同じfolderに置くためだけに
  `subgraphs/` packageを維持する必要はない。

`AgentState` と `StopReason` は graph state contract なので `state.py` に残す。
外部importは引き続き `zeroshot.pipeline.workflow` のre-exportを正規経路とし、
Hydra の `_target_: zeroshot.pipeline.workflow.AgentSpec` は変更しない。

`subgraphs/agents_fan_out_reduce.py` の空placeholderは削除する。fan-out着手時に
必要な設計で `workflow/fan_out_reduce.py` を追加する。

### 2.2 stage出力型は当面 `state.py` に置く

各stageのPydantic modelと、それを保持する `ReconstructionState` は
`workflow/state.py` にまとめる。現時点ではこれらはすべてworkflow内部のdata
contractであり、別moduleに分けるより「stateに何が載るか」を1箇所で読める方が
有益である。

次のいずれかが実際に発生した時だけ `artifacts.py` 等への分離を検討する。

- stage output schemas がworkflow外からも独立APIとして利用される
- schema versioning / migrationが `ReconstructionState` とは別に必要になる
- `state.py` 内でschema群とLangGraph state定義の変更理由が明確に分かれる

単なる行数増加だけを理由には分割しない。

### 2.3 View Registration は独立した型付きartifactにしない

ここでいう View Registration は front / top / right のview名を割り当てる処理では
ない。view名は入力DXFのlayerから既知である。

必要なのは、**異なるview layer上のどの2D primitive群が同じ3D partに由来するか**
という primitive-wise correspondence である。これはpartのsemantic解釈と分離して
確定しにくく、対応関係の `evidence` を機械的に検証できる形で要求することも難しい。
したがって初版では `ViewRegistration`、`PrimitiveRef`、`PartHypothesis` を定義しない。

semantic agentはcross-view correspondenceを推論過程として扱うが、stageの型付き出力は
`SemanticHypothesis` の `semantics: list[str]` だけとする。primitive対応、根拠、曖昧性
などmodelが返した補足情報はraw AIMessageおよびtool interactionとしてtranscriptへ残す。
後段はこの型付きartifactとtranscriptの両方を受け取る。

この契約はprimitive対応を不要と判断したという意味ではない。現時点で安定して型付け
できない情報を、根拠の弱いID schemeやfieldで固定しないという判断である。実験により
programmaticなprimitive参照が必要だと分かった時点で、実データに基づいて拡張する。

### 2.4 型付き出力とmessage transcriptを両方引き継ぐ

stage間handoffには二つの経路を併用する。

- **型付きartifact**: routing、validation、後段の確実な参照に使う。
- **raw message history**: modelが返した説明、tool result、providerが公開した
  reasoning block等を後段modelへ引き継ぐ。

型付き出力を導入しても、AIMessageは捨てない。`ReconstructionState.messages` を
run全体のcanonical transcriptとし、各stageが新たに生成したmessageだけを一度
追加する。

次stageへ渡すcontextでは、過去stageの `SystemMessage` とsyntheticなturn budget
noticeを除外する。これらはstateとevent logには残すが、別roleのsystem promptとの
競合やbudget表示の混線を避けるためmodel inputには再投入しない。Human / AI /
Tool messageと、providerから実際に返されたcontent blockは維持する。

providerが返さないhidden CoTは保存できない。本計画でいうthinkingの継承は、APIが
返すAIMessage本文・reasoning summary/content block・tool interactionの範囲を指す。

### 2.5 終了条件は従来通り「tool callなし」

stage専用の `submit_*` terminal toolは作らない。agentは現在と同じく、modelの
最終AIMessageにtool callが無ければ終了する。

推論stageのpromptでは最終応答をraw JSONにするよう指示し、終了後にそのstageを
所有するworkflow nodeが対応するPydantic modelで検証する。review loopをlocal graphへ
閉じ込める場合は、そのlocal graphのnodeが検証を所有する。JSONと型が不正ならstageを
成功扱いせず、明確なvalidation errorでrunを止める。

初版では自動reformat turnを追加しない。live実験でformat errorが無視できない割合に
達した場合だけ、元の推論をやり直さず整形だけを要求するrepair pathを検討する。

### 2.6 topologyとtool配分は `graph.py` が所有する

model、role/prompt、turn budgetは `AgentSpec` を介してHydraから指定する。
どのstageをどう接続し、どのtool集合を与えるかはPython graphが明示する。

`create_agent_subgraph(spec, tools, ...)` は既にagentごとのtool集合を受け取れるため、
初版では `AgentSpec.tools` や `toolbelt.py` を追加しない。tool配分自体をHydraから振る
ablationが必要になった時に初めてconfig化する。

---

## 3. 実装ステップと完了条件

### ✅ Step 1 — agent moduleをflat化する

作業:

1. `subgraphs/agent.py` を `workflow/agent.py` へ移す。
2. `AgentSpec` を `spec.py` から `agent.py` へ移す。
3. `workflow/__init__.py` のre-export元を更新する。
4. `spec.py` と `subgraphs/` package、空のfan-out placeholderを削除する。
5. agent loop固有のtestを `tests/zeroshot/workflow/test_agent.py` に置き、
   `test_graph.py` は親graphの合成とroutingを中心にする。

完了条件:

- production/testのどこにも `workflow.spec` / `workflow.subgraphs` importが残らない。
- `from zeroshot.pipeline.workflow import AgentSpec, create_agent_subgraph` は維持される。
- Hydraから `AgentSpec` をinstantiateできる。
- agent loopのtool往復、budget、retry、stop reasonのtestがgreenである。
- `tests/zeroshot` 全体がgreenである。

これは純粋な配置変更として実施し、stage機能追加と同じdiffには混ぜない。

### ✅ Step 2 — 最初のstage output contractとstateを定義する

作業:

1. `state.py` に `SemanticHypothesis(semantics: list[str])` を定義する。
2. semantic review loopだけが使う内部契約として、`decision: accept | revise` と
   `feedback` を持つ `SemanticHypothesisReview` を定義する。
3. 両Pydantic modelを `extra="forbid"` とし、schema driftを検出する。
4. `revise` の場合は空または空白だけの `feedback` をvalidation errorにする。
5. `ReconstructionState` にはcanonical `messages` とaccepted
   `semantic_hypothesis` だけを追加する。reviewとrevision counterは追加しない。
6. 親graphのpublic artifactである `SemanticHypothesis` を `workflow/__init__.py` から
   re-exportする。内部review型はre-exportしない。

完了条件:

- validな最終JSONから `SemanticHypothesis` / `SemanticHypothesisReview` を構築できる。
- unknown field、欠落field、不正decision、feedbackの無いrevise判定を拒否できる。
- `ReconstructionState` に `SemanticHypothesis` はあるが、reviewとrevision counterはない。
- `PrimitiveRef`、`PartHypothesis`、独立 `ViewRegistration` が存在しない。
- schemaを含む `ReconstructionState` がLangGraph checkpointerをround-tripできる。
- schema testはmodel、agent、toolを起動せず実行できる。

この時点ではoperations以降の型を定義しない。

### Step 3 — message継承とtyped result抽出を `graph.py` に実装する

作業:

1. 親stateから次agentへ渡すcontextを作る小さなhelperを `graph.py` に置く。
2. 過去のSystemMessageとturn budget noticeだけをmodel contextから除外する。
3. `context + 現stageのHumanMessage` をagentの `task` として渡す。
4. subgraph結果から、継承済みcontextを除いたcurrent-stage deltaだけを親stateへ返す。
   current stageのSystemMessage、stage instruction、tool interaction、最終AIMessageは
   deltaに含める。
5. tool callを持たない最後のAIMessageのtextをPydanticでparseし、対応するstate fieldへ
   格納する。
6. parse失敗時の例外にはstage名とPydantic validation detailを含める。ただし画像等の
   巨大contentやsecretを例外へ複製しない。

初版ではこれらを汎用 `Stage` classや新packageへ切り出さない。2つのstageを実装して
重複の形が確定してから、必要なら小さな共通関数だけ抽出する。

完了条件:

- 後段modelが上流のHuman / AI / Tool messageを受け取る。
- 上流のSystemMessageとbudget noticeは後段model inputに入らない。
- providerが返したreasoning/content blockを加工せず保持する。
- 親 `ReconstructionState.messages` では各messageが一度だけ現れる。
- typed artifactと、その元になった最終AIMessageの両方がstateに残る。
- JSON不正時にartifactを設定したまま次stageへ進まない。

### Step 4 — semantic hypothesis / review のvertical sliceを作る

作業:

1. `semantic_hypothesis.md` と `semantic_hypothesis_review.md` を追加する。
2. `graph.py` 内でsemantic stage用のnested `StateGraph` を構築する。directoryとしての
   `subgraphs/` packageは復活させず、独立moduleへの抽出も必要になるまで行わない。

   ```text
   parent:
   START -> initialize_input -> semantic_stage -> coder -> verify_final -> END

   semantic_stage local graph:
   semantic_hypothesis -> semantic_hypothesis_review
           ^                    |
           +------ revise ------+
                                |
                              accept -> return
   ```

3. graph factoryは `semantic_agent`、`hypothesis_reviewer`、`coder_agent` を明示的に
   受け取る。
4. semantic/reviewerにはDXF・画像を調査するための必要最小toolを、coderには現在の
   code生成・verification toolを渡す。
5. semantic stageのlocal stateだけにcandidate、review、revision countを置く。
   revision上限はagent turn budgetとは別物としてconfigに記録する。
6. stage完了時、親 `ReconstructionState` へ返すのはaccepted `SemanticHypothesis` と
   current-stage message deltaだけとする。reviewとrevision countは親へ返さない。
7. accepted hypothesisをtyped JSONとtranscriptの両方でcoderへ渡す。
8. 現workflow configは単一agent baselineではなくなるため、config groupを
   `baseline.yaml` から `staged.yaml` へ改名し、`default.yaml` の選択も更新する。
   過去baselineの再現は以前のcommitを使う。

完了条件:

- accept経路が従来同様にcode生成と最終verificationまで到達する。
- revise経路がfeedbackを含めてsemantic stageへ戻り、上限で必ず停止する。
- 親stateには最新 `SemanticHypothesis` と全revision transcriptが残るが、typed reviewと
  revision counterは残らない。
- 不正なtyped outputからはcoderへ進まない。
- local semantic graphのtestでaccept/revise/revision上限/checkpoint resumeを再現できる。
- parent graphのtestでaccepted hypothesisとmessage deltaだけが引き取られる。
- `tests/zeroshot` 全体がgreenである。

ここを「stage間の型付き出力」導入の最初の完了milestoneとする。
`ReconstructionState` を定義しただけでは完了とはみなさない。

### Step 5 — operation plan / reviewを追加する

作業:

1. 実装直前に `Operation`、`OperationPlan`、`OperationReview` を `state.py` へ追加する。
2. `operation_planner.md` と `verify_operations.md` を追加する。
3. accepted semantic hypothesisと全contextをoperation plannerへ渡す。
4. semantic stageと同様にoperation plan/reviewをlocal graphへまとめ、reviewとrevision
   counterはlocal stateだけに置く。
5. `verify_operations` のaccept/revise loopを追加し、workflow-level上限を設ける。
6. 親stateへ返すのはaccepted `OperationPlan` とmessage deltaだけとする。
7. coderはaccepted `OperationPlan` を必須入力とする。
8. semantic stageへ差し戻された場合、以前のoperation plan、
   code/final verificationを明示的に `None` へ戻す。operation stageのrevisionでは
   code以降だけをinvalid化する。

完了条件:

- coderがaccepted semantic hypothesisとaccepted operation planの両方を受け取る。
- operation reviseがsemantic artifactを不必要に再生成しない。
- upstream revision後に古いdownstream artifactをroutingや最終結果へ使用できない。
- accept/revise/invalidationをgraph testで確認できる。

### Step 6 — final output criticとstage間差し戻しを完成させる

作業:

1. `OutputCritique` と `RevisionTarget`（semantic / operations / code）を
   `state.py` へ追加する。
2. `output_critic.md` を追加する。
3. coder内部の `verify_output` によるsyntax、STEP export、solid、renderの自己debug loopは
   維持する。
4. coder終了後のdeterministic final verification結果をstateへ保存し、そのattemptの
   renderとDXFをcriticへ提示する。
5. criticのtyped decisionで次へroutingする。

   ```text
   accept      -> END
   revise code -> coder
   revise ops  -> operation_planner
   revise sem  -> semantic_hypothesis
   ```

6. 差し戻しtargetより後段のartifactを明示的にinvalid化する。
7. target別またはrun全体のrevision上限を設け、循環が必ず有限になるようにする。
8. `VerifyOutputResult` と既存attempt directoryで必要情報を表せる間は、重複する
   `CodeArtifact` / `RenderArtifact` classを新設しない。不足が実証された場合だけ追加する。

完了条件:

- criticがfinal renderを実際に参照できる。
- 4つのrouting経路をscripted modelで再現できる。
- 各差し戻し先で適切なartifactだけが維持され、それ以降はinvalid化される。
- code revision後にfinal verificationとcriticが必ず再実行される。
- revision上限到達時の終了理由がstate/event logから判別できる。
- end-to-endでtyped artifacts、全transcript、最終verified outputが同時に残る。

### Step 7 — stage単位のobservabilityを整える

作業:

1. subgraph namespaceからstage別turn/token/latencyを集計する。
2. revision回数、review decision、critic target、validation failureをrun summaryへ出す。
3. 単一のrun-level値と各agentのstop reasonを混同しない。

完了条件:

- どのstageがturn/tokenを消費し、どこから何回差し戻されたかをrunごとに確認できる。
- 逐次graphの既存event記録とconsole表示が壊れていない。

### Step 8 — semantic candidateのfan-out / reduceを追加する

逐次workflowでstage contractとrevision loopを評価した後に着手する。

作業:

1. この時点で初めて `workflow/fan_out_reduce.py` を追加する。
2. semantic candidateをN体で生成し、reducerは候補を収集するだけにする。
3. fan-in後のjudge agentがtyped candidate IDを選択する。
4. branchごとにmessage/namespace/candidate IDを対応付ける。
5. 並列化前に次を解決する。
   - shared workspaceのfile collision
   - verification ID採番race
   - event loggerのactive nodeをnamespace単位にする
   - console token streamの混線

完了条件:

- N候補が欠落・上書きなしで一度だけ収集される。
- judgeが存在するcandidate IDだけを選べる。
- 各candidateのtyped outputとraw transcriptを追跡できる。
- concurrencyとcheckpoint resumeのtestがgreenである。

---

## 4. 実装中のガードレール

- 各Stepは完了条件を満たしてから次へ進む。
- production変更と、その変更を検証するtestは同じStepで入れる。
- 既存 `graph.py` は直接更新し、古いgraphのcopyを並行保守しない。
- prompt内JSON schemaとPydantic modelが二重の手書き定義にならないよう、可能なら
  modelのJSON schemaをprompt構築時に埋め込む。ただしそのためのframeworkは作らない。
- routingはfree-form文章のkeyword判定ではなくtyped enum/decisionだけを見る。
- model出力由来のrevision番号やartifact lineageを信用せず、counterとinvalidationは
  workflow codeが管理する。
- stateへ保存したtranscriptとevent logで巨大画像base64を不必要に複製しない。
- fan-outまでは逐次実行を前提とし、並列対応の修正を先回りしない。

---

## 5. 最終完了条件

v3全体の完了は、単なる `ReconstructionState` の再定義ではなく、次をすべて満たした
状態とする。

- accepted semantic hypothesis、accepted operation plan、output critiqueが親stateの
  型付きartifactとして残る。各reviewはそれを所有するlocal stage stateにだけ置く。
- 各stageが上流のtyped artifactと公開済みmessage historyの両方を受け取る。
- agentはtool callなしで自然終了でき、terminal submission toolを必要としない。
- verifier/criticの判定だけがstage間routingを決定する。
- upstream revisionでstaleなdownstream artifactが確実にinvalid化される。
- coderの自主的なcode verification loopが維持される。
- 全loopに有限のrevision/turn上限がある。
- checkpoint resume後もartifactとtranscriptが一貫し、実行中のlocal loopでは
  revision counterも一貫する。
- stage別の失敗箇所、turn、token、revision回数を観測できる。
- `tests/zeroshot` がall greenである。
