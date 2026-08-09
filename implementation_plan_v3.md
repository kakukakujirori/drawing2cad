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

### ✅ Step 2.5 — agent loopを `langchain.agents.create_agent` へ載せ替える

Step 3 の前に実施した。自前の `StateGraph`（seed / should_continue / finish）を捨て、
LangChain 公式の agent loop に載せ替える。§2.1 の module 配置は維持する。

確定した判断:

- `AgentSpec` は削除し、`create_agent_subgraph(role, model, tools, ...)` を
  Hydra の `_target_` にした。各agent blockは `_partial_: true` で、tool集合と
  prompt contextはgraphが後から渡す（§2.6のtool配分の所有権はそのまま）。
- `agent.py` は残す。`create_agent` に載せてなお config で表現できないものが
  2つあるため: turn budget（通知と打ち切りが同じ数字を共有する必要がある）と、
  `ToolFeedbackError` だけを model へ返す tool error 方針。
- turn budgetは `TurnBudget` middleware に集約した。`before_model` が通知を積み、
  `after_model` が予算超過で `jump_to: end` する。tool roundの手前で止めるので、
  未応答のtool callが残る＝BUDGET_EXHAUSTED という判定は従来と同じ。
  `ModelCallLimitMiddleware` は使わない（合成AIMessageを末尾に足すため）。
- `turns` と `stop_reason` は middleware の `state_schema` に持たせ、agent自身の
  namespaceでevent logに残す。run全体の1個の値と混同しない（§Step 7の前倒し）。
- model retryは `ModelRetryMiddleware(retry_on=(NetworkError, ProtocolError))`。
  compiled graphへの `.with_retry()` はrun全体を再実行し、tool副作用を重複させる
  ため使わない（実測: run_shell が2回実行される）。
- typed outputは `response_format` に **明示的な `ProviderStrategy`** を渡す。
  bare schemaを渡すと `model.profile` から strategy が自動選択され、profileを持たない
  backendでは `tool_choice: "any"` の強制 + terminal submission tool（§2.5が却下した形）
  へ黙って切り替わる。strategyはconfigの選択肢として持つ。
- node名は `agent` → `model` に変わった。`console.py` と `aggregate_run.py` の
  参照を更新済み。過去runのevent logとは非互換。
- `AgentState` TypedDictは削除（create_agentのstateを使う）。`StopReason` は残す。

Codexエンドポイントは client 側で `text.format: json_schema` を正しく送ることを
payload生成まで確認済み。backendが実際に honour するかは live 1回で確認する。
弾かれた場合は §2.5 どおり最終AIMessageのJSONを自前parseへ落とす。

### ✅ Step 3 — message継承とtyped result抽出を `graph.py` に実装する

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

### ✅ Step 4 — semantic hypothesis / review のvertical sliceを作る

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

#### Step 3/4 の実装で確定したこと

- **nested `StateGraph` は作らなかった。** revision loopは `semantic_stage` node内の
  `for` 文で、candidate / review / revision countはローカル変数である。stage専用の
  state型（`SemanticStageState` 等）は定義しない。graph engineに載せると型と node と
  conditional edgeが要るが、買えるのは stage 内 interrupt と図示だけで、いま対価に
  見合わない。fan-out（Step 8）で並列が要るときに作り直す。
- Step 3 の当初項目のうち2つは create_agent 移行で不要になった。過去 `SystemMessage`
  の除外は `system_prompt` が state に入らないので自動達成、最終AIMessageのPydantic
  parseは `structured_response` が肩代わりする。残った helper は `_handoff`（budget
  notice を落とす）と `_added`（id差分でstage deltaを取る）の2つだけ。
- **turn budget notice は transcript に蓄積させる（必須要件）。** `before_model` が
  state へ書き、過去の notice が履歴に残る。これは表示の好みではなく挙動要件で、
  `turn 1/30 → turn 19/30` という登ってきた梯子が見えていないと、model は 30 turn を
  すべて tool call に使い切って何も出力しないことが実測されている。現在位置だけを
  毎リクエスト見せる方式（`wrap_model_call` で request にだけ足す）は一度実装して
  却下した。**この蓄積を「transcript が汚れる」という理由で消してはならない。**
- 蓄積する以上、後段へ渡すときには落とす必要がある（§2.4）。識別方法は
  `TurnBudget` に閉じ、印は `_NOTICE_KEY`（private class 変数、provider には送られない）、
  読み手は `TurnBudget.strip_notices(messages)` とする。graph.py は印の存在を知らない。
  本文の正規表現で判定する案は却下した。落としたいのは「budget middleware が書いた」
  という出自であって書式ではなく、書式を変えたときに静かにズレる。
- **turn は transcript を数えず counter で持つ。** `after_model` が
  `state["turns"] + 1` を書く。transcript継承を入れた時点で AIMessage を数える実装は
  壊れており、reviewer が `turn 3/5` から始まって後段ほど早く budget切れになる
  （実装中に実際に踏んだ）。`candidates submitted` も同様に counter にした。
  これで開始位置の記録（`inherited`）とメッセージ走査は不要になった。
- revision 上限（`max_semantic_revisions: 10`）到達時は最後の candidate で coder へ進む。
  hypothesis が1つも得られなかった場合（agentが budget を使い切って型付き回答に
  到達しなかった場合）は coder を飛ばして `verify_final` へ抜ける。END直行ではなく
  verify_final を通すのは、完走した run が必ず1つ verification report を持つように
  するため。
- 型付き出力が無いときの扱いは2つに分ける。BUDGET_EXHAUSTED は「答える前に尽きた」
  として routing で吸収し、COMPLETED なのに型付き出力が無い／壊れている場合だけ
  run を止める。
- 親 state の `agent_turns` / `stop_reason` は「run がなぜ終わったか」を保つ。
  通常経路では coder のもの、coder を飛ばした場合だけ semantic stage のものになる。
  agent 個別の stop reason は event log の namespace 側に出る。
- checkpoint resume は node 粒度で機能することを実測した。coder が例外で落ちた後に
  同じ thread_id で再invokeすると semantic stage は再実行されず、accepted hypothesis と
  transcript が復元される。node 内部（agent の途中）からの再開はできない。
- checkpoint に自前の Pydantic artifact が載るようになったため、`runner.py` で
  `JsonPlusSerializer(allowed_msgpack_modules=[...])` を明示した。LangGraph が
  未登録クラスの deserialize を将来ブロックしても resume が壊れないようにする。

未解決として残すもの:

- **`TurnBudget` はツールを一切知らない。** 通知は `[turn k/N]` のみ。以前は
  `verify_output` の呼び出し回数を `candidates submitted: N` として出していたが、
  提出回数は `verify_output` の結果が返す `verification_id`（`001`, `002`, ...）に
  既に載っており、agent は自分の transcript から読める。middleware が特定ツールの
  名前を知る理由はない。

- **proposer / critic の対応関係（Step 5/6 の前提）。** `verify_output` は
  semantic_reviewer の相方ではない。正しい対応は次の通り。

  | | proposer | 自己点検の道具 | 判定者（権限） |
  |---|---|---|---|
  | semantic | hypothesizer | run_shell / load_image | semantic_reviewer |
  | operations | operation_planner | （持たない予定） | operation_reviewer |
  | code | coder | verify_output | output critic（Step 6） |

  `verify_output` は「compile が通り STEP が出て絵が描けた」ことしか言わず、
  図面と一致するかは判定していない。したがって proposer 側の証拠収集であり、
  判定ではない。**権限は全 stage で判定者が持つ**べきで、proposer が持つのは
  「提案を書き終える権利」だけである。code stage で今 coder が stage 終了権を
  持って見えるのは Step 6 が未実装だからで、critic が入れば自動的に semantic と
  同型になる。計画 §6 の「coder の自主的な verification loop を維持する」と
  この権限移譲は両立する（維持されるのは証拠収集であって権限ではない）。

  なお semantic と operations は**決定論的な検証手段を持たない**（そもそも
  検証方法が無い）。証拠収集の段が入るのは code stage だけである。

- **proposer / critic の切り出し。** `max_semantic_revisions` が `model_retries` と
  同列に置かれているのは、semantic stage を1つの config block として表す入れ物が
  まだ無いため。Step 5 で operation planner / reviewer を書く**前に**
  `proposer_critic.py` へ `create_proposer_critic(proposer, critic, max_revisions, ...)`
  を切り出し、semantic 段を先に移してから2例目を載せる。今切り出さないのは、
  operations 段の critic が「確定した hypothesis」を追加入力として読む必要があり、
  1例目だけからパラメータを決めると2例目を型に合わせる羽目になるため。
- **後段が継承するトークン量。** semantic段が `load_image` で読んだ画像 content block も
  そのまま coder の入力に乗る。仕様（§2.4）どおりだが、実走のトークン数を見て
  `_handoff` にフィルタを足すか判断する。
- stage内のagent別 event 属性。namespace は node 名（`semantic_stage:<uuid>`）までしか
  分からず、2体目は `('semantic_stage:<uuid>', '1')` になる。AIMessage の `name` に
  role が入っているので区別はできる。集計はStep 7で行う。

### ✅ Step 5 — operation plan / reviewを追加する

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

#### Step 5 の実装で確定したこと（レビュー前の記録）

- **`proposer_critic.py` を先に切り出してから2例目を載せた。** `graph.py` に
  ループのコピーは無く、stage は宣言 2 行になった。

  ```python
  SEMANTIC = ProposerCriticSpec(proposal=SemanticHypothesis, instructions="semantic")
  OPERATIONS = ProposerCriticSpec(proposal=OperationPlan, instructions="operations")
  ```

- **critic の判定型は共有の `Review` 1つ。** `SemanticHypothesisReview` は
  `Review` へ改名した。2例書いた結果 `decision`/`feedback` が完全に同一で、
  分ける理由が無かった。**accept / revise という語彙自体がこのループの契約**であり、
  第3の答えが要る stage はループではなく routing が要る（Step 6）。したがって
  spec から `review` フィールドを削除し、ループが `Review` を固定で使う。
- **`Operation` 型は作らない。** `OperationPlan(operations: list[str])` のみ。
  §2.3 で `PrimitiveRef` を却下したのと同じ理由で、検証手段の無い中間 IR を
  CadQuery の手前にもう1つ作ることになる。型化の判断条件は、実測で「手順の
  取りこぼし・順序入れ替え」が型で防げる形で観測されたとき。
- **指示文は stage ごとのディレクトリ**に置く（`prompts/instructions/semantic/`,
  `operations/`）。ファイル名は `propose.md` / `revise.md` / `review.md` 固定で、
  spec が持つのはディレクトリ名 1つだけ。role prompt が `{role}.md` を引くのと
  同じ規約。
- **上流の確定artifactは指示文に埋める。** `run(history, **upstream)` が
  propose / revise / review の全テンプレートへ転送する。critic は proposer の
  transcript を読まないので、hypothesis も指示文経由でしか届かない。
- `state.py` に `operation_plan` を追加。stage が artifact を出せなかった場合は
  下流を飛ばして `verify_final` へ抜ける（semantic 失敗 → operations も coder も
  走らない）。
- **coder はまだ単独ノード**のまま。Step 6 で audit ノードを足すが、proposer/critic
  ループには載らない（理由は §Step 6 の設計判断）。§Step 5 項目8（上流差し戻し時の
  下流 artifact 無効化）は差し戻し経路が存在しないため未実装。

### Step 6 — final output criticとstage間差し戻しを完成させる

トポロジー:

```text
coder → verify_final → audit ─┬→ END
                              ├→ coder
                              ├→ operations_stage
                              └→ semantic_stage
```

`verify_final` は終端ノードをやめ、**監査の証拠を作るノード**になる。

作業:

1. `Audit` を `state.py` へ追加する。

   ```python
   class Audit(BaseModel):
       """The final judgement on the built model: accept it, or name the stage to go back to."""
       decision: Literal["accept", "redo_code", "redo_operations", "redo_semantics"]
       feedback: str
   ```

2. `roles/output_auditor.md` を追加する。`$output_schema` は他のroleと同じ規約で埋める。
3. coder内部の `verify_output` によるsyntax、STEP export、solid、renderの自己debug loopは
   維持する。
4. `verify_final` の結果（`last_verification`）を監査への指示文にパスとして埋め、
   criticは `load_image` で見る。**criticに `verify_output` は持たせない。**
5. `audit` ノードの `Audit` で routing する。行き先は上のトポロジーの4つ。
6. 監査差し戻しカウンタを `state.py` へ追加し、graph.py が上限到達時に `verify_final` を
   経て `END` へ抜ける。stage内の `max_revisions` とは別物。
7. 再入した stage は1周目から revise 指示で始める。`run(history, feedback=..., **upstream)`
   と引数を1つ足すだけで、ループ内部の propose/revise 分岐をそのまま使う。
8. `VerifyOutputResult` と既存attempt directoryで必要情報を表せる間は、重複する
   `CodeArtifact` / `RenderArtifact` classを新設しない。不足が実証された場合だけ追加する。

完了条件:

- criticがfinal renderを実際に参照できる。
- 4つのrouting経路をscripted modelで再現できる。
- 上流へ差し戻した後、下流のartifactが再生成されるまで参照されない。
- code revision後にfinal verificationとcriticが必ず再実行される。
- 監査差し戻し上限到達時の終了理由がstate/event logから判別できる。
- end-to-endでtyped artifacts、全transcript、最終verified outputが同時に残る。

#### Step 6 の設計判断（実装前に確定）

- **code stage は proposer-critic ループでは実装しない。** criticがfinal verificationを
  見る以上、proposerとcriticの間にグラフノード（`verify_final`）が挟まる。
  `create_proposer_critic_loop` は2体を素のPythonで連続呼び出しする作りなので、
  間にノードを置けない。したがって親グラフの3ノード＋条件付きエッジになり、
  **`proposer_critic.py` は Step 6 で無変更**。
  却下案: ループに `verify` コールバックを持たせる。3 stage中1つしか使わない分岐が増え、
  `last_verification` を親stateへ書く責務もループへ漏れる。
- **`ProposerCriticSpec` に `review` は戻さない。** `Audit` を読むのは graph.py だけで、
  ループは `Review` を固定で持つままでよい（Step 5 の判断が有効なまま）。
  同じ理由で `Review` と `Audit` に共通の基底クラスも作らない。共有したいのは
  `feedback` と validator の数行だけで、継承階層に見合わない。
- **routing は graph.py が持つ。** 「どのstageへ戻れるか」はstageの性質ではなく
  グラフの性質で、同じcode stageでも上流構成が変われば戻り先が変わる。
  サブグラフから `Command(goto=..., graph=PARENT)` で親へ飛ばす案は、
  トポロジーの決定権が2ファイルに分裂するため却下。
- **クラス名と値が差し戻し先を名指しする。** `CodeReview` では「code stageのreview」
  としか読めず、最終監査の一番重要な観点（どの段が悪いか）が名前に出ない。
  `Audit` + `redo_operations` / `redo_semantics` とし、値そのものを行き先にする。
  `decision` と「誰が直すか」を別フィールドに分けると `accept` + `redo_semantics`
  のような無意味な組が表現できてしまうので、単一 enum に畳む。
- **`redo_semantics` は最初から入れる。** `redo_operations` と実装上の非対称はゼロで、
  enum値1つと条件付きエッジの行き先1つが増えるだけ。「最上流へ戻して仕事を捨てる
  モデルが出るのでは」というのは実験上の懸念であって、実装を削る理由にならない。
- **§Step 5 項目8（上流差し戻し時の下流artifact無効化）は不要になる。** 最も上流の
  誤ったstageへ戻せば、前向きエッジが下流を順に再実行してartifactを上書きする。
  再入したstageが何も産まなければ `_settled` が `verify_final` へ逃がすので、
  古い値が読まれる経路が存在しない。明示的な `None` 代入は書かない。
- **監査差し戻し上限は `redo_code` だけでも必要。** `audit → coder → verify_final → audit`
  が閉路になるため、上流へ戻さなくても無限に回りうる。`max_revisions` は
  stage内のラウンド数なのでこの閉路を縛れない。
- **criticに `verify_output` を持たせない理由**は3つ。criticのビルドがattempt
  ディレクトリを汚さない。監査対象が実際に採点される成果物そのものになる。
  criticのターンが検証待ちで潰れない。

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
