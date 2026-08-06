# Multi-agent 化 実装計画（v2）

`implementation_plan_v1.md`（Phase 1-7、単一agent baselineの完成まで）の続き。
Phase 8「並列agent」のうち、**逐次multi-agent**を先に実装する。

進行に伴い、各Stepの完了時に決定事項と観測結果をこのファイルへ追記する。

---

## 1. 動機

baselineは単一のReAct agentが「図面理解 → CadQuery生成 → 自己検証」を1つのcontextで完結させている。過去のGemini実験のCoTを読むと、推論は6段階に分解できる傾向があった。

1. View Registration（DXFのfront/top/rightが3D空間でどう対応するか）
2. Semantic Hypotheses（各3D部分パーツがflange/boss/holeといった仮説）
3. Verify Hypotheses（仮説を図面・画像と照合して検証・棄却）
4. Convert to Operations（採択仮説を生成するための主操作と順序）
5. Convert to Code（操作列をCadQueryコマンドとして記述）
6. Verify Codes（実行検証とレンダリングの一致確認）

段階ごとに専用promptとtool集合を持つagentへ委譲すると精度が上がるのではないか、というのが**未検証の仮説**である。

**どの段階でエラーが起きやすいかの定量的観察はまだ無い。** したがって本計画の第一の成果物は「段階別のturn/token/失敗箇所を観測できる状態」であり、精度向上はその次に来る。context を切ることは情報を失うことでもあり、悪化しうる。

### 構造的な障害

1. **system promptが1本しか存在しない** — `messages.py` に `MessageBuilder` の lambda default として埋まっている。YAMLでcallableは書けないのでHydraから差し替える手段が無い。
2. **agent loopが再利用単位になっていない** — `graph.py` の `agent ⇄ tools` loop は graph factory に直書き。

---

## 2. 方針

### 2.1 グラフ構築はPython、YAMLは選択と設定のみ

**グラフ構造をYAMLで表現しようとしない。** 現行時点で `agent ⇄ tools` は既に双方向で線形ではなく、将来のfan-outも見えている。線形チェーンをYAMLで書けるようにすると `revise_to` のような特殊フィールドが増え、YAMLがPythonの劣化DSLになる。

- **topology は `workflow/graph_<name>.py` に置く**（`implementation_plan_v1.md`「グラフのversioning」の規則そのまま）
- **YAML は「どの `.py` を使うか」と「各agentのprompt/tool/予算」だけを持つ**
- 新しいtopologyが欲しければ新しい `.py` + 新しい `configs/workflow/<name>.yaml`。config group名がそのままablation表の行ラベルになる

### 2.2 現行graphを凍結せず、部品へ分解して組み直す

新graphのために現行をコピーすると必ずドリフトする。**現行graph自身を再利用可能な部品で組み直し、その部品が十分であることを現行で証明する。**

抽出する部品は3つで、いずれも**topology非依存**である（`graph_staged.py` の topology が賭けとして外れても残る）。

| 部品 | 無いと何が起きるか |
|---|---|
| `prompts/` + `PromptTemplate` | agentが2体以上になった瞬間にsystem promptを書く場所が無い |
| `workflow/toolbelt.py` | agentごとにtool集合を変えられない |
| `workflow/agent.py::create_agent_subgraph` | agent loopを2箇所目に置けない |

### 2.3 「壊さない」の定義

| 層 | 内容 | 扱い |
|---|---|---|
| **A: モデルから見た挙動** | モデルへ渡るmessage列（system prompt本文含む）、bindされたtool schema、tool callの順序と引数、tool result、artifact、`last_verification` / `stop_reason` | **完全一致を機械的に保証**（§3） |
| **B: event logの形** | `events.jsonl` の `namespace`、wrapper node分の `node_started`/`node_finished` | Step 4で**変わる**。影響調査済み（§2.4） |

層Aが不変なら「実行フローは一切改変されていない」と言える。

### 2.4 層B（event log）の変化と、その影響の調査結果

`agent ⇄ tools` をsubgraph化すると、そのイベントの `namespace` が `[]` から `["<wrapper node名>:<task_id>"]` になる。実際に追った結果:

- **`normalizer.py` の `caller` 判定は無影響。** `self._active_node == "tools"` で見ており、**subgraph内のnode名は `agent` / `tools` のまま**保たれる。
  - これが「flat graphでnode名に `vr_` 等のprefixを付ける」案を採らない理由でもある。prefixを付けると全tool callが `caller="workflow"` になり監査が壊れる。
- **`aggregate_run.py` の token/turn 集計は無影響。** `data["node"] == "agent"` で見ているため。過去runとの数値比較は維持される。
- **`node_ms`** はwrapper nodeのキーが1つ増えるだけ。既存キーの値は不変。
- **subgraphのeventは親のstreamへ流れる。**
  - `stream_events(version="v3")` は内部で `astream(..., subgraphs=True)` を呼ぶ（`langgraph/pregel/main.py:3609`）
  - 組み込みの `ValuesTransformer` 等が持つ `if params["namespace"] != self._scope_list: return` という scope filter を、`RunEventTransformer` は**持っていない**。よってsubgraphのeventも全部受け取り `namespace` に記録する
  - namespaceの各セグメントは `"<node名>:<task_id>"` 形式（`langgraph/stream/transformers.py::_parse_ns_segment`）。stage帰属が無料で手に入る

**結論: subgraph化で壊れるものは無く、増えるのはstage帰属の情報だけ。**

---

## 3. Step 0 — characterization harness ✅ 完了

### 目的

以降の全変更が「壊していない」ことを主張するための唯一の根拠。これ無しにリファクタへ入ると、既存パイプラインを壊さないことを保証できない。

### 成果物（一時的。gitignore済み、Step 4完了後に `_refactor/` ごと削除）

- `_refactor/characterize_baseline.py` — 記録・照合ハーネス
- `_refactor/baseline_snapshot.json` — 現行の挙動スナップショット

```bash
python _refactor/characterize_baseline.py record   # スナップショット作成（済）
python _refactor/characterize_baseline.py check    # 差分検出。各Step後に必ず実行
```

### 何を固定しているか

`MessageBuilder.build_initial()` から `graph.invoke()` の終端までを、runnerと同じ手順で再現し、以下を正規化してJSONに落とす。

- `initial_messages` — **system prompt本文（1620字）を含む**
- `bound_tools` — `bind_tools` へ渡した3つのtool名と**description全文**（tool descriptionはprompt surfaceである）
- `agent_inputs` — 各turnでモデルへ渡ったmessage列（4turn）
- `final_messages` — 最終transcript（system, human, ai, tool×3, ai の9通）
- `agent_turns` / `stop_reason` / `last_verification` / `attempts/` のディレクトリ一覧
- `topology_mermaid` — `graph.get_graph().draw_mermaid()`

シナリオは `access_render3d`/`feedback_render3d` を `path` と `image` の2通りで回す。ここが `messages.py` のうちリファクタで触る部分（pathを出すか base64 image blockを出すか）を分けているため。

scripted modelは `run_shell`（model.py書き込み）→ `load_image` → `verify_output` → 終了、と全toolを1回ずつ踏む。

### 決定: OCCを2箇所だけstubする

`CadQueryExecutor.execute` と `StepRenderer.render` のみを決定的な値へ差し替える。理由は、実CadQuery/OCCを通すとバージョン依存の文字列がstderrに載りスナップショットが環境依存になるため。

**その上位（`verify_output` 自身のロジック、`FeedbackManifest`、`MessageBuilder.build_feedback`）は実物が動く。** ここがリファクタ対象なので、stubするのは境界の外側だけに留めた。既存testが `create_verify_output_tool` ごとstubしている（`test_graph.py:201`）のとは意図的に層を変えている。

### 正規化（非決定要素の除去）

- message id / content block の `id`・`index` を除去
- `base64` は先頭16桁のsha256へ
- `last_verification.source` はsha256へ
- 一時workspaceのhost pathを `<WORKSPACE>` へ置換（sandbox側の `/work` パスは安定なのでそのまま残す）

### 検証結果

- 連続2回の `check` が差分なし（決定的）
- `messages.py` のsystem promptを1文字変えると exit 1 で差分を出す（**ガードが実際に機能することを確認済み**）。確認後 `git checkout` で復帰し、再度緑

---

## 4. Step 1 — `messages/` パッケージ化と `prompts/` の追加 ✅ 完了

挙動不変。2コミットに割った（import の付け替えと新機構の導入を分離し、壊れたときにどちらが原因か即座に分かるようにするため）。

### Step 1a — `zeroshot/pipeline/messages/` パッケージ化

```
zeroshot/pipeline/messages/
├── __init__.py     # MessageBuilder, InputManifest, FeedbackManifest, PromptTemplate を re-export
├── manifest.py
├── builder.py      # messages.py から改名
└── prompts/
```

**決定: `manifest.py` と `messages.py` を1パッケージにまとめる。**

当初は「`manifest` は prompt の住人ではない」として反対したが、それは親パッケージ名を `prompts/` とする案への反対であり、グルーピング自体への反対ではなかった。親を `messages/` にすると成立する。

根拠は**凝集度**である。`manifest.py` は `builder.py` に食わせるために存在する。`InputManifest` / `FeedbackManifest` は「モデルへ提示される予定のartifact集合」という定義そのものであり、`FeedbackManifest` の「path または欠損理由のどちらか一方」という不変条件は、MessageBuilder が何を見せるか決められるようにするために存在する。2ファイルで1つの関心事である。

（「manifest/messages は runner.py や sandbox.py より責務が軽いから」という理由付けは採らない。サイズでは `messages.py` 12.5KB > `runner.py` 8.5KB > `sandbox.py` 7.1KB であり、`messages.py` が最大である。）

パッケージの charter: **「モデルが見るものを構築する層」**（manifest = artifact在庫、builder = content block化、prompts = テキスト）。

- **`messages.py` → `builder.py` 改名** — `messages/messages.py` の重複を避ける。`verification/run_cadquery.py`、`event_logging/normalizer.py` と同じで、モジュール名はパッケージ名の繰り返しではなく何をするかを表す
- **外部からの import はパッケージ経由に統一**（`from zeroshot.pipeline.messages import ...`）。外から見える契約を `__init__.py` の `__all__` 1箇所に集約するため。内部モジュールを直接指す import が散ると、中身を再配置するたびに全消費者を触ることになる。`builder.py` の中だけは `messages.manifest` を直接指す（`__init__` 経由にすると循環になるため）
- `__init__.py` の docstring に manifest / builder が分かれている理由（manifest は *host* path を持ち、モデルが受け取るものへ変換してよいのは MessageBuilder だけ）を残した。v1 が load-bearing だと書いている境界なので、パッケージにまとめた以上その理由を入口に置く

移行コストは小さかった。`from zeroshot.pipeline.messages import MessageBuilder` の10箇所と `default.yaml` の `_target_: zeroshot.pipeline.messages.MessageBuilder` は**無変更で通る**（`instantiate` で実測確認）。変わったのは `pipeline.manifest` の9箇所だけ。

### Step 1b — `PromptTemplate` と `prompts/coder.md`

この時点では誰も使わないので production の挙動は 100% 不変。

決定事項:
- **`string.Template`（`$var`）を使い `str.format`（`{var}`）を使わない** — prompt本文にCadQueryのコード（`{}` を含む）を書いても壊れないため
- **`substitute()` を使い `safe_substitute()` を使わない** — 埋め忘れると `$output_path` という文字列がそのままモデルへ届く。落ちたほうがよい
- **`name` はpackage内の名前解決**（絶対pathも受け付ける） — **Hydraはjob実行時にcwdを変える**ので、YAMLに相対pathを書くとrun時に解決できない。名前解決なら罠自体が消える
- **存在しないpromptは構築時に `ValueError`** — run途中で落とさない
- **テキストを `.md` へ出す理由** — (1) promptはこの研究の主な実験変数でありgit diffが読める価値が大きい、(2) f-string内では `{}` と `"""` のescapeが地獄になる、(3) 6本の長文をPythonに置くとそのファイルがテキストの塊になる。代償（pyreflyがplaceholderを見ない）はStep 0のスナップショットで補う

#### `render()` が `.strip()` する理由（実装中に判明）

現行の system prompt は**末尾に改行を持たない**。lambda の括弧が

```python
cleandoc(f"""...""".strip() + "\n")
```

となっており、`+ "\n"` が `cleandoc()` の内側にあるため cleandoc が再び落としている。

一方 `.md` ファイルは末尾に改行を持つのが自然で、エディタやフォーマッタが勝手に付け外しする。ファイル末尾の改行有無が**モデルへ送るバイト列を変えてしまう**のは事故なので、`render()` の最後で `.strip()` する。これにより「ファイルが改行で終わるか」（ファイルの作法）と「prompt テキスト」（内容）が分離される。

#### `coder.md` は手で書き写さず機械生成した

転記ミスは Step 2 まで発覚しない可能性があるため、現行 lambda にサニタイズ用 sentinel を渡して出力を得て、sentinel を `$output_path` / `$verification_dir` に置換して生成した。本文に `$` が含まれないことも事前に確認済み（含まれていれば `$$` へのエスケープが必要だった）。

### 検証結果

- `PromptTemplate("coder").render(...)` が現行 lambda の出力と**完全一致**（2通りの置換値で確認: 1620字 / 1637字）
- `_refactor/characterize_baseline.py check` → **baseline unchanged**
- `pytest tests/zeroshot` → **370 passed**（`tests/zeroshot/test_prompts.py` の8件を追加）
- `ruff check` / `ruff format --check` ともに clean

> `ruff check` を repo 全体にかけると32件出るが、すべてレガシーの SFT 側テスト（`tests/data/`、`tests/metrics/` など）の既存分で、本作業とは無関係。

### 残っている整理（任意、独立して実施可）

production が `messages/` パッケージになったので、`tests/zeroshot/test_messages.py` と `test_manifest.py` を `tests/zeroshot/messages/` へ移すとテスト木がミラーになる（`tools/`、`workflow/`、`verification/`、`event_logging/` は既にミラー）。Step 1b の diff を濁らせないため今回は見送った。

## 5. 以降のStep

各Stepの完了時に、決定事項と `check` の結果をこのファイルへ追記する。

### Step 2 — system promptの所有者を `MessageBuilder` から graph へ

現行は**runnerが `[SystemMessage, HumanMessage]` を組み立ててgraphへ渡している**。この構造では、agentごとに異なるsystem promptを持つgraphを書けない。runnerはgraphの役割構成を知らないし、知るべきでもない。

また `MessageBuilder` はrun全体で1つ共有されるので、N体のagentのpromptを持つ器として構造的に不適切。`MessageBuilder` の責務は「モデルに**何のartifactを見せるか**」に戻す。

- `messages.py`: `system_prompt` field削除、`build_initial` → `build_input_message`（`HumanMessage` 単体を返す）
- `runner.py`: `input_message` をgraph factoryへ渡す（実行環境なのでfactory引数。graph固有設定ではないので `_partial_` には入らない）。初期stateは `messages=[]`
- `graph.py`: `prepare` node を足してSTARTを差し替え、system promptとinput messageをそこで積む

これでsystem promptは今まで通り `messages` に載り `events.jsonl` にも残る（監査要件を維持）。

### Step 3 — tool生成を `toolbelt.py` へ（純粋な移動）

`build_toolbelt(...) -> tuple[dict[str, BaseTool], BaseTool]`。`serialize_output=False` の instance を dict に入れず戻り値を分けることで、生の `VerifyOutputResult`（`source` 本文を含む）がモデルへ届く経路を構造的に塞ぐ。

### Step 4 — agent loopをsubgraph化し、**現行graphをそれで書き直す**

`create_agent_subgraph(model, tools, max_turns, ...)`。node名は必ず `"agent"` と `"tools"`（§2.4）。

現行graphは `START → prepare → agent_loop → verify_final → END` になる。`agent_loop` wrapper が子のmessagesをそのまま親へ返すので、`verify_final` が読む `state["messages"][-1]` は現行と同一になり `stop_reason` の判定も無変更。

**この時点で `AgentSpec` は作らない。** call siteが1つしかないものを束ねる理由が無い。Step 6で6箇所になったときに導入する。

**ここまでで現行パイプラインは「再利用可能な部品で組まれた、挙動が完全に同じもの」になる。**

### Step 5（任意）— 対照実験: 単一agentのまま段階を踏ませる

「段階を踏むこと」の効果と「contextを分けること」の効果は別物である。前者だけで足りるならmulti-agent化は複雑さに見合わない。

Step 4の後なら新規40行で測れる（前だとloopのコピペになる）。`graph_guided.py`: agentがtool callをやめるたびに次段階の指示 `HumanMessage` を注入し、`stage_index` を進める。contextは1本のまま。

### Step 6 — multi-agent graph

`AgentSpec`（prompt所有者 = サブエージェントの定義）、`StageReport` + `merge_reports` reducer、`graph_staged.py`。

- **reducerが必須である理由**（投機的一般化ではない）: LangGraphはreducerの無いkeyを**丸ごと置換**する。差し戻しループでcoderが2回目のreportを返すと、他の全stageのreportが消える
- **stage間受け渡しは最終AIMessage本文**。これはlossyである。緩和は2つ: (1) 全stageが `SandboxWorkdir` を共有するので「reportは要約、実データは `/work/stages/<name>/` のファイル」という分担をpromptで指示する、(2) 全stageが `input_message` を毎回受け取るので元のDXFとrenderへは常に戻れる
- **step 5と6を分けない**: coderは `verify_output` を持ったまま自己debug loopを維持する（baselineのexecution_success 18/20を支えている）。その上に「入力図面と最終renderを見比べる」だけのcritic agentを載せる。coderの自己debug（構文エラー、STEP出力失敗）とcriticの見る対象（幾何的差分）は本当に違う

### Step 7 — 集約のstage対応

`aggregate_run.py` は `data["node"] == "agent"` で数えており、stagedでは全stage合算になる。`event["namespace"]` からstage名を取り、stage別のturn/tokenを出す。`node_ms` のキーも `(namespace, node)` 相当へ。

**どのstageが予算を溶かしているかは本計画の第一の観測目的**なのでoptionalではない。

`normalizer.py` は改修不要。ただし `_active_node` がnamespaceをまたいで1つしかないため**逐次実行前提**である旨をコメントで明記する。並列fan-outではここが最初に壊れる。

---

## 6. 保留

- 仮説のfan-out（並列agent）とrun-level `BudgetPolicy` — `normalizer.py` の `_active_node` が最初に壊れる
- stage出力の構造化（`submit_<stage>` toolによるschema強制） — free textで足りるか実測してから。`AgentSpec.tools` に1つ足すだけで移行できる形にしてある
- stageごとの独立 `SandboxWorkdir` — 逐次実行では共有のほうが有利。fan-outと同時に検討
- stageごとのmodel差し替え — `AgentSpec.model` のseamだけ用意し、実際には繋がない
