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
| `workflow/subgraphs/agent.py::create_agent_subgraph` | agent loopを2箇所目に置けない |

### 2.3 「壊さない」の定義

| 層 | 内容 | 扱い |
|---|---|---|
| **A: モデルから見た挙動** | モデルへ渡るmessage列（system prompt本文含む）、bindされたtool schema、tool callの順序と引数、tool result、artifact、`last_verification` / `stop_reason` | **完全一致を機械的に保証**（§3） |
| **B: event logの形** | `events.jsonl` の `namespace`、wrapper node分の `node_started`/`node_finished` | Step 3で**変わる**。影響調査済み（§2.4） |

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

### 成果物（一時的。gitignore済み、リファクタ完了後に `_refactor/` ごと削除）

- `_refactor/characterize_baseline.py` — 記録・照合ハーネス
- `_refactor/baseline_snapshot.json` — 現行の挙動スナップショット

```bash
python _refactor/characterize_baseline.py record   # スナップショット作成（済）
python _refactor/characterize_baseline.py check    # 差分検出。各Step後に必ず実行
```

### 何を固定しているか

runnerと同じ手順で入力を組み立ててから `graph.invoke()` の終端までを再現し、以下を正規化してJSONに落とす。

- `bound_tools` — `bind_tools` へ渡した3つのtool名と**description全文**（tool descriptionはprompt surfaceである）
- `agent_inputs` — 各turnでモデルへ渡ったmessage列（4turn）。**system prompt本文（1620字）はここに入る**
- `final_messages` — 最終transcript（system, human, ai, tool×3, ai の9通）
- `agent_turns` / `stop_reason` / `last_verification` / `attempts/` のディレクトリ一覧
- `topology_mermaid` — `graph.get_graph().draw_mermaid()`

シナリオは `access_render3d`/`feedback_render3d` を `path` と `image` の2通りで回す。ここが `messages.py` のうちリファクタで触る部分（pathを出すか base64 image blockを出すか）を分けているため。

scripted modelは `run_shell`（model.py書き込み）→ `load_image` → `verify_output` → 終了、と全toolを1回ずつ踏む。

### 決定: OCCを2箇所だけstubする

`CadQueryExecutor.execute` と `StepRenderer.render` のみを決定的な値へ差し替える。理由は、実CadQuery/OCCを通すとバージョン依存の文字列がstderrに載りスナップショットが環境依存になるため。

**その上位（`verify_output` 自身のロジック、`FeedbackManifest`、`MessageBuilder.build_feedback_blocks`）は実物が動く。** ここがリファクタ対象なので、stubするのは境界の外側だけに留めた。既存testが `create_verify_output_tool` ごとstubしている（`test_graph.py:201`）のとは意図的に層を変えている。

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

### 残っている整理

production が `messages/` パッケージになったので、テスト木も `tests/zeroshot/messages/` へミラーさせる（`tools/`、`workflow/`、`verification/`、`event_logging/` は既にミラー）。Step 1b の diff を濁らせないため見送っていたが、**Step 2 で解消済み**（`test_builder.py` / `test_manifest.py` / `test_prompts.py`）。

## 5. 以降のStep

各Stepの完了時に、決定事項と `check` の結果をこのファイルへ追記する。

### Step 2 — system promptを runner / `MessageBuilder` から取り上げる ✅ 完了

現行は**runnerが `[SystemMessage, HumanMessage]` を組み立ててgraphへ渡している**。この構造では、agentごとに異なるsystem promptを持つgraphを書けない。

取り上げる理由は2つの持ち主それぞれで別:

- `runner` — runのオーケストレータであり、graphの役割構成を知らないし知るべきでもない
- `MessageBuilder` — run全体で1インスタンス共有なので、N体分の役割promptを構造的に持てない。責務は「モデルに**何のartifactを見せるか**」に戻す

#### 所有と組み立ては別の話

| | 何を決めるか | 最終的な持ち主 |
|---|---|---|
| **所有** | どのpromptがこの役割を定義するか | **agent**（Step 3の `create_agent_subgraph`、Step 5の `AgentSpec`） |
| **組み立て** | `SystemMessage` を transcript の先頭に置く | transcript を作る場所 |

Step 2の時点では agent オブジェクトがまだ存在しないので、置ける場所は graph しかない。Step 3で**同じパラメータの組み立て地点が一段内側へ移る**（`prepare` node の中身が subgraph の `seed` node になる）。導入して削除するのではない。

```python
# Step 2
create_reconstruction_graph(system_prompt=PromptTemplate("coder"), input_message=...)
  └ prepare node が [SystemMessage(render), input_message] を積む

# Step 3 — prepare は消え、subgraph の内側へ
  └ create_agent_subgraph(model, tools, prompt=system_prompt)
       └ seed node が [SystemMessage(render), *task] を積む

# Step 5
AgentSpec(name="coder", prompt=PromptTemplate("coder_staged"), tools=(...), max_turns=25)
```

#### 監査上の制約（Step 3の設計を決める）

**node の戻り値以外で `messages` に入ったものは `events.jsonl` に残らない。** 実装で確認した事実:

- `input` event は `_input_written` ガードで**最初の1回だけ**書かれる（`normalizer.py:102`）。子subgraphの初期stateは記録されない
- `updates` ハンドラは node が `messages` を返すたびに `message` event を出す（`normalizer.py:163-177`）

したがって `subgraph.invoke({"messages": [SystemMessage(...), ...]})` と呼び出し側で組み立てると、**その SystemMessage は監査ログに一切現れない**。v1 §7「system/user/tool message の再現に必要な payload を保存する」を満たせないので、呼び出し側で組み立てる案（`Agent.run(task)` のようなヘルパを含む）は採らない。

加えて `add_messages` は追記なので、呼び出し側が `messages` を先に埋めると seed が返す SystemMessage が**後ろ**に付いてしまう。

よってStep 3のsubgraphは、入力を `messages` ではなく **`task`（呼び出し側が与える human 側メッセージ列）** で受け、自分の `seed` node が `[SystemMessage(自分のprompt), *task]` を `messages` へ書く。「**agent は task を受け取り transcript を返す。system prompt は agent 自身の性質**」という形になる。

`task` は system prompt の置き換えではない。system prompt は今まで通り `messages[0]` の `SystemMessage` として載り、内容も位置も現行と同一である。Step 9では `task` が `[input_message, 上流stageのreport]` になる。

#### Step 2 の変更

- `messages/builder.py`: `system_prompt` field削除、`build_initial` → `build_input_message`（`HumanMessage` 単体を返す）
- `runner.py`: `input_message` をgraph factoryへ渡す（実行環境なのでfactory引数。graph固有設定ではないので `_partial_` には入らない）。初期stateは `messages=[]`
- `graph.py`: `prepare` node を足してSTARTを差し替え、system promptとinput messageをそこで積む

`input` event の `messages` は空になるが、直後の `prepare` の `message` event が両方を記録するので情報は失われない。

#### 実装結果

- `MessageBuilder` から `system_prompt` field と `SystemMessage` / `cleandoc` / `Callable` の import が消え、責務が「モデルに何のartifactを見せるか」だけになった
- `build_initial(manifest, workdir, output_filename, verification_dirname) -> list[BaseMessage]` → `build_input_message(manifest, workdir) -> HumanMessage`。`output_filename` / `verification_dirname` は system prompt の render にしか使われていなかったので引数ごと消えた
- `graph.py` に モジュール定数 `DEFAULT_SYSTEM_PROMPT = PromptTemplate("coder")` を置き、factory 引数の既定値にした。引数既定値に直接 `PromptTemplate("coder")` と書くと ruff B008 に触れる（`PurePosixPath` は ruff の immutable 既定リストに入っているので通るが、自作クラスは入らない）。定数にすると「このgraphの既定の役割は coder」が1箇所に明示される副次的な利点もある

#### 検証結果

`_refactor/characterize_baseline.py` を新APIへ追随させ、スナップショットのキー単位で差分を取った。**層A（モデルから見た挙動）は完全に不変**であることを機械的に確認した。

| キー | 結果 |
|---|---|
| `agent_inputs`（全4turnでモデルへ渡ったmessage列。system prompt本文を含む） | **same** |
| `final_messages` | **same** |
| `bound_tools`（3つの tool 名と description 全文） | **same** |
| `agent_turns` / `stop_reason` / `last_verification` / `attempt_dirs` | **same** |
| `topology_mermaid` | CHANGED（`prepare` node と `START → prepare → agent` の2辺のみ。意図通り） |
| `initial_messages` | 削除（harness側のキー。graph が会話を開くようになったので `agent_inputs[0]` が同じものを押さえている） |

差分確認後に再recordした。その他:

- `pytest tests/zeroshot` → **370 passed**
- `ruff check` / `format --check` clean

#### 既存testの追随（34件）

- `tests/zeroshot/messages/test_builder.py` — `build_initial(...)[1]` → `build_input_message(...)`、戻り値が2通のlistから `HumanMessage` 単体へ
- `tests/zeroshot/workflow/test_graph.py` — 全 `create_reconstruction_graph` 呼び出しに `input_message=` を追加。**taskの文言がinvoke地点からfactory地点へ移動した**のが本質的な変化。transcript先頭に `SystemMessage` が入るので索引ベースのassertionを+1
- `tests/zeroshot/test_runner.py` — factory kwargs集合に `input_message` を追加。budget notice を集めるtestは、`prepare` が書く input message も human turn なので `[turn ` 接頭辞で絞るようにした

`tests/zeroshot/test_prompts.py` は `tests/zeroshot/messages/` へ移動し、テスト木を production のパッケージ構成にミラーさせた（Step 1 の「残っている整理」を解消）。

#### 追加: `build_feedback` → `build_feedback_blocks`

`build_input_message` との対称性から `build_feedback_message` にする案があったが、**非対称であることが正しい**と判断した。

| | 何を作るか | 最終的にどうなるか |
|---|---|---|
| `build_input_message` | 会話の**1ターン** | graph が `messages` へ積む |
| `build_feedback_blocks` | **tool result のペイロード** | `verify_output` が返し、ToolNode が `ToolMessage` にする |

v1 が明示的に決めた設計（*`verify_output` は `load_image` と同じく content block の list を返し、ToolNode がそれをそのまま `ToolMessage` にする*）の通り、feedback は最終的な会話の中で `HumanMessage` になることが**一度もない**。`build_feedback_message` と名付けると実在しない human turn があるかのように読める。

実装上も、唯一の呼び出し元 `verify_output.py::_message_blocks` が `HumanMessage(content_blocks=blocks).content_blocks` で作った直後に剥がしていた。この往復が identity であることを実測で確認した上で（`id` は `create_text_block` が付与するもので、包み直しでは何も起きない）、`HumanMessage` を作るのをやめて `list[ContentBlock]` を直接返すようにした。characterization が `agent_inputs` / `final_messages` 不変を示しており、tool result のバイト列は変わっていない。

### Step 3 — agent loopをsubgraph化し、**現行graphをそれで書き直す** ✅ 完了

`workflow/agent.py` に `create_agent_subgraph` と `AgentState` を新設し、`graph.py` をそれで組み直した。

```
workflow=baseline:  START → agent_loop → verify_final → END
                             └ subgraph: START → seed → agent ⇄ tools → END
```

#### `AgentState` は `task` と `messages` を分ける

```python
class AgentState(TypedDict):
    task: list[BaseMessage]  # 呼び出し側が与える依頼
    messages: Annotated[list[BaseMessage], add_messages]  # agentが作る transcript
    turns: NotRequired[int]
```

`seed` node が `[SystemMessage(自分のprompt), *task]` を `messages` へ書く。system prompt は agent の性質なので agent が積む。理由は §Step 2「監査上の制約」の通り（呼び出し側が `messages` を先に埋めると add_messages が追記なので system prompt が後ろに回る／node の戻り値以外は event log に残らない）。

`prompt.render()` は**graph構築時に1回**行う。placeholder の渡し忘れが run 途中ではなく組み立て時に落ちる。

#### 決定: `exhausted` フィールドは作らない

当初案では budget 切れを `AgentState.exhausted` で持つ予定だったが、**最後の message に tool call が残っているかで導出できる**ので追加しない。v1 §6.1「将来必要になるかもしれないだけの field を state へ先回りして追加しない」。

#### 決定: 親は transcript を引き取らない

最初の実装では `agent_loop` が子の `messages` を丸ごと親へ返していた。これは**event log で全メッセージが二重記録される**（子の node ごとの `message` event と、親の一括 `message` event）。オフラインで会話を再生する読み手には二重に見える。テストが budget notice の3重カウントで検出した。

`agent_loop` は子の transcript を**読むが返さない**。返すのは `agent_turns` と `stop_reason` だけ。

```python
def agent_loop(state):
    result = coder.invoke({"task": [input_message]})
    last_message = result["messages"][-1]
    return {
        "agent_turns": result.get("turns", 0),
        "stop_reason": BUDGET_EXHAUSTED
        if (isinstance(last_message, AIMessage) and last_message.tool_calls)
        else COMPLETED,
    }
```

これに伴い `stop_reason` の決定が `verify_final` から `agent_loop` へ移った。「なぜ agent が止まったか」は agent loop の事実であって検証の事実ではないので、置き場所としてもこちらが正しい。`verify_final` は最終検証だけを行う。

Step 9 でも親は6本の transcript を抱えず `reports` だけを持つので、この形がそのまま効く。

#### event log の実測確認（層Bの変化）

実際に run を回して `events.jsonl` を確認した。予測はすべて当たった。

```
node_started     -                            agent_loop
node_started     agent_loop:0b277d06...       seed
message          agent_loop:0b277d06...       node=seed types=['system', 'human']
node_started     agent_loop:0b277d06...       agent
message          agent_loop:0b277d06...       node=agent types=['ai']
node_started     agent_loop:0b277d06...       tools
tool_started     agent_loop:0b277d06...       run_shell caller=model
...
stop_reason      -                            agent_loop
node_started     -                            verify_final
tool_started     -                            verify_output caller=workflow
```

- subgraph の event は親の stream へ流れ、`namespace` は `["agent_loop:<task_id>"]`
- **`caller=model` / `caller=workflow` の判定は無改修で正しい**（subgraph 内の node 名が `agent`/`tools` のままなので）
- 二重記録なし、transcript は完全に残る

#### console streaming の回帰と修正（`NestedMessagesTransformer`）

LangGraph 組み込みの `MessagesTransformer` は `if params["namespace"] != self._scope_list: return` で**自 scope に絞る**ため、モデル出力が subgraph へ移った瞬間に console のトークンストリーミングが無音になった（`tool call: run_shell` が出ない）。`[node]` / `[tool]` の進捗行は `run_events` 由来なので生きていた。

組み込み側の docstring 自身が *"Consumers that need subgraph tokens should ... register a custom transformer"* と案内している。`event_logging/nested_messages.py` に `MessagesTransformer` を継承した 30 行の subclass を置き、**scope 判定だけを外した**（delta の組み立ては基底に任せる）。projection key は衝突を避けて `agent_messages` とし、runner の interleave 対象を差し替えた。

30ターン走る run を眺められることは研究上の作業性そのものなので、後回しにせずここで直した。

#### 検証結果

| キー | 結果 |
|---|---|
| `agent_inputs`（全turnでモデルへ渡ったmessage列） | **same** |
| `bound_tools` / `agent_turns` / `stop_reason` / `last_verification` / `attempt_dirs` | **same** |
| `transcript`（新設。モデル境界から読む9通） | 形も内容も従来の `final_messages` と同一 |
| `graph_state_messages` | 9 → 0（親が transcript を持たなくなった。意図通り） |
| `topology_mermaid` | CHANGED（`START → agent_loop → verify_final → END`） |

- `pytest tests/zeroshot` → **374 passed**
- `ruff check` / `format --check` clean

harness は `final_messages`（親state）を `transcript`（モデル境界）＋ `graph_state_messages`（親state）の2キーに分けた。前者が guard の本体で、後者は「親が transcript を持たない」という決定自体を固定する。

#### 既存testの追随

transcript の住処が変わったため、`result["messages"]` を見ていた8件をモデル境界から読むように直した（`agent_inputs[-1] + 最後の応答`、あるいは tool_call_id での検索）。

**follow-up:** `test_graph.py` の大半は実際には agent loop の振る舞い（tool往復、budget、retry、tool error）を親graph経由で見ている。loop が独立した単位になった今、これらは `create_agent_subgraph` を直接叩く `test_agent.py` へ移すのが正しい。Step 3 の diff にテスト再編を混ぜないため見送った。

### Step 4 — observability層の再編（`projections.py`）✅ 完了

Step 3 で `NestedMessagesTransformer` を足した結果、`event_logging/` の構成が意図を伝えなくなった。挙動は変えず、構造だけ直す。

#### 現状の問題

1. **共通責務がファイル名から読めない。** `normalizer.py`（RunEventTransformer）と `nested_messages.py`（NestedMessagesTransformer）が並んでいても、両者が同種であることが分からない。`nested_messages` は「基底クラスからの差分」由来の名前で、責務ではなく実装の由来を指している
2. **channel名がマジックストリング。** `init()` が返す `"run_events"` / `"agent_messages"` と、runner が `interleave()` へ渡す文字列が二重管理。不一致は `KeyError`（`interleave` の docstring 明記）で落ちるが、**run 開始後**（sandbox 構築・入力staging の後）なので遅い

#### 共通責務は「整形」ではない

両者は非対称である。

- `RunEventTransformer` は**大量に整形する** — secret の redact、base64/source の sha256 要約、URL クエリ除去、イベント名の正規化
- `AgentMessageTransformer` は**何も整形しない**。scope filter を外すだけで、delta の組み立ては基底クラスの仕事

したがって「整形」をファイル名にすると片方が嘘になる。1段上げた共通責務は **「graph が吐く生の protocol event stream を、消費者が使える named channel へ射影する」**こと。「射影 / projection」は LangGraph 自身の語彙である（"projection keys"、"the `messages` projection"）。

#### 変更

```
event_logging/
├── __init__.py
├── projections.py    RunEventTransformer      → CHANNEL = "run_events"
│                     AgentMessageTransformer  → CHANNEL = "agent_messages"
├── jsonl.py          JsonlEventWriter    ← 恒久記録の出力先
└── console.py        ConsoleReporter     ← ライブ表示の出力先
```

- `normalizer.py` + `nested_messages.py` → **1ファイル `projections.py`** に統合
- `NestedMessagesTransformer` → `AgentMessageTransformer` に改名
- 各 transformer に `CHANNEL` クラス定数を置き、`init()` と runner の両方がそれを使う

**1ファイルにする理由**: (a) 共通責務がファイル名になり、開かずに「この2つは同種」と分かる、(b) 差異が**隣接**で見える。2クラスが並んでいれば分業は読めば分かり、第三のファイルの docstring に説明を書く必要がない。214 + 36 = 250行で、分割を正当化するサイズではない。

`normalizer.py` の「整形」という意図は `_redact` / `_summarize` というクラス内部の名前が担う。ファイル名は「何を produce するか」を言う方が、パッケージ全体（射影2つ・出力先2つ）の対称性と噛み合う。

**却下した案**: `event_logging/__init__.py` の docstring に2クラスの分業軸を書く。構造の不整合をドキュメントで埋めるのは誤り。

#### 2つの射影の分業（実装の指針であって、docstringで代替してはならない軸）

|  | `RunEventTransformer` | `AgentMessageTransformer` |
|---|---|---|
| 消費する protocol mode | `values` / `tools` / `updates` / `tasks` | `messages` |
| 粒度 | 完了後の**確定した事実** | 進行中の**トークンdelta** |
| 出力先 | `events.jsonl`（恒久）＋ console の進捗行 | console のみ（揮発） |
| namespace | 記録する（フィールドとして） | 無視して全部通す |

**「tool call 担当 / subgraph 担当」ではない。両方とも全 namespace を見る。** 軸は「起きたこと（恒久・完了後・完全）」対「今喋っていること（ライブ・逐次）」である。同じ AIMessage が両方に出るのは重複ではなく、記録と閲覧という別のシンクだからである。

#### 参考: LangGraph の "channel" は同名異義

用語が2つの意味で使われるので、実装時に混同しないこと。

| | 例 | 何か |
|---|---|---|
| **state channel** | `messages`, `agent_turns`, `task` | state schema のキー。reducer を持つ |
| **stream channel** | `run_events`, `agent_messages` | transformer が出す**名前付き出力キュー**（＝projection） |

流れは `graph実行 → protocol event（method + params{namespace,timestamp,data}） → StreamMux が全 transformer の process() へ配る → 各 transformer が自分の channel へ push → 消費側が interleave(*names) で (channel名, item) を受け取る`。`required_stream_modes` は「その method を実際に生成せよ」というランタイムへの要求。projection キーが衝突すると `ValueError`（`messages` を再利用せず `agent_messages` とした理由）。

#### 検証結果

純粋なリネーム・移動なので、characterization は **全キー不変**（`topology_mermaid` を含めて差分ゼロ）。`pytest tests/zeroshot` **372 passed**。

`_active_node` が逐次実行前提であることは、コメントではなく**壊れ方を書いた**形で `RunEventTransformer.__init__` に残した（「fan out すると2つの `tools` が同時に active になり、後から始まった方が両方の呼び出しを自分のものとして主張する」）。

runner のマジックストリングは `RunEventTransformer.CHANNEL` / `AgentMessageTransformer.CHANNEL` へ置換し、`interleave()` へ渡す名前と `init()` が登録する名前が単一ソースになった。

### Step 5 — `AgentSpec` 導入、baseline がそれを使う ✅ 完了

**当初は staged multi-agent の step に含めていたが分離した。** そこには性質の違う2つが混ざっていた。

| | 内容 | 性質 |
|---|---|---|
| (a) | `AgentSpec` | **topology 非依存**。どんな graph でも要る |
| (b) | `StageReport` / `graph_staged.py` / prompt / config | **賭けの本体** |

(a) を (b) に埋めたのは誤りだった。早く入れる理由は「将来必要だから」ではなく、**今の署名が既に破綻しかけているから**である。`create_agent_subgraph` は役割を定義する引数を4つ（`prompt` / `prompt_context` / `max_turns` / `announce_turn_budget`）バラで受けている。

加えて、`AgentSpec` は multi-agent 設計全体が乗る型なので**形を間違えたときのコストが最大**である。baseline という既知の対象で先に試せば、staged graph を書く前に問題が出る。Step 3/4 を入れ替えたときと同じ「不確実な方を先に」。

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str
    prompt: PromptTemplate
    max_turns: int = 30
    announce_turn_budget: bool = False


create_agent_subgraph(spec, model, tools)  # tools は引数のまま
```

- **初版に `tools` フィールドを入れない。** baseline はエージェント1体・tool 3つ全部なので仕事がない。`tools: tuple[str, ...]` は tool 部分集合が実際に必要になったとき（Step 6）に toolbelt と一緒に足す。これが「toolbelt が先」という前提を落とせた理由
- `graph.py` の `DEFAULT_SYSTEM_PROMPT` は `CODER = AgentSpec(name="coder", prompt=PromptTemplate("coder"), ...)` へ移す。役割の identity がまとまった置き場を得る
- `model: BaseChatModel | None = None`（run 全体の model を使う）の seam は入れない（§7 保留）

#### 実装結果

`create_agent_subgraph(spec, model, tools, prompt_context, model_retries)` に整理した。役割を定義する4引数が `spec` 1つになり、環境（model / tools / prompt_context）と分離された。

**`max_turns < 1` の検証は `AgentSpec.__post_init__` へ移した。** spec は単体で正しいか間違っているかが決まるので、graph へ渡された場所ではなく**書かれた場所**で落ちる。`name` の空文字と `/` も拒否する（後で report のキーや node 名になるため）。

**`AgentSpec.name` に仕事を与えた。** parent の node 名を `agent.name` にしたので、`agent_loop` だった node が `coder` になり、**event log の namespace がエージェント名を運ぶ**ようになった。

```
node_started   ns=-                     coder
node_started   ns=coder:d3fcdd02...     seed
tool_started   ns=coder:d3fcdd02...     run_shell  caller=model
```

Step 9 の stage 別集計がこの namespace をそのまま使える。仕事のないフィールドを足さない、という方針とも一貫する（`name` を足すなら今使う）。

**config が「1エージェント = 1ブロック」になった。**

```yaml
workflow:
  _target_: ...create_reconstruction_graph
  _partial_: true
  agent:
    _target_: zeroshot.pipeline.workflow.AgentSpec
    name: coder
    prompt: {_target_: zeroshot.pipeline.messages.PromptTemplate, name: coder}
    max_turns: 30
    announce_turn_budget: true
  model_retries: 5
```

`max_agent_turns` / `announce_turn_budget` は graph の引数から消え、agent の属性になった。`model_retries` は transport の関心事なので graph 側に残す。prompt の差し替えが `workflow.agent.prompt.name=coder_v2` の1行で効くようになり、**baseline のままでも prompt ablation が CLI から回せる**。

#### 検証結果

- characterization は `topology_mermaid` のみ変化（`agent_loop` → `coder` の node 改名と2辺）。**層Aは完全に不変**
- `pytest tests/zeroshot` → **372 passed**
- Hydra が `AgentSpec` を1ブロックから組み立てることを実測確認

#### 停止理由は agent が報告する（`finish` node）

当初は親の `agent_loop` が子の最後の message から `BUDGET_EXHAUSTED` を導出していた。これだと親の wrapper が「state schema の変換」以外の判断を持ち、しかも判定条件が `should_continue`（子）と wrapper（親）に分かれて drift しうる。

subgraph の末尾に `finish` node を置き、`should_continue` が `"tools"` か `"finish"` を返す形にした。**なぜ止まったかを知っているのは agent 自身**なので、判定は routing の隣に来る。

```
START → seed → agent ⇄ tools
                 └─(tool callなし or 予算切れ)→ finish → END
```

親の wrapper は**純粋な射影**になった。

```python
def agent_loop(state):
    del state
    result = coder.invoke({"task": [input_message]})
    return {"agent_turns": result["turns"], "stop_reason": result["stop_reason"]}
```

Step 9 では6回ともこの機械的マッピングで済む。

**`stop_reason` event が2回出るようになったが、これは重複ではない。** 子の `finish` が出すものは namespace 付き（`coder:<task_id>`）で「**このagentが**なぜ止まったか」、親が出すものは namespace 空で「**runが**なぜ終わったか」。エージェント1体では値が一致するだけで、6体になれば namespace 付きの方が「どのagentが予算を溶かしたか」を示す。Step 9 の集計が使う情報そのものなので、抑制しない。test でも両方を namespace ごと assert している。

（Step 3 で潰した transcript の二重記録とは性質が違う。あちらは同じ内容が2回出ていた。）

#### 既存testの追随

`max_agent_turns=N` を渡していた箇所（test_graph.py 6件、test_runner.py 3件、test_hydra_config.py、test_run_pipeline.py）を `agent=AgentSpec(...)` へ。test_graph.py には `_agent(**overrides)` ヘルパを置き、各テストが変えたい属性だけ指定する形にした。

### Step 6a — `workflow/subgraphs/` 再編（純粋な移動）✅ 完了

fan-out（`fan_out_reduce`）の導入がほぼ決定事項になったため、**2つ目のテンプレートを書く直前に構造を変えるより先に整える**。挙動は変えない。

#### なぜ今か

当初は「テンプレートが1つのうちは flat のまま、2つ目を書く瞬間に切る」と判断していた。その前提（fan-out は当分先）が変わったので判断も変える。直前に構造を変えると、新機能の diff に move が混ざって読めなくなる。

#### 非対称の正体

`agent.py` は2種類のものを持っている。

`agent.py` は「機械」と「語彙」を混ぜて持っている。語彙を外に出せばテンプレート同士が対称になる。

| 中身 | 移す先 | 理由 |
|---|---|---|
| `AgentSpec` | `spec.py` | `create_fan_out_reduce_subgraph(worker: AgentSpec, judge: AgentSpec, ...)` も使う |
| `AgentState` | `state.py` | **fan_out_reduce は内部で agent subgraph を回すのでこの型を読む。** テンプレートが兄弟の型を所有していると、その一つだけが特別になる |
| `_budget_notice` / `create_agent_subgraph` | そのまま | 機械そのもの |

当初は「`AgentState` は1エージェント template 固有だから `agent.py` に残す」と判断したが、誤りだった。`spec.py` を作って spec 語彙を集めた以上、state 語彙も集めなければ一貫しない。**軸は「語彙は root、機械は `subgraphs/`」**である。

#### レイアウト

```
workflow/
├── spec.py                 AgentSpec                              spec語彙
├── state.py                AgentState, ReconstructionState,
│                           StopReason                             state語彙
├── subgraphs/
│   ├── __init__.py
│   ├── agent.py            create_agent_subgraph                  機械
│   └── fan_out_reduce.py   （Step 9以降）                          機械
├── graph.py                合成
└── graph_<name>.py         合成
```

root = 合成 + 語彙、`subgraphs/` = **再利用テンプレートの部品棚（機械のみ）**。テンプレートは互いの型を所有しない。

**階層ではなく並置である。** `subgraphs/` は `graph_<name>.py` の下位ではなく、どの graph からも同じように使える部品置き場。「グラフごとに subgraph と agent を定義し直す」形にしてはならない（可搬性が失われる）。`subgraphs/` の中では `agents_` 接頭辞は冗長なので `fan_out_reduce.py`。

#### fan_out_reduce の形（Step 9以降で実装）

**同一タスクを N 本サンプリングし、judge が1案を選ぶ**（(a) 型）。仮説を1つずつ配る (b) 型ではないので splitter は不要。

```python
create_fan_out_reduce_subgraph(worker: AgentSpec, judge: AgentSpec, n: int, ...)
```

#### 検証結果

- characterization **全キー不変**（`topology_mermaid` を含む）。純粋な移動であることが機械的に確認できた
- `pytest tests/zeroshot` **378 passed**、未収集のテスト関数ゼロ
- `workflow/__init__.py` が re-export を保つので、テストの `from zeroshot.pipeline.workflow import AgentSpec` も config の `_target_: zeroshot.pipeline.workflow.AgentSpec` も**1文字も変えずに通った**（`instantiate` で実測）。外から見た契約が変わっていない

`workflow/__init__.py` に re-export を集約してあったことが効いた。内部モジュールを直接指す import が散っていたら、この移動は全消費者に波及していた（Step 1a で決めた方針の配当）。

**作業中の注意点**: IDE の auto-import が削除済みの `workflow.agent` を指す行を復活させた。移動を伴う変更では、`ruff check` が通っても import 元が古いモジュールを指していないか確認すること。

### Step 6b — model を `AgentSpec` へ内包 ✅ 完了

`create_reconstruction_graph(models, agent, ...)` で model と agent が**並列に並んでいた**。「agent が model を内包する」という判断と矛盾しており、実際に読みづらかった。

#### 最初の実装は失敗だった（記録として残す）

当初は `AgentSpec.model: str | None`（名前で指す）+ runner が `{DEFAULT_MODEL: model, **extra_models}` を組む形にした。却下した理由は3つ。

1. **1つの意図に2つの記法が必要になった** — `+model@models.terra=gpt5_6_terra_codex workflow.agent.model=terra`。どちらがどの部分を変えるのか読み取れない
2. **概念が3つ増えた** — `DEFAULT_MODEL` / `extra_models` / `self.models`。いずれも歪みの後始末でしかない
3. **元の違和感が消えていない** — model と agent はやはり並列のままだった

「`AgentSpec` を純粋な値に保つため実体ではなく名前を持たせる」という私の論拠が、この複雑さの原因だった。

#### 採用した形

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str
    prompt: PromptTemplate
    model: BaseChatModel  # 実体を持つ
    max_turns: int = 30
    announce_turn_budget: bool = False
```

```yaml
# workflow/baseline.yaml — run に追従
agent:
  model: ${model}

# agent ごとに別backendが要る graph — 使うモデルを自分の defaults list へ宣言
defaults:
  - /model@fast_model: gemma4_ollama
agents:
  view_registration: {model: ${fast_model}}
  coder:             {model: ${model}}
```

**agent ブロックの記法は `model: ${...}` の1種類だけ**になった。Hydra の補間が `_partial_` の内側でも再帰的に instantiate されることを実測で確認している。

消えたもの: `DEFAULT_MODEL` / `extra_models` / `self.models` / `models` mapping / `"default"` マジックキー / `+model@...` という第2の記法。

#### runner contract が縮んだ

`PipelineRunner` から `model` が消え、`run_pipeline.py` の `instantiate(config.model)` も不要になった。model は workflow config が agent 経由で instantiate する。graph factory の引数も1つ減った。**contract が広がるのではなく縮んだ**のは、置き場所が正しくなった証拠と見てよい。

#### 手放したもの

`AgentSpec` が純粋な値でなくなった（ライブの client を持つ）。当初これを理由に反対したが、**恒久的な記録は `.hydra/config.yaml` の YAML 側**にあり、そこには `model: ${model}` と解決後の値の両方が残るので監査上の損失はない。`==` で spec を比較していた hydra test はフィールド比較へ変えた。

#### 副次的な変更

`agent` が必須引数になり `CODER` の既定値が消えた。**graph は既定の配役を持たず、config が必ず宣言する。** production では Hydra が `_partial_` で束ねるので影響はテストのみで、テストには `_graph_factory(model, **overrides)` ヘルパを置いた。

#### 検証結果

| 設定 | `agent.model` |
|---|---|
| `workflow=baseline model=gpt5_6_luna_codex` | gpt-5.6-luna |
| `workflow=baseline model=gpt5_6_terra_codex` | gpt-5.6-terra |
| agent を固定した graph + `model=gpt5_6_terra_codex` | **gemma4:e2b**（run に引きずられない） |

- characterization **全キー不変**
- `pytest tests/zeroshot` **378 passed**

#### 未決: subgraph テンプレートの命名

`create_reconstruction_graph` と `create_agent_subgraph` が並ぶと「graph が2種類ある」と読める。軸が2つある。

| 案 | `subgraphs/agent.py` | fan-out テンプレート |
|---|---|---|
| A: 何であるかで命名 | `create_agent` | `create_agent_ensemble` |
| B: 形で命名 | `create_react_loop` | `create_fan_out_reduce` |

合成側が `coder = create_agent(...)` とドメインの言葉で読めるのは A。B にするなら `agent.py` 側も改名しないと一貫しない。**未決。**

### Step 7 — tool生成を `toolbelt.py` へ、`AgentSpec.tools` を足す

`build_toolbelt(...) -> tuple[dict[str, BaseTool], BaseTool]`。`serialize_output=False` の instance を dict に入れず戻り値を分けることで、生の `VerifyOutputResult`（`source` 本文を含む）がモデルへ届く経路を構造的に塞ぐ。

同時に `AgentSpec.tools: tuple[str, ...]` を足し、名前で部分集合を選べるようにする。未知の名前は subgraph 構築時に `ValueError`。config は tool を**構築せず名前で指名する**だけなので、sandbox 依存物が YAML へ漏れない。

**tool を `AgentSpec` に持たせてよい理由**（一度は「tool 選択は graph 所有であるべき」と考えたが撤回した）:

- `prompt` を config で振れる時点で、config は既にエージェントの振る舞いを完全に決められる。「critic に `verify_output` を渡せてしまう」を理由に `tools` だけ隠すのは一貫しない。壊せることはフィールドを隠す理由ではなく、検証を置く理由である
- **tool 部分集合はそれ自体が実験変数**である（「critic に `run_shell` を与えると良くなるか」はこの研究が回したい ablation そのもの）。CLI から振れるべき
- 概念上は「エージェントごとに ToolBelt が1つ」で正しい。`ToolNode` が複数できるのは LangGraph の都合にすぎず、tool の実体は toolbelt が一度だけ生成した closure を共有するので、コストは実質ゼロ

Step 9 に畳んでもよい（tool 部分集合が要るのはそこが最初なので）。単独で入れるか一緒にするかは着手時に決める。

### Step 8（任意）— 対照実験: 単一agentのまま段階を踏ませる

「段階を踏むこと」の効果と「contextを分けること」の効果は別物である。前者だけで足りるならmulti-agent化は複雑さに見合わない。

Step 6の後なら新規40行で測れる（前だとloopのコピペになる）。`graph_guided.py`: agentがtool callをやめるたびに次段階の指示 `HumanMessage` を注入し、`stage_index` を進める。contextは1本のまま。

### Step 9 — staged multi-agent graph

`StageReport` + reducer、`graph_staged.py`、prompt 6本、`configs/workflow/staged.yaml`。

- **reducerが必須である理由**（投機的一般化ではない）: LangGraphはreducerの無いkeyを**丸ごと置換**する。差し戻しループでcoderが2回目のreportを返すと、他の全stageのreportが消える
- **`reports` は `dict[str, list[StageReport]]` に一様化する。** 単一エージェントの stage も要素1つの list にする。fan-out（§6）で N 体分を入れる必要があり、後から形を変えるより最初から揃える方がよい。reducer が1本で済み、読み出し側も分岐しない
- **stage間受け渡しは最終AIMessage本文**。これはlossyである。緩和は2つ: (1) 全stageが `SandboxWorkdir` を共有するので「reportは要約、実データは `/work/stages/<name>/` のファイル」という分担をpromptで指示する、(2) 全stageが `input_message` を毎回受け取るので元のDXFとrenderへは常に戻れる
- **コード生成と検証を分けない**: coderは `verify_output` を持ったまま自己debug loopを維持する（baselineのexecution_success 18/20を支えている）。その上に「入力図面と最終renderを見比べる」だけのcritic agentを載せる。coderの自己debug（構文エラー、STEP出力失敗）とcriticの見る対象（幾何的差分）は本当に違う

### Step 10 — 集約のstage対応

`aggregate_run.py` は `data["node"] == "agent"` で数えており、stagedでは全stage合算になる。`event["namespace"]` からstage名を取り、stage別のturn/tokenを出す。`node_ms` のキーも `(namespace, node)` 相当へ。

**どのstageが予算を溶かしているかは本計画の第一の観測目的**なのでoptionalではない。

`RunEventTransformer` は逐次実行の範囲では改修不要。ただし `_active_node` が namespace をまたいで1つしかないため**逐次実行前提**である旨をコメントで明記する（fan-out で最初に壊れる。§7 参照）。

---

## 6. fan-out（並列agent）の設計メモ

実装は先だが、この会話で決まったことと、先に決めておかないと手戻りになることを残す。

### 想定する形

仮説を N 体に独立して出させ、**fan-in 後に判定エージェントが1案を選ぶ**。CadQuery を書く前にどの仮説で行くかを決めさせ、coder の責務が膨らむのを防ぐ。

```
semantic_hypotheses ──Send×N──▶ hypothesis_agent   （同一subgraph × 異なるtask）
                                      │
                                      │ reducer が list[StageReport] に集約
                                      ▼
                                   judge          （別roleのagent。1案を選ぶ）
                                      ▼
                                 operations ──▶ coder ──▶ ...
```

### 決まったこと

- **判定役は reducer には置けない。** reducer は「同じkeyへの複数書き込みをどう畳むか」の純粋関数であり、モデルを呼べない。**reducer は集めるだけ、判定は fan-in 後のノード**
- fan-out は conditional edge から `Send` を返す。`Send("hypothesis_agent", {"task": [...]})` で**エージェント毎に別の task** を渡せる。`AgentState` が `task` と `messages` を分けているのがそのまま効く（役割は1つの compiled subgraph、個体差は task だけ）
- compiled graph は再入可能なので、同一 subgraph を N 並列に呼ぶこと自体に問題はない
- 過去 run との比較不能性は問題にならない。**グラフが違えば config group が違い、ablation 表の別の行になる。** reducer の形を変えると比較できなくなるのは同一グラフ内の話

### 先に壊れるもの（fan-out の step で必ず直す）

1. **`caller` 判定。** `RunEventTransformer._active_node` は `str | None` で**全 namespace に1つ**しかない。N体並行だと複数の `tools` ノードが同時に active になり誤判定する。修正は `dict[tuple[str, ...], str]` にして namespace で引く形（数行）
2. **共有 `SandboxWorkdir`。** N体が同じ `/work` に scratch を書く。`_issue_verification_id_and_dir` の `max(existing)+1` 採番（`tools/verify_output.py`）は並行で競合。`read_only_subdirs` も共有リストへの mutation。**逐次 multi-agent（Step 9）では共有が利点**（前段のdumpを後段が読める）なので、ここは Step 9 に混ぜず fan-out と同時に扱う
3. **console のトークンストリーミング。** `ConsoleReporter.render_message` は `for event in message:` で1本を完了まで回すので、2体同時だと片方をブロックするか混ざって読めなくなる。方針は**代表1体だけトークン展開し、他は `[model] <namespace>` の行だけ出して畳む**。`ChatModelStream` は `.namespace` / `.node` を公開しているので材料はある（実測確認済み）。実装時の注意として、掴まなかったストリームを一切 iterate しないと pump が詰まる可能性があるので drain の要否を実測すること

### 疑問として残しておくこと

**仮説「検証」を N 分割するのが良いかは疑問。** 矛盾する仮説の組を解消するには全仮説を同時に見る必要があり、N体に1つずつ渡すと各自が局所的に妥当と判定して矛盾が残る。fan-out がより自然なのは検証より**候補生成**（N通りの仮説、N通りのCadQuery）の側だと思われる。上図はその形にしてある。

---

## 7. 保留

- run-level `BudgetPolicy`
- stage出力の構造化（`submit_<stage>` toolによるschema強制） — free textで足りるか実測してから。`AgentSpec.tools` に1つ足すだけで移行できる形にする
- stageごとのmodel差し替え — `AgentSpec.model` の seam
- **`test_graph.py` の loop テストを `test_agent.py` へ移す**（Step 3 の follow-up）。大半は実際には agent loop の振る舞いを親graph経由で見ている
