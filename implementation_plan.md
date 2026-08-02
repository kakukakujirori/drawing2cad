# Zero-shot Pipeline 再実装計画（Final）

## 1. シニアレビュー結論

旧Claude版pipelineは、責務別のフォルダ分けには再利用価値があった。一方、今回の第一目的である「実装者自身が内部ロジックを理解し、将来一人で変更できること」に対して、元の実装計画は妥当ではなかった。

特に、次の点を修正する。

- 旧巨大runnerを先に多数のファイルへ切り出さない
- CLI agent用sandbox、credential、provider監査をAPI版へ一括移植しない
- DrawingIR、pose推定、仮説fan-outを、動くbaselineより先に実装しない
- LangGraphのgraphだけ先に大きく作り、stub nodeを後から埋める方式を採らない
- 入力、実行、feedback、評価を一度に実装しない
- 推論中の自己検証と、GTを使うオフライン評価を混同しない
- projection IoUや閾値を、十分なcalibrationなしに候補の自動棄却へ使わない
- 比較実験用budgetを、将来の並列agent構成が未定の段階でgraphへ埋め込まない

新実装では、最小のend-to-end経路を最初に完成させ、その後に一機能ずつ追加する。各追加機能は、単体テスト、少数サンプルでの観察、アブレーションの順で評価する。

旧実装はコピー元や完成形とせず、再実装中は次の用途だけに参照した。

- 入出力contractの確認
- security、CAD実行、監査で考慮すべき失敗例の収集
- DrawingIR、pose、projectionの研究仮説の参照
- 新実装との回帰比較

Phase 4完了時点で必要なcontractとtestの移植を終え、旧実装はworktreeから削除した。必要な場合はgit履歴を参照する。

## 2. 目的と優先順位

優先順位は以下とする。

1. 実装者が各moduleの目的、入出力、失敗条件、テスト方法を説明できる
2. 一サンプルの全message、tool call、artifact、状態遷移を再現・監査できる
3. 評価の誤りやGT leakageがないことを保証する
4. zero-shot 3D CAD復元精度を、測定可能な変更によって改善する
5. 将来、並列agent、Gemini/Claude CLI backend、追加の図面解析を導入できる

精度向上は最終目標だが、「機能を増やしたから改善したはず」とは判断しない。必ず、同一サンプル、同一モデル条件による比較結果を残す。

## 3. 確定した設計要件

### 3.1 Package名

正式名称は次に統一する。

```text
zeroshot/
```

新実装の内部ロジックは`zeroshot.pipeline`として通常のPython packageにする。ユーザー向けCLIだけは`zeroshot/run_pipeline.py`に置く。`zero-shot`表記、`sys.path`への場当たり的な追加、ハイフン付きpackageのimport workaroundは新実装へ持ち込まない。

実行方法は最終的に次へ統一する。

```bash
python -m zeroshot.run_pipeline ...
```

### 3.2 一サンプルの入力

正式な入力は以下の4点とする。

```text
techdraw/dxf/<id>.dxf
render_3d/hlg_perspective/<id>.png
render_3d/hlg_translucent_faces_perspective/<id>.png
render_3d/transparent_shaded_edges_perspective/<id>.png
```

3枚のPNGは三方向のviewではなく、同じ部品を異なる表現styleで描いた3種類のperspective renderである。

`target/`または`target_step/`は、独立したオフライン評価processだけが読む。推論graph、model、tool、feedback生成processからは到達不能にする。

### 3.3 初回入力のaccess config

```yaml
access_render3d: image           # none | path | image
access_render3d_styles:
  - hlg_perspective
  - hlg_translucent_faces_perspective
  - transparent_shaded_edges_perspective
```

意味は以下の通り。

- DXFは必須入力として、`InputManifest`が持つinput DXF pathと役割を常に初回user messageへ入れる
- `access_render3d: none`: 初回は3D renderを与えない
- `access_render3d: path`: 選択styleのpathとstyle名だけを伝える
- `access_render3d: image`: 選択styleの画像をmultimodal user messageへ添付する

`access_dxf` configは設けない。raw DXF textまたはDXF JSON tagsをmessageへ直接注入せず、modelが必要に応じてDXF解析toolを使う。

### 3.4 次ターンのfeedback config

```yaml
feedback_render3d: image         # none | path | image
feedback_render3d_styles:
  - hlg_perspective
  - hlg_translucent_faces_perspective
  - transparent_shaded_edges_perspective
```

意味は以下の通り。

- verified STEPからfeedback DXFを生成できた場合、そのpathを次のfeedback user messageへ常に入れる
- `feedback_render3d: none`: 候補の3D renderを返さない
- `feedback_render3d: path`: 選択styleのpathだけを返す
- `feedback_render3d: image`: 選択styleの画像を次のmultimodal feedback user messageへ添付する

`access_render3d_styles`と`feedback_render3d_styles`は独立させる。初回入力とfeedbackで異なるstyle集合を使えるようにし、別々にアブレーションする。

`feedback_dxf` configは設けない。DXF feedback機能が実装された後はpathを常に返す。ただしsyntax error、timeout、invalid STEP、projection failureなど、DXFが存在しない場合は架空のpathを返さず、生成不能理由をexecution feedbackへ含める。

`FeedbackManifest.render3d_paths`には生成に成功したstyleのキーとpathだけを入れ、失敗したstyleのキーを入れない。valueへ`None`を入れない。MessageBuilderは存在するstyleだけを返し、0枚ならrender用content block自体を追加しない。欠損理由は必須のexecution feedbackへ含める。

`feedback_execution`を無効化するconfigは設けない。実行成否、syntax/AST error、timeout、STEP生成、solid数、kernel validityなど、信頼済み実行器が得た結果は必ずmessage historyと監査logへ残す。runが継続する場合は必ず次のmodel turnへ返す。

### 3.5 Modelとbackend

Phase 4で接続する対象は以下とする。

- laptop上の開発用probeとして、OllamaでserveするGemma 4
- GPU server上のSGLangでserveするQwen 3.6
- ChatGPT Codex OAuth backendのGPT-5.6-luna

LangGraphはworkflowを実行し、model呼び出しはLangChainのchat model integrationが担当する。GemmaとQwenはOpenAI-compatible endpointへ`ChatOpenAI`系clientで接続し、GPTは`_ChatOpenAICodex`で接続する。workflowが要求するboundaryはLangChainの`BaseChatModel`と`bind_tools()`だけであり、provider固有factoryや独自`AgentModel` protocolは、実際に異なるinterfaceが必要になるまで追加しない。Qwenについてのみ、SGLangの`reasoning_content` deltaをLangChainの共通content blockへ保持する薄い`SGLangChatOpenAI` subclassを置く。

workflowはprovider名、base URL、credentialへ依存させない。Hydraのmodel configから生成したtool-call可能なchat modelをagent nodeへ注入する。

将来、Gemini/Claudeを非APIのCLI経路で追加する。その際は、CLI outputをLangGraphが扱う`AIMessage`とtool callへ正規化するadapterを別実装する。CLI固有のcredential、filesystem jail、native tool、provider trace監査を、初期API版へ先回りして入れない。

### 3.6 Agent数とbudget

初期版は単一agentとする。仮説fan-out、候補ごとの並列agent、投票、branch mergeは実装しない。

将来の並列化に備え、初期版から以下のIDを持つ。

- `run_id`
- `sample_id`
- `agent_id`
- `verification_id`
- tool call ID
- artifact IDと所有verification

比較実験用の`max_model_turns`、`max_verifications`などは初期graphへ固定実装しない。並列構成が決まった後、graphから独立した`BudgetPolicy`として設計する。

ただし、運用上の安全装置として以下は初期版から必要である。

- model API timeout
- CAD実行timeout
- OCC render timeout
- 高めで変更可能なLangGraph再帰安全上限
- 外部からのrun cancellation
- 安全停止理由の監査記録

## 4. Architecture

### 4.1 信頼境界

処理を次の4領域へ分ける。

| 領域 | 役割 | GT access |
|---|---|---|
| Agent workflow | message、tool選択、`model.py`編集、検証要求 | 禁止 |
| Persistent sandbox workdir | model生成shell/Pythonによる入力解析、数値計算、CAD試行 | 禁止 |
| Trusted reconstruction runtime | sandbox起動、verification登録、STEP検証、feedback生成 | 禁止 |
| Offline evaluator | 完了した予測STEPとGT STEPの比較 | 許可 |

model生成commandとcodeはすべてuntrustedとする。通常のprocessで直接実行せず、共通の`SandboxRunner`からtool callごとにfreshなtimeout可能隔離processを起動する。Linuxの`bwrap --unshare-all`を必須の隔離境界とし、`bwrap`が見つからない場合は`SandboxRunner`の構築を失敗させ、非隔離subprocessへfallbackしない。実行時にsandbox processを起動できない場合は`INFRA_ERROR`としてfail closedにする。

sampleごと・agentごとに一つの`SandboxWorkdir`を作り、run終了まで保持する。`SandboxWorkdir`は、内部生成した一時`host_bind_dir`を所有するmodeと、trusted callerから渡された既存directoryを借用するmodeを持つ。どちらのmodeでもsandbox側のbind先は`/work`である。`run_shell`と`verify_output`はtool callごとにfreshなbwrap processを起動するが、毎回同じ`SandboxWorkdir`をread-writeでbindする。これによりprocess、Python変数、環境変数は共有しない一方、modelが作成・編集した`model.py`、解析script、JSON、dump画像などの中間fileはtool call間で保持する。

### 4.2 Pathと生成artifactの扱い

`InputManifest`はhost側の通常の`Path`で必須のinput DXFとinput render mappingを保持する。`FeedbackManifest`はverification ID、必須のexecution feedback、optionalなfeedback DXFとfeedback render mappingを保持する。original inputとverification feedbackは寿命と生成元が異なるため、一つのcombined manifestへまとめない。style名の固定listをpipeline内へ重複定義せず、`InputManifest.render3d_paths`のkey集合をそのsampleで利用可能なstyleのsource of truthとする。`target/`と`target_step/`はいずれのmanifestにも含めない。

`SandboxWorkdir`作成後、trusted callerが`Path`と`shutil`を使い、active sampleのDXFとconfigで許可されたinput renderだけを`host_bind_dir`配下の固定pathへcopyする。この初期copyをstagingと呼ぶ。`SandboxWorkdir`と`SandboxRunner`はmanifest、file配置、staging policyを知らない。専用の`Workspace` class、`WorkspacePath`、host/sandbox pathの再帰変換API、`copy_in`/`copy_out` wrapperも作らない。host側の元fileをmountしないため、workdir内のcopyをmodelが変更しても元fileには影響しない。`verify_output`がfeedback DXFとperspective viewsを生成した後は、trusted callerがそのverificationの生成物を同じworkdirへ追加する。GT、repository、他sample、未許可render、credentialはstageしない。

modelへは`host_bind_dir`を教えず、`/work`配下のsandbox pathのみを提示する。`run_shell`のcwdは常に`/work`である。manifestはtrusted host fileの存在検証とmessage用画像読取に使い、modelへ提示するpath文字列はstaging時に決めたpathを使う。`load_image`は`host_bind_dir`配下の画像だけを読み、path traversalとworkdir外を指すsymlinkを拒否する。入力DXFは正規の問題入力なので`run_shell`と解析scriptから読み取れる。最終`model.py`は自己完結させ、`CadQueryExecutor.validate_source()`がcomment以外の文字列定数に`.dxf`を含むsourceを実行前に拒否する。一方、GT、repository、他sample、credential、networkはどのmodel生成commandからも到達不能にする。

workdir内fileはmodelが自由に変更できるscratchであり、それだけではtrusted artifactとしない。`verify_output`ごとに現在の`model.py`をhash付きimmutable artifactとして保存し、verified STEPと対応するDXF/renderを同じverification IDへ紐付ける。複数領域で共通の登録・hash・manifest処理が実際に重複した時点で`ArtifactStore`を抽出する。

### 4.3 依存方向

依存は次の一方向にする。

```text
run_pipeline
  -> PipelineRunner
       -> workflow
            -> tools
                 -> sandbox
                 -> verification
       -> event_logging
  -> Hydra model config
       -> LangChain BaseChatModel
            -> zeroshot.models  # provider互換差分が必要な場合だけ

serve_model  # pipelineとは別process / environment
  -> SGLangServer

offline evaluation
  -> src.evaluation
  -> src.metrics
```

`workflow`から削除済みの旧runnerをimportしない。

## 5. 目標フォルダ構成

最初から空ファイルをすべて作らず、後述のphaseで必要になった時点で追加する。

```text
zeroshot/
├── run_pipeline.py                  # 新pipelineの薄いHydra composition root
├── models/
│   └── sglang.py                    # SGLang reasoning streamの薄い正規化
├── server/                          # pipelineと独立したSGLang launcher
│   ├── serve_model.py
│   ├── sglang.py
│   └── configs/
├── pipeline/
│   ├── event_logging/               # run eventの正規化、JSONL、console表示
│   │   ├── __init__.py
│   │   ├── normalizer.py
│   │   ├── jsonl.py
│   │   └── console.py
│   ├── manifest.py                  # InputManifest / FeedbackManifestとpath検証
│   ├── messages.py                  # access/feedback message builder
│   ├── runner.py                    # application wiringとsample単位のrun lifecycle
│   ├── sandbox.py                   # SandboxWorkdirと全model生成command用の共通隔離実行基盤
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── run_shell.py
│   │   ├── load_image.py
│   │   └── verify_output.py
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── run_cadquery.py         # CadQuery実行、STEP生成・検証
│   │   └── render.py               # Phase 5: 三面図DXF・perspective views生成
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── graph.py                # node実装とroutingが小さい間は同居させる
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── score_run.py
│   ├── drawing/                     # 後続TODO
│   │   ├── schema.py
│   │   ├── extract.py
│   │   ├── summary.py
│   │   └── pose.py
└── configs/
    ├── default.yaml
    └── model/                       # Phase 4で追加
        ├── gemma4_ollama.yaml
        ├── gpt5_6_luna_codex.yaml
        └── qwen3_6_sglang.yaml

tests/
└── zeroshot/
    ├── event_logging/
    ├── live/
    ├── models/
    ├── server/
    ├── tools/
    ├── verification/
    ├── workflow/
    └── test_*.py
```

`run_pipeline.py`はHydraによるconfig composition、依存objectの生成、内部pipelineの呼び出しだけを担当し、sample loopやworkflow構築を置かない。Phase 1では`zeroshot/configs/default.yaml`と、`pipeline/`直下の`manifest.py`、`messages.py`だけを追加する。`run_pipeline.py`と`runner.py`を含む後続ファイルは必要になるphaseまで作らない。

設定専用の`config.py`は作らない。YAMLの`_target_`と各runtime classのconstructorをcontractにし、`hydra.utils.instantiate()`で生成する。未知の引数や不足した必須引数はinstantiate時に拒否し、値の許容範囲や組合せのようなsemantic validationは、その値を所有するclass自身が行う。

ファイルが大きくなってから責務に沿って分割する。最初から`constants.py`、`provider.py`、`credentials.py`などを機械的に量産しない。

## 6. LangGraph Agent設計

### 6.1 State

`messages`にはLangGraphの`add_messages` reducerを使う。それ以外は単一agentが一箇所ずつ更新し、未使用の並列reducerを先回りして入れない。

概念上のstateは以下とする。

```python
class ReconstructionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_verification: NotRequired[VerifyOutputResult]
```

Phase 3ではgraph node間で実際に受け渡す`messages`と、workflow主導の最終検証結果`last_verification`だけをstateへ置く。run/sample IDは`PipelineRunner`、verification IDは`verify_output`とfilesystem、監査情報はevent logが所有する。`SandboxWorkdir`、`SandboxRunner`、renderer、model objectなどのruntime dependencyや、将来必要になるかもしれないだけのfieldをstateへ先回りして追加しない。新しいnode間で永続的に共有する値が具体化した時点で追加する。

### 6.2 bindするtool

初期agentへ次の3toolを公開する。

1. `run_shell(command)`
2. `load_image(path)`
3. `verify_output()`

`run_shell`はmodelが生成した任意のbash commandを、run中共有する`SandboxWorkdir`をbindしたsandbox内で実行する。modelに公開するtool引数は`command`だけとし、timeoutはtrusted側の`SandboxRunner`設定を使う。`SandboxRunner`と`SandboxWorkdir`はtool factoryのclosureで保持する。tool resultは`SandboxResult`を`status`、`returncode`、`stdout`、`stderr`からなるJSON化可能なmappingへ変換し、non-zero exit、`TIMEOUT`、`INFRA_ERROR`もtool自体の未処理例外にせず結果として返す。modelはshell、Python、標準commandを使ってworkdir内のfileを自由に読み書きし、`model.py`、解析script、JSON、dump画像をtool call間で継続利用できる。sandboxには`ezdxf`、Pillow、NumPyと、現在対応しているCAD libraryを用意する。初期vertical sliceではCadQueryを対象とする。workdir以外のhost filesystem、GT、repository、credential、networkへ到達させない。

`load_image`はworkdir pathを受け取り、workdir外へのescapeとsymlinkを拒否した上で画像をmultimodal contentとしてmodelへ返す。これによりmodelはinput/feedback renderだけでなく、自ら生成したdump画像も確認できる。

modelへ公開する`verify_output` toolは引数を取らず、共有`SandboxWorkdir` rootの`model.py`だけを対象とする。`tools/verify_output.py`がsourceを安全に読み、`CadQueryExecutor`へ渡し、model-visibleな構造化reportへ変換する。`CadQueryExecutor`はsourceをverification ID配下へimmutable artifactとして保存した後、内部でfreshな`SandboxWorkdir`を作り、trusted export epilogue付きwrapperをbwrap process内で実行して、STEP再読込、single solid、kernel validityを確認する。`verification/verify_output.py`という追加use case層と`tools/verify_output_tool.py`という別adapterは作らない。

Phase 3/4では`render_views=False`だけを使い、`True`なら明示的に`NotImplementedError`を送出する。この例外はmodel reconstruction failureではなく、未実装機能を有効にしたconfiguration errorとしてrun開始前またはtrusted runtime境界で扱う。Phase 5で`True`を実装し、verifiedの場合だけ`render.py`で三面図DXFとperspective viewsを生成してverification ID配下のartifactとworkdir内feedbackへ保存する。

`verify_output`のmodel-visibleな戻り値は、Phase 5でもexecution status、verification ID、stdout/stderr、errorからなるJSON化可能な構造化reportに保ち、source本文は含めない。trusted側の最終reportはsourceを保持し、監査eventでは本文を保存せずhashとbyte数へ要約する。生成したDXF/renderの管理には`FeedbackManifest`を使い、構造化reportのJSON文字列を`execution_feedback`へ格納し、生成に成功したartifactのpathだけをoptional fieldへ登録する。`FeedbackManifest`自体や画像bytesをtool resultのJSONへ直接混ぜず、model-visibleな`ToolMessage.content`から分離したtrusted側handoffとしてworkflowへ保持する。具体的なtransportは`ToolMessage.artifact`またはgraph stateを候補とし、feedback node実装時に一方を選ぶ。

`feedback_render3d: path | image`の選択と画像bytesの読込みは`verify_output`の責務にせず、`FeedbackManifest`を受け取る`MessageBuilder`が行う。`path` modeでagentへ提示するのはsandboxから参照できる`/work`配下のpathだけとし、manifestがartifact読取り用に保持するhost pathをそのままmessageへ出さない。host artifactをshared workdirへstageしてsandbox pathを組み立てる具体的なAPIは、visual feedbackを実装するPhase 5で決める。

Phase 4完了時点で`SandboxRunner`が提供するresource guardは、wall timeoutと返却するstdout/stderrの設定可能な切り詰めまでである。切り詰めは`subprocess.run(capture_output=True)`の完了後に行うため、model contextへ巨大なtool resultを返さない効果はあるが、process実行中のhost memory使用量は制限しない。CPU、memory、PID数、workdir disk使用量のhard limitもまだ提供しない。Qwenのlive runではDXF dumpによるcontext膨張を返却時の上限で抑えられた一方、in-flight captureや追加hard limitが必要な障害は観測していない。これらはPhase 4の必須条件にせず、実測で必要性と妥当な制限値が判明した時点で追加する。

system promptはCadQuery scriptと`result`変数というtask contractに集中させる。利用可能toolは`bind_tools`でmodelへ提示し、shellで可能な操作、利用可能package、workdirのpath、tool result contractは各tool descriptionへ記述する。ezdxfやCadQueryの個別API tutorialをsystem promptへ埋め込まない。

### 6.3 中間実行と最終実行

同じ`create_verify_output_tool()` factoryから、用途の異なる2つのtool instanceを作る。

1. modelが自発的に呼ぶ中間検証用。JSON化可能なmappingを返す
2. modelがtool callなしで完成を表明した後にworkflowが呼ぶ最終検証用。`VerifyOutputResult`を返す

最終検証では、中間検証済みで`model.py`のhashが同じでも再実行する。これによりworkdirに残った古いSTEPではなく、現在の`model.py`と対応するSTEPであることを保証する。監査eventには呼び出し元を`model`または`workflow`として記録する。

model主導の`verify_output`は、`run_shell`、`load_image`と同様に対応する`ToolMessage`でtool protocolを閉じ、その構造化結果を次のagent turnが直接解釈する。Phase 3では追加の`build_feedback` nodeや`HumanMessage`を挟まない。Phase 5でvisual feedbackを追加するときに、画像をtool resultから分離する必要が生じた場合だけ専用nodeを導入する。

workflow主導の最終検証は成功・失敗にかかわらず終了する。失敗後にagentへ戻すと停止条件が不明瞭な再試行loopになるため、Phase 3では行わない。最終検証は最後に新しいverification IDを発行するので、`model.py`が存在して検証が実行されたrunでは`attempts/`内の最大連番が最終提出物である。verified STEPを別の`final/`へ二重copyしない。

### 6.4 Graph topology

`create_react_agent`で隠蔽せず、学習のため`StateGraph`とconditional edgeを明示的に組み立てる。Phase 3の3toolはいずれも「対応する`ToolMessage`を追加してagentへ戻る」という同じ遷移なので、一つの`ToolNode`へ渡す。tool固有の実行内容とschemaは各`create_*_tool()` factoryが所有する。nodeごとの遷移やretry policyが本当に分かれた時点で`agent.py`や`routing.py`への分割を検討する。

```mermaid
flowchart TD
    START --> agent
    agent -->|tool callあり| tools
    tools --> agent
    agent -->|tool callなし| verify_final
    verify_final --> END
```

tool callを生成するかはmodelが決める。tool callの引数検証、実行、次nodeへのroutingはworkflowが決める。

### 6.5 Image tool resultの互換性

初回の`image` modeは通常のmultimodal user messageとして送る。

`load_image`では、tool call IDに対応する`ToolMessage`を必ず作る。その上で画像content blockを同じToolMessageで扱えるか、GPT backendとSGLang/Qwen backendのlive capability testを行う。

両backendで同じ形式が使えない場合は、全backendで次へ統一する。

1. `ToolMessage`でartifact ID、style、読取成功を返す
2. 続くuser messageへ画像を添付する

providerごとに異なる会話意味論を使うのではなく、両者がサポートする共通形式を採用する。

Phase 5で`feedback_render3d: image`を実装する際は、`verify_output`のToolMessageへ直接混ぜず、対応するToolMessageの後に`build_feedback` nodeが作るmultimodal user messageへ添付する。これによりtool protocolと「feedbackを次のuser instructionで返す」という実験条件を分離する。modelが自発的に生成したdump画像はPhase 3から`load_image`で個別に確認できる。

## 7. 監査設計

LangGraphを採用する主目的の一つは、agentの外部から観測可能な行動を監査することである。

各runで少なくとも次を保存する。

- run/sample/agent/verification ID
- graph nodeと遷移元・遷移先
- provider、model、base URL識別子、sampling parameter
- access/feedback configとeffective style
- Hydraでresolveした実効config
- system/user/tool messageの再現に必要なpayloadまたはartifact参照
- prompt template versionとrendered prompt hash
- tool名、検証済み引数、tool call ID
- toolの呼び出し元`model | workflow`
- 開始・終了時刻、duration、status、短いerror
- verification時の`model.py` hash
- execution、STEP verification、render結果
- token usageなどproviderが返すusage
- safety stopとinfra error

監査対象はmessage、tool call、tool result、artifact、状態遷移である。非公開chain-of-thoughtの生成や保存を要求しない。

LangGraphのcheckpointerだけを唯一の研究記録にしない。Phase 3では人間が読みやすくversionに依存しにくい`events.jsonl`をcanonicalなrun logとし、checkpointerはresumeやdebugの補助として扱う。dataset全体を集約するrun manifestやartifact manifestは、実model、renderer、scorerの出力contractが具体化する後続phaseで追加する。

Phase 3のevent正規化とJSONL出力は、規模が小さい間は`pipeline/event_log.py`へまとめる。`RunEventTransformer`がLangGraph protocol eventを安定したrun eventへ射影し、`JsonlEventWriter`が1 event 1行で逐次flushする。Transformerはgraph topologyへ埋め込まず、streamとwriterを所有する`PipelineRunner`が`stream_events()`のcall-time optionとして登録する。

Phase 4でconsoleへのlive表示という二つ目のconsumerが必要になったため、`pipeline/event_logging/`へ分割する。`normalizer.py`は正規化、`jsonl.py`はcanonical logの永続化、`console.py`はRichによる人間向け表示だけを所有する。`PipelineRunner`は`run_events`と`messages` projectionを駆動して振り分けるが、表示形式は持たない。独立したstream抽象やreporter protocolは、二つ目の実装が必要になるまで追加しない。

canonicalなrun metadataとsandbox workdirのsnapshotは同じdirectoryへmergeしない。前者はsample artifact root、後者はその`workspace/`配下へ保存し、modelが作成できる同名fileによる`events.jsonl`やcheckpointの置換をdirectory境界で防ぐ。`workspace/attempts/`だけはtrusted側の`verify_output`が作成し、sandboxへread-only bindするpipeline管理領域とする。

credential、API key、authorization headerは保存しない。base URLにsecret queryが含まれる場合もredactする。

## 8. 実装フェーズ

各phaseは前のphaseの成果を実際に読み、テストし、説明できてから進む。

### ✅ Phase 0: datasetと既存評価器の基準線を確認する

#### 確認するもの

- 新package名を`zeroshot`に確定する
- `data/test_vlm/test_vlm.csv`に記載された20件すべてを、固定debug兼regression sampleとする
- 既存testにある手書きDXF、画像、CadQuery code、STEP fixtureを再利用できることを確認する
- `src.evaluation`、`src.metrics`、solid検査の既存testを実行し、評価基準線とする

詳細contractは全体像が見えない段階で一括固定しない。各phaseの実装直前に、そのphaseが必要とする最小の入出力だけをtestと型で確定する。旧runnerはその時点で必要な振る舞いを調べる参照資料とし、contractを無条件に転記しない。

model backend用dependencyのversion固定もこのphaseでは行わない。Phase 4ではmodel名、接続方式、samplingとserver起動configをcodeとして残す。library versionの列挙はrunごとには保存せず、再現可能な比較実験を開始する段階でenvironmentまたはlockfileとして管理する。

#### 評価器のsanity check

- GT STEPを自身と比較した場合に高得点になる
- missing predictionは成功率の分母から消えず、失敗として数えられる
- invalid STEP、multi-solid、timeoutがrun全体を落とさない
- 明らかに異なる形状または変形形状でscoreが低下する
- shape-normalized metricとunnormalized metricを混同しない

#### 理解チェック

次を自分の言葉で説明できること。

- 推論中のfeedbackとGT評価の違い
- なぜ図面から絶対scaleが復元できない場合があるか
- なぜOCC処理をchild processへ隔離するか

#### 完了条件

modelもLangGraphも呼ばず、20件の入力集合が揃っていることと、既存fixtureに対する評価器の正否を確認できる。

#### 完了記録

2026-07-27に`drawing2cad` conda environmentで、次の既存testを実行した。

- `tests/evaluation/test_evaluator.py`
- `tests/evaluation/test_scoring.py`
- `tests/metrics/test_eccv.py`
- `tests/metrics/test_geometry.py`
- `tests/data/audit/test_solid_checks.py`

結果は39 test passed、40 subtests passed。`data/test_vlm`は20 IDすべてについてDXF、3種類のrender、GT STEPのID集合が一致したため、Phase 0を完了とする。

### ✅ Phase 1: Hydra component config、入力、message builder

#### 実装するもの

- `zeroshot/configs/default.yaml`
- `manifest.py`
- `messages.py`
- 対応するunit test

#### 順序

1. `InputManifest`を実装し、必須input DXFとinput render mappingを保持する。全pathの存在とDXFの拡張子を検証し、render画像の拡張子は固定しない
2. `FeedbackManifest`を別に実装し、verification ID、必須execution feedback、optional feedback DXFとfeedback render mappingを保持する
3. 選択されたinput/feedback render bytesをそれぞれのmanifestから必要時に読めるようにする
4. `InputManifest`を受け取り、DXF pathを常に含め、render access configをconstructor引数に持つmessage builderで初回messageを構築する
5. `FeedbackManifest`を受け取り、存在するDXF/render feedbackだけを返すfeedback構築methodを作る
6. `default.yaml`からmessage builderを`hydra.utils.instantiate()`できることをtestする
7. 全config組合せをtable-driven testにする

dataset全体のsample列挙はPhase 1で抽象化しない。Phase 3でsample loopが必要になった時点で、まず`runner.py`内の短い処理として実装し、複数箇所から必要になるか複雑になった場合だけ関数またはclassへ抽出する。

`InputManifest`と`FeedbackManifest`は通常のfilesystem pathを持つ独立したimmutable snapshotとする。original inputはsample単位、feedbackはverification単位であり、ID、必須field、生成時期が異なるため一つのmanifestへ統合しない。独自のlogical path DTOや汎用`dto/`packageは作らない。生成artifactの管理抽象と配置も、具体的な重複が現れるまで延期する。

`DEFAULT_SYSTEM_PROMPT`と、renderの`none | path | image`およびfeedback種別に応じて追加する文言は`messages.py`に置く。現段階では別の`prompts.py`やHydra prompt configへ分割しない。system promptにはsampleに依存しない役割、出力contract、必要時にtoolを利用できることだけを簡潔に置き、具体的なtool名とschemaは各tool descriptionに置く。sample固有pathと選択されたmodalityの説明は`HumanMessage`側で組み立てる。`none`でも「提供されている場合はperspective renderを併用する」という条件付き一般文はsystem promptへ残してよい。

render styleの固定listを`src.data.render.config`からimportせず、pipeline内にも重複定義しない。MessageBuilderは選択されたaccess/feedback styleが`InputManifest.render3d_paths`のキーに含まれることを検証する。これにより現在の3 styleを固定せず、manifestへ追加された将来の画像styleへ対応する。`FeedbackManifest.render3d_paths`はその部分集合を許可する。

#### 必須test

- DXF pathが全ての初回messageに現れ、raw DXF textは現れない
- renderの`none`では個別のrender style、path、画像payloadがHumanMessageに現れない
- renderの`path`では`InputManifest`のpathとstyle名だけが現れる
- `image`では選択された画像だけが宣言順に添付される
- 複数画像は各style名のtext blockと対応するimage blockの順に並ぶ
- access styleとfeedback styleが混ざらない
- manifestのinput render keyにない選択styleと重複styleを拒否する
- feedback renderが3/2/1枚なら存在するstyleだけを返し、0枚ならrender blockを追加しない
- feedback mappingのvalueへ`None`を許可しない

#### 完了条件

modelとCADを使わず、Hydraからcomponentを生成し、全message payloadをtestで目視・比較できる。

#### 完了記録

2026-07-28にPhase 1を完了した。

- `InputManifest`と`FeedbackManifest`を分離し、それぞれのpathを検証してrender mappingをimmutableなsnapshotとして保持する
- `MessageBuilder`で初回入力とexecution/DXF/render feedbackを構築し、access/feedbackのmodeとstyleを独立に検証する
- render styleの固定listを設けず、各sampleのinput render mappingを利用可能styleのsource of truthとする
- Hydraの`ListConfig`をconstructor内でtupleへ正規化し、`default.yaml`から`MessageBuilder`を生成できる
- `tests/zeroshot`は42 test passed
- `ruff check`と`ruff format --check`はPhase 1対象の全fileで通過した

### ✅ Phase 2: Trusted CAD execution

#### 最初の対象

まずCadQueryだけでvertical sliceを完成させる。`cad_format`をcontractには含めるが、build123d対応はCadQuery executorの理解とtest完了後に同じinterfaceへ追加する。

#### 実装するもの

- AST/source validation
- trusted executorが実行ごとにUUIDのexecution IDを発番し、immutable sourceを保存
- owned/borrowedの両modeを持つ`SandboxWorkdir`
- kill可能な隔離subprocess
- `output.step`生成contract
- STEP再読込
- single solidとkernel validityの検証
- `CadQueryExecutionReport`

#### 重要方針

- model codeをpipeline process内で`exec`しない
- 後続の`run_shell`と`verify_output`内部のCadQuery実行が再利用する共通`SandboxRunner`をここで実装する
- `SandboxRunner`はLinuxの`bwrap --unshare-all`を必須とし、利用不能時に非隔離subprocessへfallbackしない
- `CadQueryExecutor`の`artifact_root`はtrusted runtimeだけが読書きする永続artifact置き場であり、sandboxの一時work directoryとして公開またはmountしない
- `SandboxRunner.run(command, workdir: SandboxWorkdir, timeout_s)`は`workdir.host_bind_dir`だけをread-writeの`workdir.sandbox_bind_dir`（初期値`/work`）として公開し、host pathのstaging方針を持たない
- trusted callerは`SandboxWorkdir.host_bind_dir`を通常の`Path`として直接操作する。file stagingやartifact回収のためだけのwrapper methodは追加しない
- `host_bind_dir=None`なら`SandboxWorkdir`が一時directoryを所有してcontext終了時に削除し、既存directoryを渡した場合は借用して削除しない
- Python専用runnerにはせず、`command`はsandbox内の`bash -c`へ渡す。Pythonおよびその子processは同じfilesystem/network境界に閉じ込める
- AST validationはPython構文だけを確認し、import allowlistを設けない。利用可能moduleとfilesystem accessはsandbox viewで制御する
- 固定fixtureを対象とするPhase 2ではsample入力をstageしない。active DXFと許可済みrender/feedbackのstagingは、具体的なtool contractを実装するPhase 3で追加する
- GT、repository、他sample、未許可render、credentialをsandboxへ渡さない
- wall timeout時はbwrap processをkillし、`--die-with-parent`で子processを残さない
- validation失敗とinfra failureを区別する
- 既存runnerのsecurity testを、コードではなくfixture/期待結果として参照する
- STEP検証は再import後のsingle solidとCadQuery/OpenCascadeの`isValid()`をPhase 2の最小条件とする。free edge検査やmesh watertightness監査は必要性を評価して後続で追加する
- sandboxが生成した`output.step`は、symlinkでない通常fileであることとSTEP validityをtrusted側で確認してからartifact directoryへcopyする。検証前の`shutil.copyfile()`でsandbox生成symlinkをhost側から辿らない

#### 必須test

- valid single box
- syntax error
- sandbox外fileを指定した動的importが失敗する
- filesystem探索で許可外入力、GT、repositoryへ到達できない
- host networkへ到達できない
- 子processを起動してもsandbox外filesystemへ到達できない
- owned workdirの削除、borrowed workdirの存続、同じworkdirを使う複数process間のfile永続性
- sandbox生成`output.step`がworkdir外を指すsymlinkなら拒否する
- STEP未生成
- zero solid / multi-solid
- invalid STEP
- infinite loop timeout
- 同じexecution artifactの上書き拒否

#### 完了条件

固定code fixtureだけを用い、安全に`CadQueryExecutionReport`とverified STEPを生成できる。

2026-07-30にPhase 2を完了した。

- `SandboxRunner`は`bwrap`でhost filesystemとnetworkを隔離し、fresh process、wall timeout、stdout/stderrの返却時切り詰めを提供する
- `SandboxWorkdir`は内部所有する一時directoryとcallerから借用する既存directoryの両方を、sandbox側の`/work`へ対応付ける
- `CadQueryExecutor`は元sourceをexecution ID配下へ保存し、sandbox用sourceにtrusted export epilogueを追加して`result`を`output.step`へexportする
- sandbox生成STEPはsymlink検査を含めtrusted側で再importし、single solidとkernel validityを確認したものだけを永続artifactへcopyする
- subprocess failure、timeout、sandbox infrastructure failureを別statusで返す
- `CadQueryExecutionReport`はsandbox内processのstdout/stderrを加工せず保持し、trusted executor側のvalidation/verification errorは`executor_error`へ分離する
- 実bwrapを用いるCadQuery boxのend-to-end testとsandbox生成symlinkの回帰testを含め、Sandbox/CadQuery対象は36 test passed
- `tests/zeroshot`全体は90 test passed
- `ruff check`と`ruff format --check`はPhase 2対象fileで通過した

### ✅ Phase 3: Fake modelによるLangGraph vertical slice

#### 実装するもの

- `tools/run_shell.py`
- `tools/load_image.py`
- `tools/verify_output.py`
- `verification/run_cadquery.py`
- `workflow/state.py`
- `workflow/graph.py`
- `pipeline/runner.py`
- `zeroshot/run_pipeline.py`を薄いHydra composition rootとして実装
- `pipeline/event_logging/normalizer.py`と`jsonl.py`による最小監査event変換・JSONL writer
- scripted fake chat model

Phase 2の`execution/run_code.py`は`verification/run_cadquery.py`へ移動する。`CadQueryExecutionReport`、source validation、STEP validationの責務は維持する。`tools/verify_output.py`がrun中共有する`SandboxWorkdir`から現在の`model.py`を読み、source文字列だけを`CadQueryExecutor`へ渡す。`CadQueryExecutor`は実行ごとに内部で一時`SandboxWorkdir`を作るため、候補scriptは共有workdir内の補助fileへ依存しないself-containedなprogramとする。sourceとverified STEPはverification ID配下へimmutable artifactとして保存する。

#### 作る順序

1. sample/agent用の`SandboxWorkdir`を一度作り、trusted callerがactive DXFと許可されたinput renderを`host_bind_dir`配下の固定pathへstageする
2. `SandboxRunner`と共有`SandboxWorkdir`をclosureに保持し、modelへ`command`だけを公開する`run_shell` toolを実装する
3. fake modelが`run_shell`で解析fileを作り、次の`run_shell`から同じfileを読めることを確認する
4. fake modelがworkdir rootへ`result`を定義した`model.py`を保存する
5. fake modelがtool callなしで完成を表明し、workflowが`verify_output(render_views=False)`で最終検証する
6. verified STEP、対応する`model.py`、監査logを保存し、feedback loopなしの直線経路を完成させる
7. fake modelが中間で`verify_output` toolを呼ぶ経路を追加する
8. その`ToolMessage`に含まれるexecution/STEP verification結果を受けて`model.py`を修正し、再度最終検証するloopを追加する
9. `load_image`でinput画像またはmodel自身のdump画像を確認する経路を追加する

`SandboxWorkdir`はtool callごとに作り直さない。processだけを毎回freshにし、同じ`host_bind_dir`を`/work`へbindする。stagingはworkdir作成直後の一度だけ行い、それ以降はmodelが作った中間fileをrun終了まで保持する。Phase 5でvisual feedbackを追加した後は、そのfileも同じworkdirへ蓄積する。host側の元fileやmanifestのsource pathをsandboxへ直接公開せず、`SandboxRunner`へstaging責務も追加しない。staging処理が一箇所にしかない間はapplication wiring内の短い`Path`/`shutil`操作として書き、専用classへ抽出しない。

Phase 3ではvisual feedbackを実装しない。trusted側のverification関数は将来の拡張点として`render_views: bool = False`を持つが、`True`は`NotImplementedError`とする。LangGraph tool schemaにはこのflagを公開せず、Phase 3のtool wrapperとworkflowはいずれも必ず`False`で呼ぶ。最初に直線経路を通してから中間検証loopを足すことで、graph/routingの問題とrendererの問題を分離する。

#### 必須test

- tool call IDとToolMessage IDが一致する
- `run_shell`のtool schemaが`command`だけを公開し、runtime dependencyをmodelへ公開しない
- `run_shell`がbash、Python、workdir内fileのread/writeを行える
- 別々の`run_shell` callから同じ中間fileを読み書きできる
- workdir内のinput copyを変更してもhost側の元fileへ影響しない
- `access_render3d: none`または未選択styleのrenderをworkdirから読めない
- `run_shell`と`verify_output`からGT、repository、他sample、networkへ到達できない
- `run_shell`のtimeoutと出力上限がrun全体を落とさない
- `load_image`がworkdir内画像だけを返し、path traversalとsymlink escapeを拒否する
- `run_shell`が生成したscratch fileをverified artifactへ自動昇格しない
- `verify_output`が共有workdir rootの`model.py`だけを読み、source snapshotをverification ID配下へ保存する
- `verify_output(render_views=False)`がverified STEPを生成する
- `verify_output(render_views=True)`が`NotImplementedError`を送出する
- 最初のfake model scenarioがfeedback loopなしの直線経路で完了する
- model主導の中間検証結果を`ToolMessage`としてagentへ返す
- tool callなしresponseの後にworkflowが最終検証を再実行する
- 最終検証失敗後はagentへ戻らず、未検証STEPを成功扱いせず終了する
- graph図が設計図と一致する
- eventsを順番に再生するとrunを説明できる
- Hydra composition rootが依存objectとmanifestを構築でき、moduleの`--help`が成功する

#### 完了条件

外部APIとvisual feedbackなしで、`SandboxWorkdir`への入力staging、複数回のshell作業、`load_image`、`model.py`の作成、workflow主導の最終検証、verified STEP保存までの直線経路を通す。その後、同じPhase内でmodel主導の中間検証結果を受けた修正loopまで通す。run終了後も`events.jsonl`、checkpoint、workdir snapshotと全verification attemptがsample artifact directoryに残り、成功runでは`attempts/`の最大verification IDから最終sourceとverified STEPを取得できる。

2026-08-01にPhase 3を完了した。

- scripted fake modelが複数回の`run_shell`、`load_image`、中間`verify_output`、tool callなしの完成表明を同じ明示的`StateGraph`上で実行する
- valid boxの直線経路と、構文errorの中間検証結果を受けてsourceを修正する経路を、実bwrap・実CadQueryでend-to-end検証した
- workflow主導の最終検証を常に最後に実行し、成功runでは最大verification IDのsourceとSTEPを最終提出物として保存する
- `events.jsonl`は逐次flushされ、message、node、tool、verification、正常・異常終了を順序付きで記録する。SQLite checkpointは同じrun IDで保存する
- `run_pipeline.py`はHydra composition rootに限定し、dependencyとmanifestの構築、およびmoduleの`--help`をtestした
- `tests/zeroshot`は147 test passed、`ruff check`、`ruff format --check`、`git diff --check`も通過した

### ✅ Phase 4: Gemma/Ollama、GPT/Codex、Qwen/SGLangを接続する

#### 実装するもの

- LangChain `BaseChatModel`を共通boundaryとするHydra model config
- `configs/model/gemma4_ollama.yaml`
- `configs/model/gpt5_6_luna_codex.yaml`
- `configs/model/qwen3_6_sglang.yaml`
- SGLang reasoning deltaだけを補う`SGLangChatOpenAI`
- model profileを差し替えられる独立したSGLang server launcher
- credential redaction
- stdout/stderrの設定可能な返却時切り詰め
- 通常testから分離したbackend capability smoke test
- one-sample live CLI
- Rich consoleへのnode、tool、verification、model text/reasoningのlive表示
- 成否にかかわらず残るsample workspace

SGLang serverはpipelineと別process、可能なら別environmentで起動する。pipeline側は`base_url`、`model`、認証情報だけを受け取り、`sglang[all]`へ直接依存しない。

CPU、memory、PID数、workdir size、in-flight stdout/stderr bufferのhard limitは、必要な制限値をlive runから決められないためPhase 4から外す。現行のwall timeout、bwrap隔離、返却時切り詰めで観測を続け、具体的な障害が出た場合に追加する。

#### capability test

`ZEROSHOT_LIVE_MODEL_CONFIG`で明示的に有効化する共通live testとone-sample pipelineで、backendごとに必要な範囲を実測する。

- `bind_tools()`したschemaを認識する
- structured tool argumentsを返す
- multimodal initial user messageを扱える
- imageを含むtool resultの共通形式を扱える
- token usage、finish reason、tool call IDを取得できる
- pipeline上で`run_shell`へcommandを渡し、結果を次turnで利用できる
- 複数turnにわたるshell/image/verification tool callを扱える

#### live smoke test

最初は1 sample、1 model、CadQueryで実行する。成功率を論じる段階ではなく、workdirの変更、message、tool call、verification、state遷移を一行ずつ追う。

#### 完了条件

GPTでは、DXF解析、workdir内file編集、画像参照、中間検証、toolなし完成表明、workflow主導の最終検証まで、一サンプルが同じgraph上で完走する。

Qwen/SGLangでは、同じgraphとtool schemaを変更せずにDXF解析、workdir内file編集、画像tool result、複数turn、中間検証まで進み、検証済みSTEPを生成できることを確認する。toolなし完成表明へ収束するかはmodelの推論挙動であり、backend統合の完了条件とは分ける。未収束runは成功runとせず、event logへ実際の停止理由を残す。

Gemma/OllamaはGPU serverを必要としない開発用backendとして、共通capability testと一サンプルのpipelineが動くことを確認する。

#### 完了記録

2026-08-02にPhase 4を完了した。

- Gemma 4をOllamaのOpenAI-compatible endpointから`ChatOpenAI`で呼び、共通live testとsample `000364`のpipeline動作を確認した
- GPT-5.6-lunaを`_ChatOpenAICodex`で呼んだsample `000364`は、DXF解析、`model.py`作成、model主導の中間検証`000`、toolなし完成表明、workflow主導の最終検証`001`を通り、`VERIFIED`で完走した。監査記録は`outputs/zeroshot_gpt5_6_luna/000364/events.jsonl`にある
- Qwen 3.6 35B-A3B FP8を別GPU環境のSGLang serverで起動し、SSH tunnel越しに同じpipelineへ接続した。DXF解析、`load_image`による画像参照、workdir編集、失敗した検証`000`から`003`を受けた修正、model主導の検証`004`と`005`の`VERIFIED`まで確認した。その後もmodelが解析を継続したため手動停止しており、toolなし完成表明とworkflow最終検証は未確認である。監査記録は`outputs/zeroshot_qwen3_6_nothinking_bounded/000364/events.jsonl`にある
- Qwen runで判明したSGLang固有のreasoning stream欠落は`SGLangChatOpenAI`で正規化し、thinkingの有効/無効とmodel turnごとの総出力上限をHydra override可能にした
- `event_logging/`はcanonical JSONLとRich console表示を分離し、model text、reasoning、tool callをstream中に表示する。未知のevent/contentも黙って捨てずfallback表示する
- sample workspaceをartifact directory上で直接使うことで、正常終了時だけでなく例外・手動停止時のmodel生成fileも保持する
- `python -m pytest tests/zeroshot`は157 passed、3 skipped。3件はbackendを明示しない通常実行ではskipされるlive testである。`ruff check zeroshot tests/zeroshot`も通過した

### Phase 5: Feedback modeとrenderer calibration

Phase 3/4で未実装としていた`verify_output(render_views=True)`をここで実装する。LangGraphの遷移は変えず、verified STEPに対するvisual feedback生成と提示方法を追加する。

#### 実装順

1. `verification/render.py`を追加し、verified STEPから三面図DXFとperspective viewsを生成する
2. `verify_output(render_views=True)`から、STEP検証成功時だけrendererを呼ぶ
3. 構造化reportと生成に成功したDXF/render pathから`FeedbackManifest`を作る
4. `FeedbackManifest`をtool resultのJSONへ混ぜず、workflow内のtrusted artifactとしてfeedback message構築まで渡す
5. 生成成功時、三面図DXF pathをfeedback user messageへ常に入れる
6. workdirから元DXFと各verificationのfeedback DXFを参照できるようにする
7. `feedback_render3d: none`
8. `feedback_render3d: path`
9. `feedback_render3d: image`
10. style選択、部分失敗、message形式の組合せtest
11. identical STEP再renderと既知形状fixtureでrendererの再現性を測る

renderer本体は再実装せず、trusted側から以下を呼ぶ。

- `src.data.render.techdraw.generate_techdraw`
- `src.data.render.render3d.generate_render3d`
- `src.data.render.config`のpath/style contract

これらは同期処理なので、kill可能なchild processとtimeoutを設ける。CadQuery実行またはSTEP検証に失敗した場合はrenderしない。renderだけ失敗した場合は、実行・STEP検証の成功とrender failureを別statusで返す。

初期版では、再投影DXFの自動IoU閾値によるverification失敗判定を行わない。まずartifactをmodelへ返し、精度向上効果を測る。数値projection checkは後続の独立機能とする。

#### 必須test

- verified STEPとprojection成功時にfeedback DXF pathを必ず返す
- STEP未生成またはinvalid STEPではDXF/renderを生成しない
- DXF pathだけではraw DXF内容を自動注入しない
- workdirが元DXFと同じagentのverification feedbackだけを参照できる
- `image`で選択styleだけを添付する
- render生成が部分失敗した場合は成功したstyleだけを返し、0枚ならrender blockを作らない
- input画像とverification feedback画像を取り違えない
- renderer timeoutをexecution failureと混同しない
- feedback artifactがverification directory外へ出ない

#### 完了条件

必須DXF feedbackとrender access/feedbackの全modeが、fake modelと両live backendで意図したmessageになる。

### Phase 6: 評価と段階的アブレーション

#### 推論と評価の分離

推論runを完了・closeした後、別CLIでGT評価する。

```bash
python -m zeroshot.pipeline.evaluation.score_run \
  --run-dir <run_dir> \
  --target-dir <target_dir>
```

evaluatorは`src.evaluation.scoring`と`src.metrics`をimportして使い、metric実装をコピーしない。

#### 最初に報告する指標

- prediction coverage
- code execution success
- valid single-solid率
- normalized Voxel IoU
- normalized ECCV surface/edge/vertex/topology F1
- BoundingBox error
- unnormalized metricは参考値として別欄
- model call数、token、CAD実行数、wall time

自己feedbackで使う再投影DXFやrenderは、GT metricではない。自己整合性が高くても正しい3D形状とは限らないため、別欄に分ける。

#### データ運用

- debug set: 実装確認用の少数固定ID
- locked evaluation set: promptや閾値調整に使わない固定ID
- failed/missing predictionを集計から落とさない
- infra errorとmodel reconstruction failureを分ける
- sample ID list、model、sampling、config、code versionをrun manifestへ保存する

#### アブレーション順

全configの直積を最初から回さない。次の順で原因を分離する。

1. render access modality: `none/path/image`
2. access render style
3. render feedback: `none/path/image`
4. feedback render style
5. DrawingIRなし/あり

比較は可能な限り同一sampleのpaired resultで行う。小規模runでは平均値だけで結論を出さず、sample別の改善・悪化とfailure classを読む。

#### 完了条件

少なくとも一つの変更について、「どのsampleで、どのmetricが、どのcost増加とともに変化したか」を説明できる。

### Phase 7: DrawingIRを理解してから追加する

DrawingIRは削除対象ではなく、重要な将来の精度改善候補である。ただし既存codeを一括コピーしない。

#### `drawing/extract.py`

確認する内容:

- DXF entity typeとanalytic geometryへの変換
- layerからのfront/top/right view割当
- hidden linetype解決
- loop抽出
- view間correspondence
- degenerate entityの扱い

小さなDXF fixtureを一種類ずつ追加し、期待するIRを手で確認する。

#### `drawing/summary.py`

確認する内容:

- 何をpromptへ残し、何を省略するか
- hidden lineが存在しない図面の表現
- truncationによって重要featureが落ちないか
- 同じIRから常に同じsummaryを作れるか

golden testでsummary全文をreview可能にする。

#### `drawing/pose.py`

確認する内容:

- front/top/rightとworld axisの対応
- right-handed mapping
- 対称形状や同一extent軸による複数解
- scale/translation/rotationの責務
- 最初のmappingを無条件採用しないこと

cube、直方体、軸対称形状、非対称形状をfixtureにする。

#### 導入方法

```yaml
use_drawing_ir: false
```

で無効化可能にする。modelが`run_shell`から自由にezdxf解析するbaselineと同一条件で比較し、次のどちらかを説明できた場合のみ標準化を検討する。

- GT metricを改善する
- failure解析を明確に改善する

単体testが通るだけでは標準有効化しない。

### Phase 8: 後続拡張

#### Gemini/Claude CLI backend

- LangChain `BaseChatModel`としてworkflowへ渡せるadapterを作る。これが不可能または不自然な場合だけ共通model protocolを抽出する
- CLIのnative toolを使わせるか、LangGraph toolだけに制限するかを実測して決める
- native toolを使う場合はprovider traceをcanonical event logへ正規化する
- credential stagingとjailはbackend内部へ閉じ込める
- 旧runnerの実装を丸ごと再利用せず、必要なsecurity contractとtestだけをgit履歴から抽出する

#### 並列agent

- 単一agent loopをsubgraph化する
- agentごとに独立`SandboxWorkdir`とartifact namespaceを持つ
- merge policyと最終選択基準を明示する
- その段階でrun-level `BudgetPolicy`を設計する

#### 数値projection feedback

- identical STEP再投影によるrenderer ceiling
- wrong-shape negative control
- hidden lineなしview
- missing/degenerate view
- per-view scoreとaggregate score
- thresholdによるfalse reject

をcalibrateしてから追加する。単一のmean IoUだけで候補を捨てない。

## 9. 出力contract

Phase 3で実装するsample単位のcontractは次のとおりとする。

```text
<out_dir>/
└── <sample_id>/
    ├── events.jsonl
    ├── checkpoints.sqlite
    └── workspace/
        ├── inputs/                  # run開始時にstageした入力
        ├── model.py                 # modelが最後に編集したactive source
        ├── ...                      # modelのscratch file
        └── attempts/                # sandboxからread-onlyのpipeline管理領域
            └── <verification_id>/
                ├── model.py
                └── output.step      # verification成功時だけ存在
```

model主導の中間検証とworkflow主導の最終検証は同じ`attempts/`へ単調増加するIDで保存する。workflow主導の最終検証が最後に実行されるため、`model.py`が存在するrunでは最大verification IDが最終提出物を表す。成功はgraph stateと`events.jsonl`に記録された最終status、およびそのdirectoryの検証済み`output.step`で判断する。`final/`への二重copyは行わない。

`workspace/`のうち`attempts/`以外はmodelが自由に変更できるscratchのrun終了時snapshotであり、検証済みartifactとして扱わない。`attempts/`はhost側がdirectoryと内容を作成し、sandboxからread-onlyにする。`events.jsonl`と`checkpoints.sqlite`はmodelから書き換えられないsample artifact rootへ保存する。

`run_manifest.json`、`run_summary.json`、`resolved_config.yaml`、集約用artifact manifestなどのrun全体の出力は、複数sampleを集約して評価するPhase 6で追加する。Phase 4の一sample live runではsample単位の出力contractを変えない。

## 10. Test方針

### Unit test

- config validation
- `InputManifest`と`FeedbackManifest`のpath、拡張子、必須/optional field、読取validation
- message builderの全mode
- tool schemaと引数validation
- routing
- canonical event serialization
- sandbox result / CadQuery execution report / verification report

### Integration test

- fake modelによる完全なagent loop
- real CadQuery boxの隔離実行
- STEPからDXF/render生成
- scorerによるGT self-checkとinvalid prediction

### Security / leakage test

- target path拒否
- symlink escape拒否
- `run_shell`と解析scriptはactive DXF、configで許可されたinput render、同じworkdirのfeedbackだけを読める
- 最終`model.py`のDXF参照をsource validationで拒否する
- `run_shell`、解析script、`model.py`はrepository、GT、他sample、未許可renderを読めない
- network accessを拒否する
- modelが起動した子processも同じfilesystem/network境界に閉じる
- credentialがlog/artifactへ残らない

### Live test

Gemma/GPT/Qwenのlive testは通常のtestから分離し、`ZEROSHOT_LIVE_MODEL_CONFIG`でbackendを明示した場合だけ実行する。quota、認証、service outage、remote server停止をmodel精度失敗として数えない。

基本commandは次とする。

```bash
python -m pytest -q tests/zeroshot
ruff check zeroshot tests/zeroshot
python -m zeroshot.run_pipeline --help
```

## 11. 既知のriskと回帰確認

旧実装のgit履歴から研究機能を参照する場合、少なくとも以下を先にfixture化する。

- hidden lineが0本でも正常な図面
- BYLAYER linetype解決
- 同じextentを持つ軸が複数あるpose
- left/rightまたはfront/backが曖昧な形状
- missing viewとdegenerate view
- renderer outputがartifact root外を指さないこと
- diff画像pathと実ファイルの一致
- empty hidden maskをIoU失敗として扱わないこと
- mean scoreが一つの悪いviewを隠さないこと

また、DXF analysis codeの大きなstdout/stderrは実際にcontextを圧迫したため、返却時のbyte上限をrun configで調整できるようにした。in-flight buffer制限は未実装であり、host memory問題が観測された場合に再検討する。provider間のimage tool result差分は共通live testで検知し、OCCがnative hang/crashする可能性はwall timeoutとprocess隔離で扱う。

## 12. 実装順の要約

```text
Phase 0  dataset・既存評価器の基準線
   ↓
Phase 1  Hydra config・input・message
   ↓
Phase 2  trusted CAD execution
   ↓
Phase 3  fake model LangGraph vertical slice
   ↓
Phase 4  Gemma/Ollama・GPT/Codex・Qwen/SGLang backend
   ↓
Phase 5  feedback mode・renderer calibration
   ↓
Phase 6  評価・アブレーション
   ↓
Phase 7  DrawingIR
   ↓
Phase 8  CLI backend・並列agent・数値projection
```

最初の実装着手点はPhase 0であり、フォルダを一括作成することではない。各phaseの完了時に、実装者がcode walkthroughを行い、次のphaseへ進む前に「何が動き、何がまだ動かないか」をREADMEへ追記する。

## 13. Finalize状態

設計上のopen questionは現時点で残していない。実装中に新しい事実が判明した場合は、既存の決定を暗黙に変更せず、短いADRまたは本計画の変更履歴として理由と影響範囲を記録する。
