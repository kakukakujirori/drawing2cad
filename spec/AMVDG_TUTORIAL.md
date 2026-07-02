# AMVDG v0.3 チュートリアル: JSONの読み方と2D図面の復元

このチュートリアルでは、`spec/example_flange_v0.3.json` を題材に、AMVDG (Annotated Multi-View Drawing Graph) がどのような構造になっており、そこからどのように2D図面が復元されるのかを解説します。

> **注記 (v0.3)**: 以下のコードスニペットは `example_flange_v0.3.json`（`spec/flange.step` をレンダラに通した実出力：プレート⌀120・ハブ⌀76・ボア⌀40・ボルト穴⌀11×6）から実際に抜き出したものです。ただし `features[].members` のように長い配列は `…` で省略しています（実ファイルは全メンバを持ちます）。v0.3 で追加された `prov.topo_origins`（2Dプリミティブの3D由来）・中心線プリミティブ・円弧の `start_angle`/`end_angle` は [`spec/README.md`](README.md) の **「v0.2 → v0.3」節**（`## v0.2 → v0.3`）を参照してください（ルートの `README.md` ではなく `spec/` 内の README です）。
>
> **用語の対応**: JSON では第4レイヤー「provenance（来歴）」は各要素の `prov` キーとして実体化されます（トップレベルに `provenance` というキーはありません）。`prov.topo_origins` はその中に入ります。この来歴レイヤーは **synthetic-GT 専用**で、実図面から推論する 2D→AMVDG レッグの出力には現れません。次章の `features`（第3レイヤー correspondence）とは**別物**です — 関係は §5.1 を参照。

## 1. 全体構造 (The Envelope)

JSONのルートには、図面全体のメタデータが格納されています。

```json
{
  "amvdg_version": "0.3",
  "part_id": "FLANGE",
  "profile": "vectorized",
  "sheet": {
    "size": "A3",
    "projection": "third_angle",
    "scale": [1, 2],
    "width_px": 1800,
    "height_px": 1273,
    "px_per_mm": 4.2857
  },
  ...
}
```
ここから、この図面が **A3サイズ**、**第三角法**、図面スケール **1:2** で描かれており、画像化された際の解像度が **1800x1273 px**（1mmあたり約4.28ピクセル）であることが分かります。この `px_per_mm` が、後述の2D座標を実寸にスケールする際の基準になります。`profile` はこのグラフの完成度（`vectorized` = 座標付き）を示します（詳細は `spec/README.md` の検証プロファイル節）。

## 2. ビューの定義 (Views)

`views` 配列には、正面図 (front)、上面図 (top)、右側面図 (right) などが含まれます。

```json
{
  "name": "front",
  "view_type": "front",
  "projection_dir": [0, -1, 0],
  "frame": {
    "origin_px": [220.71, 805.71],
    "scale": 0.5,
    "px_per_mm": 2.1429,
    "axis_remap": { "matrix": [[1, 0], [0, -1]], "px_x": "+X", "px_y": "-Z" }
  },
  "primitives": [ ... ]
}
```
* `projection_dir`: このビューを3D空間上でどの方向から見ているか（part→viewer の単位ベクトル。正面図ならY軸マイナス方向）。
* `frame.origin_px`: このビューの原点が、画像（シート）全体のどのピクセル座標に配置されているか。
* `scale`: このビューの図面スケール（ここでは 1:2 = 0.5）。`px_per_mm` はシート全体の値にこの scale を掛けたビュー実効解像度。
* `axis_remap` (v0.3): ピクセル軸 ↔ 符号付きモデル軸の対応（正面図は画像X=+X、画像Y=−Z）。2D座標を3Dに逆変換する際の鍵で、`prov.topo_origins` と組で使います。

## 3. 2Dプリミティブの復元 (Geometry)

各ビューの `primitives` 配列に、そのビューを構成するすべての線や円が定義されています。ここから2D図面が完全に復元できます。

```json
{
  "id": "F0",
  "type": "line",
  "line_role": "visible",
  "feature_tag": "outline",
  "p1": [220.71, 887.14],
  "p2": [477.86, 887.14],
  "prov": {
    "topo_origins": [
      { "dim": 2, "id": "Face_11", "role": "edge-on" },
      { "dim": 1, "id": "Edge_0",  "role": "edge" },
      { "dim": 2, "id": "Face_0",  "role": "parent_face" }
    ]
  },
  "feature_id": "cyl0"
}
```
* **復元プロセス**: 描画スクリプト（SVGレンダラなど）は、この `type="line"` を見て、画像上の `p1` から `p2` に向かって直線を引きます。
* **線のスタイル**: `line_role="visible"`（実線・外形線）、`"hidden"`（破線・隠れ線）、`"center"`（一点鎖線・中心線）などの役割に応じて線のスタイル（太さ、破線パターン）を適用します。
* **`prov.topo_origins` (v0.3, GT専用)**: この1本が3Dのどの実体から投影されたか。ここではプレート円柱 `Face_0` の底面 `Face_11` がエッジオンに潰れた稜線（`Edge_0`）であることを示します。実図面には無い情報で、2D復元自体には不要ですが、3D復元とビュー間対応の根拠になります（§5.1）。
* **`feature_id`**: この線が属する3Dフィーチャ（`cyl0`）への逆引き。

すべてのビューのすべてのプリミティブをループして描画するだけで、完璧な2Dの三面図（寸法抜き）が完成します。`prov`/`feature_id` は描画には使いません。

## 4. 寸法と注釈 (Annotations)

図面に寸法（Dimension）を描画し、かつその寸法が「どのジオメトリを測っているか」を定義するのが `annotations` 配列です。

```json
{
  "id": "D1",
  "kind": "linear",
  "subtype": "horizontal",
  "param_role": "width",
  "value": 120.0,
  "view": "front",
  "refs": ["F5", "F1"],
  "feature_id": null,
  "prov": { "feature_id": "bbox", "param": "dx", "origin": "synthetic_gt" }
}
```
* この寸法（`D1`）は、正面図（`view="front"`）に描かれています。
* **意味**: プリミティブ `F5` と `F1` の間の水平距離 (`subtype="horizontal"`) が `120.0` mm であることを示します。`param_role="width"` と `prov.param="dx"` から、これは部品バウンディングボックスの幅（X方向）寸法だと分かります。
* **復元プロセス**: レンダラは、`F5` と `F1` の座標を取得し、その間をまたぐように「寸法線」と「120」というテキストを描画します。
* 直径寸法は `kind="diameter"` になり、対応する円の `feature_id`（例 `cyl0`）を持ちます（例: フランジ外形 ⌀120 は `D4`）。

## 5. ビュー間の対応関係 (Features & Correspondences)

AMVDGの最大の特徴であり、2Dから3Dへの復元に不可欠なのが `features` 配列です。

```json
{
  "feature_id": "cyl2",
  "kind": "cylinder",
  "axis": "Z",
  "r_mm": 5.5,
  "members": [
    { "view": "top",   "primitive_id": "T5",  "projection_role": "visible" },
    { "view": "front", "primitive_id": "F34", "projection_role": "hidden" },
    { "view": "right", "primitive_id": "R15", "projection_role": "hidden" }
  ],
  "prov": { "feature_id": "cyl2", "feature_3d": "cylinder",
            "occ_face_ids": ["Face_3"], "params": { "r_mm": 5.5, "axis": "Z" } }
}
```
（上の `members` は各ビュー1本ずつに抜粋。実ファイルの `cyl2` は16メンバを持ちます）

* **意味**: `cyl2` という1つの3Dフィーチャ（半径5.5=⌀11のボルト穴、軸Z）が、
  * 上面図（top）では `T5`（円）として見え、
  * 正面図（front）では `F34`（隠れ線）として見え、
  * 右側面図（right）では `R15`（隠れ線）として見えている。
* **なぜ重要か**: 人間は図面を見て「この上面図の円は、正面図のこの破線対のことだな」と無意識に結びつけますが、AIやプログラムにはそれが分かりません。`members` に各ビューの `primitive_id` をリストアップすることで、**「異なるビューに存在するこれらの2D線は、すべて同じ1つの3D形状から投影されたものである」**という確固たる対応関係（Correspondence）を定義します。
* **どう束ねたか (v0.3)**: この対応は手作業ではなく、各メンバの `prov.topo_origins` が同じ B-rep 面（ここでは `Face_3`）を由来に持つことから機械的に導出されています（§5.1）。

### 5.1 `features` と `prov.topo_origins` の違い（v0.3）

よくある誤解ですが、この2つは**別レイヤーの別物**で、「3D→2D か 2D→3D か」という方向の違いではありません。

| | `features[]`（第3レイヤー: correspondence） | `prov.topo_origins`（第4レイヤー: provenance, v0.3新規） |
|---|---|---|
| 単位 | 3Dフィーチャ1個 = 1エントリ | 2Dプリミティブ1本 = 1リスト |
| 中身 | `members[]{view, primitive_id, projection_role}` — **ビュー横断**で同一フィーチャに属する線を列挙 | `[{dim, id, role}]` — その1本が投影されてきた **B-rep実体**（`Face_0`, `Edge_2` …）と役割（edge / silhouette / boundary / edge-on / parent_face / axis） |
| 生成 | topo_origins から**導出**される（成果物） | オラクル投影器が算出する（機構・一次情報） |

関係は「**topo_origins が機構、features がその成果物**」です。レンダラは各線の topo_origins（共有 B-rep 面ID）を突き合わせて、`members` をビュー横断で束ねています。例えば `cyl0`（プレート外形の円柱）の `members` には、正面図の外形線群・上面図の外形円・右側面図の外形線群がすべて入りますが、これは「これらの線が同じ `Face_0` を由来に持つ」と topo_origins が語っているからです。

学習での立場も異なります:
- `features` は 2D→AMVDG レッグで**モデルが予測すべき**対応レイヤー。
- `topo_origins` は `prov` 配下 = **GT専用**で、実図面には B-rep 面IDが写っていないため 2D→AMVDG の出力には出しません（2D→3D レッグの教師信号としては使えます）。

なお各プリミティブは自身の所属を `feature_id`（例 `"feature_id": "cyl0"`）としても持っており、`features[].members` への逆引きになっています。

## 6. まとめ

1. **Geometry (`views[].primitives`)** を読めば、そのまま2Dグラフィックとして図面をピクセル単位で復元できます。
2. **Annotation (`annotations`)** を読めば、その図面に書き込まれた寸法値と、それがどの2D線を指しているかが分かります。
3. **Correspondence (`features`)** を読めば、別々のビューに描かれた2D線同士が「3D空間では同じ1つのフィーチャである」という関係性が分かり、2D→3Dの推論（Z方向の奥行きの特定など）が可能になります。
