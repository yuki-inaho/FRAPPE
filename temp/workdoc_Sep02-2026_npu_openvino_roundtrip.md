# 作業計画書 兼 記録書: FRAPPE を Intel NPU / iGPU 上で OpenVINO 経由で回す

---

**日付：** `2026年09月02日`
**作業ディレクトリ・リポジトリ:** `/home/inaho/Project/FRAPPE`（Git リポジトリ、ブランチ `npu`、起点 `2426b37`）
**作業者：** `yoshikawa@inaho.co` / Claude Opus 5

**前提メモ:** 本書の前段として `temp/memo_Sep02-2026_npu_offload_requirements.md` がある。
ハードウェア構成、既存成果物、元論文の要点、実測済みの数値はそちらに書いてある。
本書はそれを踏まえた**実行計画**であり、事実の列挙は繰り返さない。

---

## 1. 作業目的

* **目標1:** FRAPPE コーデックの**エンコード → JPEG-LS 圧縮 → デコード復元**という一気通貫の経路を、
  この NUC14 Pro 上で OpenVINO 経由の NPU / iGPU / CPU 実行として成立させる。
* **目標2:** どのデバイスに何を載せると実際に速いのかを、**同一画像集合・同一規約で測って**確定させる。
  「NPU に載せたから速い」ではなく、測った数字で配置を決める。
* **目標3:** （副目標）JPEG-LS 符号化の規則的な前半（MED 予測・勾配量子化・コンテキスト ID）を
  OpenVINO グラフとして iGPU／可能なら NPU で計算し、CPU の Golomb 符号化へ渡す経路を作る。
  参考実装 `npu-jpegls-offload` が定めた 2 出力契約をそのまま踏襲する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:**
  「NUC14 Pro の NPU を使って、エンコーディング + JPEG-LS 圧縮 → 復元までここでやってもらおうと思っている」。
  つまり、学習済み FRAPPE コーデックを**この端末のアクセラレータで動く配備物にする**こと。
  学習は終わっており、CR 40 / CR 50 の 2 動作点の重みと来歴は手元にある。ここから先は配備の作業。

* **明示要求:**
  1. Intel NPU 上でエンコードを行う。
  2. JPEG-LS 圧縮を行う。
  3. デコードして復元まで行う。
  4. （必須ではない）JPEG-LS も `npu-jpegls-offload` を参考に、OpenVINO 経由で iGPU／できれば NPU で符号化する。
  5. 実データは私有 RGB 62 枚を使う。**データ名・ファイルが staging されないようにする。**
  6. main 側に入った ONNX 周りの更新を取り込んだ上で作業する。→ 取り込み済み（`2426b37`）。

* **暗黙制約:**
  * パッケージ管理は **pixi**（本リポジトリの規約。テンプレートの `uv run` は `pixi run` と読み替える）。
  * **暗黙 fallback 禁止**: `--device NPU` を要求して NPU が使えないときに黙って CPU に落ちない。
    参考実装 `npu-jpegls-offload/docs/DESIGN_JA.md` §5 の `npu` / `auto` の区別をそのまま採る。
  * **ビット完全性の維持**: 符号器の出力平面は参照実装（`entropy_coding.arrange_latents` + shift）と
    バイト一致し、そこから作った JPEG-LS ペイロードも参照ビットストリームとバイト一致すること。
    `tools/export_onnx.py` が既にこの水準で自分を検証しており、デバイスを変えても水準を下げない。
  * **私有データの非混入**: パス・ファイル名・画素のいずれも git 追跡下に置かない。
    私有パスは `.private/local_paths.env`（gitignore 済み）に隔離し、文書とコードは
    環境変数経由でのみ参照する。本リポジトリには既に同じ規約がある（`MANAGED_TRAINING.md`）。
  * **監査可能性**: 主張はすべて再実行可能なコマンドと機械可読な JSON に紐づける。
  * **レート規約の明示**: bpp は長さ接頭辞なし（`evaluate_joint_prefix` 系）で統一し、
    JSON には接頭辞込みの値も併記する（`2426b37` が定量化した 3.29e-4 bpp の差のため）。

* **非ゴール:**
  * 再学習・fine-tune。重みは凍結して扱う。
  * 新しい動作点（CR 以外の点）の作成。
  * near-lossless JPEG-LS（`NEAR>0`）、10/12/16 bit、インターリーブ、リスタートマーカー。
  * ONNX 以外の配備形式（OpenVINO IR の手書き、TFLite 等）。
  * FRAPPE のモデル構造そのものの変更。

* **成功条件:**
  1. 私有評価データ 1 枚を入力して、`encode(device) → uint8 平面 → JPEG-LS → decode(device) → 再構成 PNG`
     が最後まで通る。CR 40 / CR 50 の両動作点で通る。
  2. OpenVINO の各デバイスが出す符号平面が、PyTorch 参照とバイト一致する（NPU / iGPU / CPU すべて）。
  3. 62 枚全体で PSNR / bpp / CR が測定され、既知の値（CR50: 28.63 dB / CR 50.53、
     CR40: 29.78 dB / CR 40.47）と**同じ規約で**比較できる形の JSON が残る。
  4. encoder / decoder × CPU / iGPU / NPU の 6 通りのレイテンシ表が測定済みで再現可能。
  5. `--device NPU` を指定して NPU が使えない状況では、CPU に落ちずに明示的に失敗する。

* **リスクと前提:**
  * **R-A**: 新しい ONNX は `pixel_unshuffle`（→ SpaceToDepth）と `repeat_interleave`（→ Tile 系）を含む。
    旧グラフは einops 由来の Reshape だったので、**NPU プラグインの対応は未確認**。落ちたら
    デコーダの adapt 部分をグラフ外（CPU）に出すか、`--io float` 形を試すかの分岐が要る。
  * **R-B**: NPU は静的形状を要求する。動的グラフを reshape してから compile する手順が必須。
    旧グラフでは成功済みだが、新グラフの symbolic dim（`32*units_h` 形式）で同じ手が効くかは未確認。
  * **R-C**: pixi の既定環境は CUDA 12.8 版 torch（820 MB）+ nvidia_* ホイール 15 個を引く。
    この機械に NVIDIA GPU は無く、実測 ~960 KB/s なので完了に 1 時間規模かかる。
    torch が要るのは ONNX 再エクスポートと参照検証とテストだけ。
  * **R-D**: 副目標の JPEG-LS オフロードは、CharLS（C++）が既に速いため**純粋な速度では負ける可能性が高い**。
    価値があるとすれば CPU 解放と電力。主目標を人質に取らない順序で進める。
  * **R-E**: 配布アーカイブ同梱の ONNX は旧契約。再エクスポートしないと平面形状も dtype も違う。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-0 | 実行環境の確立と前提の再生成 | 目標1の前提 | pixi 既定環境 / 新契約 ONNX (CR40, CR50) | `pixi run test` が通り、`export-onnx` の自己検証が通る |
| SG-1 | OpenVINO 上でのデバイス制約を確定 | 目標1・目標2 | デバイス別対応表（JSON + 本書への追記） | 各デバイスで compile と 1 回の推論が成功し、失敗は理由付きで記録される |
| SG-2 | デバイス選択可能な OpenVINO ランタイム | 目標1 | `src/compressors/frappe/openvino_runtime.py` | 単体テストがビット一致と明示失敗を検証する |
| SG-3 | ラウンドトリップ CLI | 目標1 | `tools/roundtrip_openvino.py` | 1 枚で完走し、参照とバイト一致、再構成 PNG が出る |
| SG-4 | デバイス別ベンチマーク | 目標2 | `tools/benchmark_openvino_devices.py` + 結果 JSON | 6 通りのレイテンシが再現可能な形で残る |
| SG-5 | 62 枚での RD 評価 | 目標2 | 評価 JSON（PSNR / bpp / CR、両レート規約） | 既知の CR40/CR50 値と同一規約で比較できる |
| SG-6 | JPEG-LS 前半の OpenVINO オフロード | 目標3（副） | `src/compressors/frappe/jpegls_openvino.py` + テスト | CPU 参照と MED/Qs が完全一致し、`.jls` が ffmpeg で復号できる |
| SG-7 | 記録と監査性 | 全体 | 本書の作業記録 + 全 JSON | 各 Trace ID に証跡が対応する |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | NPU 上でエンコードする | SG-1, SG-2, SG-3 / 手順 6, 9, 12 | `roundtrip_*.json` の `encoder.device == "NPU"` と `execution_devices` |
| TR-2 | JPEG-LS 圧縮を行う | SG-3 / 手順 12 | `roundtrip_*.json` の `bitstream_bytes`, `bpp_payload_only`, `bpp_with_length_prefix` |
| TR-3 | デコードして復元する | SG-2, SG-3 / 手順 10, 12 | 再構成 PNG と `psnr_db` |
| TR-4 | ビット完全性を落とさない | SG-2 / 手順 8, 9, 11 | `tests/test_openvino_runtime.py` と `roundtrip_*.json` の `plane_mismatched_symbols == 0` |
| TR-5 | 暗黙 fallback 禁止 | SG-2 / 手順 7 | `test_explicit_device_failure_is_not_silently_downgraded` |
| TR-6 | 私有データを staging しない | 全フェーズ / 手順 1, 19 | `git status --porcelain` が私有物を出さないこと、`.private/` の gitignore 判定 |
| TR-7 | レート規約の明示 | SG-3, SG-5 / 手順 12, 16 | JSON に両規約の bpp が併記されていること |
| TR-8 | 測って配置を決める | SG-4 / 手順 14 | `benchmark_devices.json` の 6 通りのレイテンシ |
| TR-9 | main の ONNX 更新に追従する | SG-0 / 手順 3, 4 | 新契約 ONNX の `report.json`（`io: "uint8"`, symbolic dims） |
| TR-10 | JPEG-LS も OpenVINO で（副） | SG-6 / 手順 17, 18 | `jpegls_openvino` のテストと ffmpeg 相互運用結果 |

---

## 2. 作業内容

### フェーズ 0: 環境の確立と前提の再生成 (見積: 1.5h、うち大半は待ち時間)

1. **pixi 既定環境の確立:**
   * **タスク内容:** `pixi install` を完走させる。`2426b37` で `pixi.toml` に
     `onnxsim` / `onnxscript` / `beartype` / `jaxtyping` / `ruff` / `ty` / `radon` / `pytest-cov` が
     追加されているので、マージ後の lock で再実行する。
   * **目的:** 新しい `tools/export_onnx.py`（dynamo exporter + onnxsim）が動く環境を作る。
   * **対応サブゴール/Trace ID:** SG-0 / TR-9
2. **既存テストの通過確認:**
   * **タスク内容:** `pixi run test` を実行し、`2426b37` 時点のテストがこの機械で全て通ることを確認する。
   * **目的:** 以降の失敗が自分の変更由来か環境由来かを切り分けられる基準線を作る。
   * **対応サブゴール/Trace ID:** SG-0 / TR-4
3. **新契約 ONNX の再エクスポート:**
   * **タスク内容:** CR-40 (17ch) と CR-50 (16ch) のチェックポイントから `--io uint8` で再エクスポートする。
     入力データは私有評価データ（環境変数経由）。
   * **目的:** 配布アーカイブ同梱の ONNX は旧契約なので、配備に使うグラフを作り直す。
   * **対応サブゴール/Trace ID:** SG-0 / TR-9
4. **エクスポート自己検証の確認:**
   * **タスク内容:** `export_onnx.py` が自前で行う検証（平面のバイト一致、JPEG-LS ペイロードのバイト一致、
     4 解像度での再実行）の結果を読み、report JSON を保存する。
   * **目的:** 以降の OpenVINO 側の比較対象となる「正解」を確定させる。
   * **対応サブゴール/Trace ID:** SG-0 / TR-4, TR-9

### フェーズ 1: OpenVINO デバイス制約の確定 (見積: 1.0h)

1. **新グラフの静的化手順の確定:**
   * **タスク内容:** 新契約の動的グラフ（`32*units_h` 形式の symbolic dim）を 800×608 に
     `Model.reshape` して NPU / iGPU / CPU で `compile_model` できることを確認する。
   * **目的:** R-B を潰す。NPU は静的形状を要求するため、ここが通らないと全部が止まる。
   * **対応サブゴール/Trace ID:** SG-1 / TR-1
2. **未確認オペレータの対応確認:**
   * **タスク内容:** `SpaceToDepth`（`pixel_unshuffle` 由来）と `repeat_interleave` の下ろし先、
     `uint8` 入出力、reflect `Pad`、`Gelu(tanh)`、`ConvolutionBackpropData` が
     NPU プラグインで通るかを確認し、通らないものは名指しで記録する。
   * **目的:** R-A を潰す。落ちる場合の回避策（該当部分を CPU 側へ出す等）をここで決める。
   * **対応サブゴール/Trace ID:** SG-1 / TR-1
3. **設計方針の確定と文書化:**
   * **タスク内容:** ランタイムの置き場所、デバイス指定の意味論、コンパイルキャッシュの扱い、
     プロパティ（`INFERENCE_PRECISION_HINT` / `NPU_TURBO` / `CACHE_DIR` / `PERFORMANCE_HINT`）の既定値を決め、本書へ追記する。
   * **目的:** フェーズ 2 の実装を、判断を先送りせずに一直線で書けるようにする。
   * **対応サブゴール/Trace ID:** SG-1, SG-2 / TR-5

### フェーズ 2: OpenVINO ランタイムの実装 (見積: 2.0h)

1. **ランタイム層の実装:**
   * **タスク内容:** `src/compressors/frappe/openvino_runtime.py` に、ONNX パスとデバイス名と
     入力解像度を受け取り、reshape → compile → 推論を行う薄い層を実装する。
     encoder は「画像 uint8 → 平面 uint8 のリスト」、decoder は「平面 → 再構成 uint8」。
   * **目的:** デバイスの差を 1 箇所に閉じ込め、上位のツールがデバイスを意識しないようにする。
   * **対応サブゴール/Trace ID:** SG-2 / TR-1, TR-3
2. **明示失敗の実装:**
   * **タスク内容:** デバイス名が `auto` 以外のとき、そのデバイスで使えなければ例外にする。
     `auto` のときだけ順に試し、選ばれた理由と落ちた理由を戻り値に載せる。
   * **目的:** 暗黙 fallback 禁止（TR-5）を、規約ではなくコードで保証する。
   * **対応サブゴール/Trace ID:** SG-2 / TR-5
3. **単体テストの実装:**
   * **タスク内容:** NPU が無い機械でも意味のあるテストを書く。CPU デバイスでのビット一致、
     デバイス不在時の明示失敗、reshape 前後の形状関係。NPU/GPU を要するものは
     デバイス列挙に基づく skip とする（環境で結果が変わることを隠さない）。
   * **目的:** ビット完全性（TR-4）と明示失敗（TR-5）を回帰から守る。
   * **対応サブゴール/Trace ID:** SG-2 / TR-4, TR-5

### フェーズ 3: ラウンドトリップと計測 (見積: 2.0h)

1. **ラウンドトリップ CLI の実装:**
   * **タスク内容:** `tools/roundtrip_openvino.py` を実装する。画像 → encode → 平面 → JPEG-LS →
     （復号）→ decode → 再構成 PNG。encoder と decoder のデバイスを別々に指定できる。
     PSNR / bpp（両規約）/ CR / 段別レイテンシを JSON に出す。
   * **目的:** 目標1そのもの。「ここで全部やる」を実行可能な 1 コマンドにする。
   * **対応サブゴール/Trace ID:** SG-3 / TR-1, TR-2, TR-3, TR-7
3. **デバイス別ベンチマークの実装:**
   * **タスク内容:** `tools/benchmark_openvino_devices.py` を実装し、encoder × decoder ×
     {CPU, GPU, NPU} を測る。初回コンパイル込みと定常状態を分けて測る
     （参考実装 `CONCEPT_JA.md` §11 が挙げる測定単位に従う）。
   * **目的:** 目標2。どこに何を置くかを数字で決める。
   * **対応サブゴール/Trace ID:** SG-4 / TR-8
4. **62 枚での評価:**
   * **タスク内容:** 私有評価データ全体で PSNR / bpp / CR を測り、既知の CR40 / CR50 の値と
     同一規約で並べる。撮影対象が違うので値そのものは一致しないが、規約が揃っていることを示す。
   * **目的:** 配備した経路が、既知の性能を再現していることを確認する。
   * **対応サブゴール/Trace ID:** SG-5 / TR-7

### フェーズ 4: JPEG-LS の OpenVINO オフロード（副目標） (見積: 3.0h)

1. **参考実装の CPU 参照経路の切り出し:**
   * **タスク内容:** `npu-jpegls-offload` の CPU 参照 MED/Qs 計算と CPU 側符号器を、
     正しさのオラクルとして使える形で取り込む。ライセンスは MIT。
   * **目的:** OpenVINO 版の出力を突き合わせる相手を先に用意する（TDD の Red を作れる状態にする）。
   * **対応サブゴール/Trace ID:** SG-6 / TR-10
2. **OpenVINO グラフの実装:**
   * **タスク内容:** A/B/C/D 抽出（2×3 畳み込み 4 出力）、分岐なし MED、勾配量子化、
     コンテキスト ID `Qs = 81·Q1 + 9·Q2 + Q3` を OpenVINO グラフとして構築し、iGPU / NPU で実行する。
     契約は参考実装と同じ 2 出力（`prediction: int16[R,C]`, `context_id: int16[R,C]`）。
   * **目的:** 目標3。
   * **対応サブゴール/Trace ID:** SG-6 / TR-10
3. **相互運用の検証:**
   * **タスク内容:** 生成した `.jls` を ffmpeg の `jpegls` デコーダで復号し、元画素と一致することを確認する。
   * **目的:** 自前の符号器と自前の復号器が同じ誤りを持つ可能性を排除する。
   * **対応サブゴール/Trace ID:** SG-6 / TR-10

### フェーズ 5: 検証・記録 (見積: 1.0h)

1. **品質ゲート:**
   * **タスク内容:** `pixi run test`、`ruff check`、`ruff format --check` を実行する。
   * **目的:** 追加コードがリポジトリの基準を満たすことを示す。
   * **対応サブゴール/Trace ID:** SG-7
2. **私有データ非混入の最終確認:**
   * **タスク内容:** `git status --porcelain` と、コミット対象の全ファイルに対する私有パス文字列の検索。
   * **目的:** TR-6 を、思い込みではなく検査で保証する。
   * **対応サブゴール/Trace ID:** SG-7 / TR-6
3. **作業記録の完成:**
   * **タスク内容:** 本書 §7 に時刻付きで全フェーズの開始・完了・発見・失敗を記録する。
   * **目的:** 監査エージェントが Trace ID から証跡へ辿れる状態にする。
   * **対応サブゴール/Trace ID:** SG-7

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。*

### フェーズ 0: 環境の確立と前提の再生成

### 手順 1: 私有パスの隔離を確立する
- [x] 🖐 **操作**: `.private/local_paths.env` に `FRAPPE_PRIVATE_RGB_SRC` / `FRAPPE_LOCAL_DATA_ROOT` / `FRAPPE_BUNDLE_ROOT` を書き、`chmod 600` する。以降のコマンドは `set -a; source .private/local_paths.env; set +a` で読み込む。
- [x] 🔎 **確認**: `git check-ignore -v .private/local_paths.env` が `.gitignore:8:.private/` を返す。`git status --porcelain` に私有物が現れない。
- [x] 🧪 **テスト**: `git status --porcelain | grep -c private` が `0` であること。
- [x] 🛠 **エラー時対処**: `.private/` が ignore されていない場合は `.gitignore` を確認する。追跡済みになっていたら `git rm --cached` で外してから再確認する。

### 手順 2: 私有評価データを匿名 800×608 ImageFolder にする
- [x] 🖐 **操作**: 62 枚の私有 PNG を、bicubic で 800×608 に拡大し、メタデータを落とし、`image_%08d.png` の連番で `$FRAPPE_LOCAL_DATA_ROOT/imagefolder/validation/` に書き出す。元名は保持せず、SHA256 との対応表だけ `.private/data/manifest.json` に置く。
- [x] 🔎 **確認**: 62 ファイル。全て `format == "PNG"`, `mode == "RGB"`, `size == (800, 608)`, `image.info` が空。`tools/resize_rgb_dataset.py::_valid_rgb_png` と同じ判定を満たす。
- [x] 🧪 **テスト**: 書き出し後に全ファイルを開き直して上記 4 条件を assert する（スクリプト内で実施済み）。
- [x] 🛠 **エラー時対処**: `image.info` が空でない場合は `Image.new` + `putdata` で画素だけを移し替える。800×600 以外の入力が混ざっていたら、その枚数を記録した上で除外するか個別に扱う。

### 手順 3: pixi 既定環境を完成させる
- [ ] 🖐 **操作**: `PATH="$HOME/.pixi/bin:$PATH" pixi install` を（マージ後の `pixi.lock` で）実行する。
- [ ] 🔎 **確認**: 終了コード 0。`pixi run python -c "import torch, onnx, onnxscript, onnxsim, openvino; print(torch.__version__, openvino.__version__)"` が動く。
- [ ] 🧪 **テスト**: 上記 import が例外なく通ること。初期状態では `ModuleNotFoundError` で失敗するはずで、それが成功へ変わることを確認する。
- [ ] 🛠 **エラー時対処**: ダウンロードが遅い/落ちる場合は再実行する（rattler と uv のキャッシュが効くので進捗は失われない）。CUDA 版 torch が不要という判断に至った場合は、**既定環境を書き換えず** `[feature]` として CPU 版環境を追加する形にし、その理由を本書に記録する。

### 手順 4: 既存テストで基準線を取る
- [ ] 🖐 **操作**: `pixi run test` を実行する。
- [ ] 🔎 **確認**: 全件通過。件数を記録する（`2426b37` 時点で 66 件 + `adapt_to_decoder` の 5 パラメータ = 71 件見込み）。
- [ ] 🧪 **テスト**: `tests/test_prefix_model.py::test_adapt_to_decoder_matches_the_einops_formulation` が 5 通りすべて通ること（新契約の前提そのもの）。
- [ ] 🛠 **エラー時対処**: 失敗したテストがこの機械固有（GPU 前提など）かを切り分ける。環境由来なら理由を記録し、コード由来なら先に直す。

### 手順 5: 新契約 ONNX を CR-50 で再エクスポートする
- [ ] 🖐 **操作**: `pixi run export-onnx --checkpoint "$FRAPPE_BUNDLE_ROOT/model_cr50/final16.pth.tar" --output-stem .private/onnx/frappe_cr50_16ch --io uint8 --dataset-root "$FRAPPE_LOCAL_DATA_ROOT/imagefolder" --split validation --report .private/onnx/cr50_report.json`
- [ ] 🔎 **確認**: 標準出力の入出力署名が `image UINT8 [1, 3, '32*units_h', '32*units_w']` と scale 群ごとの `plane_p*` になっている。自己検証が「平面 0 不一致 / JPEG-LS byte-identical / 4 解像度すべて ok」で終わる。
- [ ] 🧪 **テスト**: ツール自身の検証が gate（`mismatched_symbols != 0` または payload 不一致なら `SystemExit`）。ここを通ることが合格条件。
- [ ] 🛠 **エラー時対処**: dynamo exporter が落ちる場合は `onnxscript` の導入を確認する。`--no-simplify` で onnxsim を切って切り分ける。解像度再実行で落ちる場合は symbolic dim の関係式が壊れているので、`describe()` の出力を本書に貼って原因を特定する。

### 手順 6: 新契約 ONNX を CR-40 でも再エクスポートする
- [ ] 🖐 **操作**: 手順 5 と同じコマンドを `model_cr40/final17.pth.tar` / `frappe_cr40_17ch` で実行する。
- [ ] 🔎 **確認**: 手順 5 と同じ合格条件。scale 群の構成が 17ch 側（`ps=[32,16×5,8×3,4×6,2×2]`）になっている。
- [ ] 🧪 **テスト**: 同上。
- [ ] 🛠 **エラー時対処**: 同上。片方だけ失敗する場合はチェックポイントの `config.ps` を突き合わせる。

### フェーズ 1: OpenVINO デバイス制約の確定

### 手順 7: 新グラフを 3 デバイスでコンパイルできるか確かめる
- [ ] 🖐 **操作**: 再エクスポートした encoder / decoder を `openvino.Core().read_model` で読み、800×608 相当に `reshape` して `compile_model` を CPU / GPU / NPU で試すプローブを書いて実行する。
- [ ] 🔎 **確認**: 各デバイスについて「成功 / 失敗（例外型とメッセージ）」が表になって出る。失敗は握り潰さず記録する。
- [ ] 🧪 **テスト**: 成功したデバイスで 1 回推論し、出力形状が期待どおり（`plane_p32` が `(1, n_s*19, 25)` 等）であること。
- [ ] 🛠 **エラー時対処**: NPU が動的形状で拒否する場合は reshape 後に再試行する。特定オペレータで落ちる場合はメッセージからオペレータ名を特定し、手順 8 の対象に加える。

### 手順 8: 落ちたオペレータの回避策を決める
- [ ] 🖐 **操作**: 手順 7 で NPU / GPU が拒否したオペレータがあれば、それだけを含む最小グラフを作って再現させ、回避策（グラフ分割・`--io float` 形・当該部分の CPU 実行）を比較する。
- [ ] 🔎 **確認**: 採用する回避策と、採らなかった選択肢の理由が本書に書かれている。
- [ ] 🧪 **テスト**: 回避策適用後のグラフが対象デバイスで compile でき、出力が回避前と一致すること。
- [ ] 🛠 **エラー時対処**: 回避策が無い場合は「そのデバイスではこのグラフは動かない」と明記する。動くふりをしない。

### 手順 9: デバイスプロパティの既定値を決める
- [ ] 🖐 **操作**: `core.get_property(device, "SUPPORTED_PROPERTIES")` で実在を確認した上で、`CACHE_DIR` / `INFERENCE_PRECISION_HINT` / `PERFORMANCE_HINT` / `NPU_TURBO` の既定値を決める。
- [ ] 🔎 **確認**: 存在しないプロパティを設定していない。既定値の選択理由が本書にある。
- [ ] 🧪 **テスト**: 既定値ありとなしで 1 回ずつ推論し、レイテンシ差を記録する。
- [ ] 🛠 **エラー時対処**: プロパティ設定で例外が出る場合はそのデバイスの `SUPPORTED_PROPERTIES` を出力して、実在するものだけに絞る。

### フェーズ 2: OpenVINO ランタイムの実装

### 手順 10: ランタイムのテストを先に書く（Red）
- [ ] 🖐 **操作**: `tests/test_openvino_runtime.py` に、(a) CPU デバイスで encoder の平面が PyTorch 参照とバイト一致する、(b) 存在しないデバイスを明示指定すると例外になる、(c) `auto` は使えるデバイスを選び理由を返す、の 3 テストを書く。
- [ ] 🔎 **確認**: 3 件とも `ModuleNotFoundError` または `AttributeError` で失敗する（実装が無いため）。
- [ ] 🧪 **テスト**: `pixi run pytest tests/test_openvino_runtime.py -q` が 3 failed になること。
- [ ] 🛠 **エラー時対処**: 失敗理由が意図と違う（例: fixture のパス誤り）場合はテスト側を直してから進む。

### 手順 11: ランタイムを実装する（Green）
- [ ] 🖐 **操作**: `src/compressors/frappe/openvino_runtime.py` を実装する。ONNX パス・デバイス・解像度を受け取り、reshape → compile → 推論。encoder と decoder を別クラスにし、コンパイル済みモデルをキャッシュする。
- [ ] 🔎 **確認**: 手順 10 の 3 件が通る。
- [ ] 🧪 **テスト**: `pixi run pytest tests/test_openvino_runtime.py -q` が 3 passed へ変わること。
- [ ] 🛠 **エラー時対処**: バイト不一致が出た場合、まず入力の正規化（uint8 のまま渡しているか）を疑う。新契約では正規化はグラフの中なので、外で `/127.5-1` をかけていたら誤り。

### 手順 12: NPU / iGPU でのビット一致を確認する
- [ ] 🖐 **操作**: 同じテストを NPU と iGPU に対して実行する（デバイス列挙で存在する場合のみ）。
- [ ] 🔎 **確認**: 3 デバイスすべてで符号平面が PyTorch 参照とバイト一致する。一致しないデバイスがあれば、不一致シンボル数と最大差を記録する。
- [ ] 🧪 **テスト**: `plane_mismatched_symbols == 0` を assert する。fp16 実行で一致しない場合は、それが `INFERENCE_PRECISION_HINT` に起因することを実験で示す。
- [ ] 🛠 **エラー時対処**: 不一致が出たら `INFERENCE_PRECISION_HINT=f32` で再試行する。それでも不一致なら、そのデバイスは符号器に使えないと明記し、デコーダ専用にする。

### フェーズ 3: ラウンドトリップと計測

### 手順 13: ラウンドトリップのテストを先に書く（Red）
- [ ] 🖐 **操作**: `tests/test_roundtrip_openvino.py` に、合成画像 1 枚で encode → JPEG-LS → decode が通り、JPEG-LS 往復後の平面が符号器出力と一致することを検証するテストを書く。
- [ ] 🔎 **確認**: 実装が無いため失敗する。
- [ ] 🧪 **テスト**: `pixi run pytest tests/test_roundtrip_openvino.py -q` が failed。
- [ ] 🛠 **エラー時対処**: 合成画像のサイズは 32 の倍数かつ `units >= 2`（すなわち最小 64×64）にする。これを外すと reflect padding の制約に当たる。

### 手順 14: ラウンドトリップ CLI を実装する（Green）
- [ ] 🖐 **操作**: `tools/roundtrip_openvino.py` を実装する。`--encoder-device` / `--decoder-device` / `--checkpoint-onnx-stem` / `--image-index` / `--output` / `--report`。JPEG-LS は `entropy_coding.encode_latents` / `decode_latents` をそのまま使う。
- [ ] 🔎 **確認**: 手順 13 のテストが通る。私有データ 1 枚で実行して再構成 PNG が出る。
- [ ] 🧪 **テスト**: JSON に `plane_mismatched_symbols == 0`、`psnr_db`、`bpp_payload_only`、`bpp_with_length_prefix`、段別レイテンシが載ること。
- [ ] 🛠 **エラー時対処**: `decode_latents` が返す平面の形が合わない場合、新契約では平面が `(C*H, W)` である点と、デコーダグラフが `unflatten` を内蔵している点を突き合わせる。

### 手順 15: 6 通りのデバイス組み合わせを測る
- [ ] 🖐 **操作**: `tools/benchmark_openvino_devices.py` を実装・実行する。encoder × decoder × {CPU, GPU, NPU}、初回コンパイル込みと定常状態を分離、JPEG-LS の CPU 時間も別途計上する。
- [ ] 🔎 **確認**: 表が JSON と標準出力に出る。どの組み合わせが最速かが読み取れる。
- [ ] 🧪 **テスト**: 各測定が最低 10 回の中央値であること。ウォームアップを計測から除外していること。
- [ ] 🛠 **エラー時対処**: 測定がばらつく場合は電源モードと他プロセスの負荷を記録し、繰り返し数を増やす。

### 手順 16: 62 枚全体で評価する
- [ ] 🖐 **操作**: 最速構成と「NPU で符号化する構成」の両方で、62 枚の PSNR / bpp / CR を測る。
- [ ] 🔎 **確認**: 両構成で符号がビット一致し、したがって bpp と PSNR が完全に同一であること（デバイスは速度だけを変え、結果を変えない）。
- [ ] 🧪 **テスト**: 2 つの JSON の `psnr_db` と `bpp_payload_only` が厳密一致すること。
- [ ] 🛠 **エラー時対処**: 一致しない場合は手順 12 に戻る。デバイスによって結果が変わるのは、この作業では欠陥として扱う。

### フェーズ 4: JPEG-LS の OpenVINO オフロード（副目標）

### 手順 17: CPU 参照 MED/Qs をオラクルとして用意する
- [ ] 🖐 **操作**: 参考実装の CPU 参照計算（境界付き行バッファ、MED、勾配量子化、`Qs`）を、本リポジトリのテストから呼べる形にする。MIT ライセンスの帰属を残す。
- [ ] 🔎 **確認**: FRAPPE の実際の潜在平面（例: CR-50 の `plane_p2` が 304×400）に対して MED/Qs が計算できる。
- [ ] 🧪 **テスト**: 参考実装の既存テストのうち、NPU を要さないものが通ること。
- [ ] 🛠 **エラー時対処**: 参考実装が現行 Python で動かない場合は、必要な関数だけを移植し、移植元をコメントで明記する。
- [ ] 🛠 **判断ポイント**: この時点で「CharLS より速くなる見込みが無い」と判明した場合は、実装ではなく**測定結果と判断を記録して終える**。副目標を目的化しない。

### 手順 18: OpenVINO で MED/Qs を計算する
- [ ] 🖐 **操作**: A/B/C/D 抽出 → 分岐なし MED → 勾配量子化 → `Qs` を OpenVINO グラフとして構築し、iGPU / NPU で実行する。
- [ ] 🔎 **確認**: 出力が CPU 参照と**完全一致**（整数値なので厳密一致を要求する）。
- [ ] 🧪 **テスト**: 全タイル比較で不一致 0。不一致が 1 つでもあれば符号化を中止する（参考実装と同じ規約）。
- [ ] 🛠 **エラー時対処**: FP16 で表現範囲を外れる場合は、参考実装 `CONCEPT_JA.md` §7 の範囲議論を確認し、必要なら `INFERENCE_PRECISION_HINT=f32` を使う。

### 手順 19: ffmpeg で外部相互運用を確認する
- [ ] 🖐 **操作**: 生成した `.jls` を `ffmpeg -i out.jls -f rawvideo -` で復号し、元の平面と一致することを確認する。
- [ ] 🔎 **確認**: 全画素一致。
- [ ] 🧪 **テスト**: 自前復号器と ffmpeg の双方で一致すること（自作同士が同じ誤りを持つ可能性を排除する）。
- [ ] 🛠 **エラー時対処**: ffmpeg が読めない場合はマーカー構造を `npu-jpegls inspect` 相当で確認する。

### フェーズ 5: 検証・記録

### 手順 20: 品質ゲートを通す
- [ ] 🖐 **操作**: `pixi run test`、`pixi run ruff check .`、`pixi run ruff format --check .` を実行する。
- [ ] 🔎 **確認**: すべて成功。追加したテストの件数を記録する。
- [ ] 🧪 **テスト**: 追加テストを含む全件が通ること。
- [ ] 🛠 **エラー時対処**: lint の失敗は自動修正可能なものと設計上の指摘を切り分けてから対処する。

### 手順 21: 私有データ非混入を検査する
- [ ] 🖐 **操作**: `git status --porcelain` と、`git diff --cached` および追加した全ファイルに対する私有パス文字列（`2026_TW_TVA` など）の検索を行う。
- [ ] 🔎 **確認**: 検索結果 0 件。`.private/` 配下が一切 staging されていない。
- [ ] 🧪 **テスト**: `git grep -I -n '2026_TW_TVA' $(git ls-files)` が何も返さないこと。
- [ ] 🛠 **エラー時対処**: 1 件でも出たら該当箇所を環境変数参照へ書き換え、コミット前に必ず再検査する。既にコミットしていたら履歴の扱いをユーザーに確認する。

### 手順 22: 作業記録を完成させる
- [ ] 🖐 **操作**: 本書 §7 に、各フェーズの開始・完了時刻、実行コマンド、結果、想定外の発見を記入する。
- [ ] 🔎 **確認**: §1.3 の全 Trace ID に対応する証跡が存在する。
- [ ] 🧪 **テスト**: Trace ID を 1 つ選び、そこから証跡ファイルへ実際に辿れること。
- [ ] 🛠 **エラー時対処**: 証跡が無い Trace ID があれば、未達として明示する。達成したことにしない。

---

## 4. 作業に使用するコマンド参考情報

### 環境

```bash
# 私有パスの読み込み（毎回、リポジトリルートで）
set -a; source .private/local_paths.env; set +a

# pixi（グローバル導入済み: ~/.pixi/bin/pixi）
export PATH="$HOME/.pixi/bin:$PATH"
pixi install
pixi task list
```

### テストと品質管理

```bash
pixi run test
pixi run pytest tests/test_openvino_runtime.py -q
pixi run ruff check .
pixi run ruff format --check .
```

### ONNX の再エクスポート

```bash
pixi run export-onnx \
  --checkpoint "$FRAPPE_BUNDLE_ROOT/model_cr50/final16.pth.tar" \
  --output-stem .private/onnx/frappe_cr50_16ch \
  --io uint8 \
  --dataset-root "$FRAPPE_LOCAL_DATA_ROOT/imagefolder" \
  --split validation \
  --report .private/onnx/cr50_report.json
```

### OpenVINO のデバイス確認

```bash
python3 -c "
import openvino as ov
core = ov.Core()
print(core.available_devices)
for d in core.available_devices:
    print(d, core.get_property(d, 'FULL_DEVICE_NAME'))
"
```

### ラウンドトリップとベンチマーク

```bash
pixi run python tools/roundtrip_openvino.py \
  --onnx-stem .private/onnx/frappe_cr50_16ch \
  --dataset-root "$FRAPPE_LOCAL_DATA_ROOT/imagefolder" --split validation --image-index 0 \
  --encoder-device NPU --decoder-device GPU \
  --output .private/out/cr50_recon.png \
  --report .private/out/cr50_roundtrip.json

pixi run python tools/benchmark_openvino_devices.py \
  --onnx-stem .private/onnx/frappe_cr50_16ch \
  --dataset-root "$FRAPPE_LOCAL_DATA_ROOT/imagefolder" --split validation \
  --devices CPU GPU NPU --repeats 10 \
  --report .private/out/benchmark_devices.json
```

### JPEG-LS の外部検証

```bash
ffmpeg -hide_banner -i plane.jls -f rawvideo -pix_fmt gray - | cmp - plane.raw
```

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします*

- [ ] **DoD-1 (目標1 / TR-1,2,3)**: CR-40 と CR-50 の両方で、私有評価画像 1 枚に対し
      `encode → JPEG-LS → decode → 再構成 PNG` が 1 コマンドで完走し、再構成 PNG と
      レポート JSON が生成されている。encoder を NPU に置いた構成でも完走する。
- [ ] **DoD-2 (TR-4)**: OpenVINO の CPU / iGPU / NPU が出す符号平面が、PyTorch 参照と
      **バイト一致**する。一致しないデバイスがある場合は、不一致数・最大差・原因・
      そのデバイスを符号器に使わない判断が本書に記録されている。
- [ ] **DoD-3 (TR-8)**: encoder × decoder × {CPU, GPU, NPU} のレイテンシが、初回コンパイル込みと
      定常状態に分けて測定され、JSON に残っている。どの構成を採るかの判断が数字に紐づいている。
- [ ] **DoD-4 (TR-7)**: 62 枚での PSNR / bpp / CR が、長さ接頭辞なし・込みの**両規約**で
      記録され、既知の CR40 (29.78 dB / CR 40.47) / CR50 (28.63 dB / CR 50.53) と
      同一規約で並べられる形になっている。
- [ ] **DoD-5 (TR-5)**: `--device NPU` を明示して NPU が使えない状況を作ったとき、CPU に
      落ちずに例外で止まることがテストで示されている。
- [ ] **DoD-6 (TR-6)**: `git status --porcelain` に私有物が現れず、追跡対象の全ファイルに
      私有パス文字列が含まれないことが検査で確認されている。
- [ ] **DoD-7 (TR-9)**: 配備に使う ONNX が `2426b37` の新契約（uint8 I/O、symbolic dims、
      平面出力）で再エクスポートされ、その自己検証が通っている。旧 ONNX は配備に使っていない。
- [ ] **DoD-8 (TR-10 / 副)**: JPEG-LS の OpenVINO オフロードについて、実装できたなら
      CPU 参照との完全一致と ffmpeg 相互運用が示されている。実装しない判断をしたなら、
      その根拠（CharLS との速度見積もり）が数字で記録されている。
- [ ] **DoD-9**: `pixi run test` と lint が通り、追加したテストが回帰を守る形になっている。
- [ ] **DoD-10**: §1.3 の全 Trace ID に証跡が対応している。未達の項目は未達と明記されている。

---

## 7. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-02 | 08:35 JST | Claude | ブランチ整備: 他ブランチのマージ確認と `npu` 作成 | `origin/joint-prefix-training` は `origin/main` と同一コミット `c881650` でマージ対象なし。`npu` を `c881650` から作成 |
| 2026-09-02 | 08:40 JST | Claude | 元論文 (FRAPPE, 8 ページ) 精読 | 解析路は 5 個の `Conv2d` + companding に collapse。合成路がパラメータ/FLOPs のほぼ全部。非対称配備が設計思想 |
| 2026-09-02 | 08:41 JST | Claude | pixi をグローバル導入 | `~/.pixi/bin` に pixi 0.78.0。`~/.zshrc` を更新 |
| 2026-09-02 | 08:41 JST | Claude | 配布アーカイブ展開と検証 | 189 MB / 45 エントリ。SHA256 全 8 件 OK。会話 JSONL 463 レコード、欠損なし |
| 2026-09-02 | 08:45 JST | Claude | ハードウェアとランタイム確認 | NPU (8086:7d1d) + Arc iGPU + OpenVINO 2025.1。`available_devices == ['CPU','GPU','NPU']`。追加ドライバ不要 |
| 2026-09-02 | 08:46 JST | Claude | **旧 ONNX の 3 デバイス実測** | **重要発見**: encoder は CPU 1.70 / GPU 16.07 / NPU 17.70 ms、decoder は CPU 213.11 / GPU 45.35 / NPU 178.21 ms。**得をするのは decoder の iGPU 化であって encoder の NPU 化ではない** |
| 2026-09-02 | 08:46 JST | Claude | 要求整理メモ作成 | `temp/memo_Sep02-2026_npu_offload_requirements.md` |
| 2026-09-02 | 08:52 JST | Claude | main の 3 コミットを取り込み | **重要発見**: ONNX 契約が変更。ビットストリーム整形がグラフ内へ、I/O が uint8、形状が真に動的。**配布アーカイブの ONNX は旧契約となり再エクスポートが必要** |
| 2026-09-02 | 08:54 JST | Claude | 私有評価データの匿名化 (手順 1, 2 完了) | 62 枚を 800×608 / RGB / メタデータなし / `image_%08d.png` に変換。`git status` に現れないことを確認 |
| 2026-09-02 | 08:58 JST | Claude | pixi install の帯域測定 | torch cu128 は 820 MB、実効 ~960 KB/s。nvidia_* ホイール 15 個。この機械に NVIDIA GPU は無く CUDA 分は死荷重。完了まで 1 時間規模の見込み |
| 2026-09-02 | 09:00 JST | Claude | 本作業計画書を作成 | フェーズ 0〜5、手順 1〜22、DoD 10 項目、Trace ID 10 本を定義 |
| 2026-09-02 | 09:02 JST | Claude | 並列調査 4 本を起動（リポジトリ規約 / ONNX 契約 / OpenVINO 制約 / JPEG-LS 移植） | 3 本完了。**指摘 1**: グラフが既に +127 シフト済みなので `encode_latents` に渡すと二重シフトになり、**もっともらしい誤った画像に復号される**。**指摘 2**: `plane_p{ps}` はパッチサイズが再出現するスケジュールで重複しうるので、ポートは名前でなくインデックスで束縛すべき |
| 2026-09-02 | 09:05 JST | Claude | `src/compressors/frappe/openvino_runtime.py` 実装 (手順 10, 11) | TDD: `tests/test_openvino_runtime.py` を先に書き 12 failed → 12 passed。メタデバイス (AUTO/HETERO/MULTI/BATCH) を拒否、`select_device` が却下理由を記録、ポートはインデックス束縛 |
| 2026-09-02 | 09:08 JST | Claude | `src/compressors/frappe/bitstream.py` 実装 | worklog `2426b37` が求めた「ビットストリーム規約を 1 モジュールに集約し接頭辞を明示引数に」を実装。torch 非依存。二重シフト防止のため uint8 以外の平面を拒否。`tests/test_bitstream.py` に `entropy_coding.encode_latents` とのバイト一致検査を用意（torch 待ち） |
| 2026-09-02 | 09:12 JST | Claude | `tools/roundtrip_openvino.py` / `tools/benchmark_openvino_devices.py` 実装 | encoder/decoder のデバイスを個別指定。レートは payload-only を主とし接頭辞込みを併記。ベンチは encoder と decoder が独立なので 3×3 ではなく 3+3 を測る |
| 2026-09-02 | 09:15 JST | Claude | **旧グラフを使った実機配線検証** | **重要**: CPU / iGPU / NPU の 3 デバイスすべてで reshape → compile → 推論が成功。`execution_devices` が `['CPU'] / ['GPU.0'] / NPU` を返し、実際にそのデバイスで走ったことを確認。存在しないデバイスは例外で停止（暗黙 fallback なし） |
| 2026-09-02 | 09:15 JST | Claude | 平面幾何の独立検証 | **重要**: `plane_shapes_for(CR50, 608, 800)` = `[(19,25),(190,50),(228,100),(912,200),(304,400)]` が、旧グラフの実際の符号形状 `(1,19,25),(5,38,50),(3,76,100),(6,152,200),(1,304,400)` の `rows = C × h` 分解と厳密に一致（5×38=190, 3×76=228, 6×152=912）。再エクスポート前に幾何の導出が正しいことが確定 |
| | | | | |
