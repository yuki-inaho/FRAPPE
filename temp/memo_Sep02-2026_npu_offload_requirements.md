# メモ: NPU オフロード作業の目的・要求・要件整理

作成: 2026-09-02 08:46 JST / ブランチ `npu` / リポジトリ `/home/inaho/Project/FRAPPE`

このメモは作業計画書 `workdoc_Sep02-2026_npu_openvino_roundtrip.md` の前段。
「何を作るのか」の合意を取るための要約であり、手順は書かない。

---

## 1. ユーザーの最終目的（直観的な言葉のまま）

> NUC14 Pro の NPU を使って、エンコーディング + JPEG-LS 圧縮 → 復元まで、ここでやる。

つまり **FRAPPE コーデックのラウンドトリップ一式を、この端末のアクセラレータ上で完結させる**。
学習は完了済みで、成果物（重み・ONNX）は手元にある。ここから先は「配備」の作業。

---

## 2. 明示要求の分解

| ID | 要求 | 状態 |
| :--- | :--- | :--- |
| R-1 | 他ブランチの更新を main にマージし、そこから `npu` ブランチを作る | 完了（マージ対象なし・`npu` 作成済み） |
| R-2 | 元論文を読む | 完了（IEEE 形式 8 ページ、後述の要点） |
| R-3 | pixi をグローバルインストールし、セットアップする | pixi 0.78.0 導入済み / `pixi install` 実行中 |
| R-4 | 配布アーカイブを展開し、チェックポイントと軽量化会話 JSONL を確認する | 完了（SHA256 全 8 件 OK） |
| R-5 | **NPU 上でエンコード + JPEG-LS 圧縮 → 復元まで動かす** | 本作業の主目標 |
| R-6 | JPEG-LS も OpenVINO 経由で iGPU／できれば NPU で符号化する（必須ではない） | 本作業の副目標 |
| R-7 | 実データは私有 RGB データを使う。**データ名・ファイルを staging しない** | 全工程を通じた制約 |
| R-8 | 各種確認 → 目的・要求・要件整理 → メモ → 作業計画 + DoD → 実施 | 本メモと作業計画書がこれ |

---

## 3. 暗黙制約（明示されていないが外せないもの）

1. **私有データの非混入**: パス・ファイル名・画素のいずれも git 追跡下に置かない。
   本リポジトリには既に同じ規約がある（`MANAGED_TRAINING.md`: 「ソースアーカイブ名をリポジトリ設定や
   コマンドログに置かない」）。私有パスは `.private/local_paths.env`（gitignore 済み）に隔離し、
   文書とコードは環境変数 `$FRAPPE_PRIVATE_RGB_SRC` 経由でのみ参照する。
2. **暗黙 fallback の禁止**: `--device NPU` を要求して NPU が使えないとき、黙って CPU に落ちない。
   参考実装 `npu-jpegls-offload` の設計思想（`backend=npu` は失敗、`auto` のみフォールバックし理由を記録）
   をそのまま踏襲する。
3. **ビット完全性の維持**: 符号器出力 int8 は、PyTorch 参照とビット単位で一致すること。
   既存 `tools/export_onnx.py` が「符号の不一致が 1 シンボルでもあれば異常終了」という規約を持っており、
   デバイスが変わってもこの規約を下げない。
4. **パッケージ管理は pixi**（uv ではない）。テンプレートの `uv run` は `pixi run` に読み替える。
5. **監査可能性**: 各主張に対して、再実行可能なコマンドと機械可読な成果物（JSON）を残す。

---

## 4. 確認済みの事実（すべて本セッションで実測）

### 4.1 ハードウェアとランタイム

| 項目 | 値 |
| :--- | :--- |
| CPU | Intel Core Ultra 5 125H（14 コア / 2 スレッド per core, Meteor Lake） |
| NPU | Intel AI Boost — PCI `8086:7d1d`, `intel_vpu` ドライバ, `/dev/accel/accel0` |
| iGPU | Intel Arc Graphics (iGPU) — `/dev/dri/renderD128` |
| NPU アクセス権 | ユーザー `inaho` は `render` グループ所属済み。追加設定不要 |
| OpenVINO | Python 側 **2025.1.0** / apt 側 2024.4.0。NPU・GPU・CPU プラグイン全て導入済み |
| ランタイム | `intel-level-zero-npu 1.10.0`, `intel-opencl-icd 24.39`, `level-zero 1.17.44` |
| ffmpeg | 7.x、`jpegls` デコーダあり（JPEG-LS の外部相互運用検証に使える） |

`openvino.Core().available_devices` → `['CPU', 'GPU', 'NPU']`。**追加のドライバ導入は不要**。

### 4.2 決定的な実測: 既存 ONNX は 3 デバイスすべてで無改造で動く

> **2026-09-02 08:55 追記**: この計測は `c881650` 時点の**旧**エクスポータが出した ONNX に対するもの。
> 直後に main へ 3 コミット（`5af15f9` / `773b38c` / `2426b37`）が入り、ONNX の契約が変わった（§4.5）。
> 数字はデバイスの相対関係を示す参考値として残すが、**配備に使うグラフは再エクスポートが必要**。

配布アーカイブの CR-50 ONNX を静的形状 800×608 に reshape してコンパイル・推論した結果:

| グラフ | CPU | iGPU | NPU |
| :--- | ---: | ---: | ---: |
| encoder (16ch, 0.04 MB) | **1.70 ms** | 16.07 ms | 17.70 ms |
| decoder (16ch, 13.5 MB) | 213.11 ms | **45.35 ms** | 178.21 ms |

コンパイルは encoder が CPU 0.7 / GPU 7.6 / NPU 2.1 秒、decoder が 0.2 / 0.8 / 7.2 秒。全て成功。

**ここから読み取れること（計画を左右する）:**

- **オフロードで得をするのは decoder であって encoder ではない。** decoder は iGPU で
  CPU 比 **4.7 倍速**。一方 encoder は CPU が最速で、NPU/iGPU に出すと 10 倍遅くなる。
  encoder は畳み込み 1 段 + 要素ごと演算だけ（0.04 MB）なので、転送と起動の固定費が計算を上回る。
- したがって encoder を NPU に載せる価値は**レイテンシではなく、CPU 解放と電力**にある。
  これは参考資料 `npu-jpegls-offload/docs/CONCEPT_JA.md` §11 の指摘と一致する。
- 上記は素の fp32・キャッシュ無し・同期実行の初期値。NPU は fp16 化・`NPU_TURBO`・
  コンパイルキャッシュで改善余地があるため、**最適化前の下限値**として扱う。

### 4.3 手元の成果物（`$FRAPPE_BUNDLE_ROOT`、SHA256 全件検証済み）

| 動作点 | ch | validation | test (held-out) | ONNX 符号一致 |
| :--- | ---: | :--- | :--- | :--- |
| CR 40 | 17 | 29.78 dB / CR 40.47 / 違反 0 | 30.06 dB / CR 39.31 | 1,833,500 シンボル全一致 |
| CR 50 | 16 | 28.63 dB / CR 50.53 / 違反 0 | 28.89 dB / CR 48.66 | 1,347,100 シンボル全一致 |

会話 JSONL は `agent-jsonl-compact` で 4.68 MB → 1.27 MB に軽量化されたもの。
463 レコード（user 32 / assistant 74 / tool 178 組）、2026-09-01 15:47Z〜19:29Z、欠損なし。

### 4.4 実データ

`$FRAPPE_PRIVATE_RGB_SRC`: PNG 62 枚、RGB、**800×600**、計 64 MB。

800×600 は最大パッチサイズ 32 で割り切れない（600 / 32 = 18.75）ため、そのままでは通らない。
**800×608 へバイキュービック拡大**して使う。これは学習時とまったく同じ前処理で
（`MANAGED_TRAINING.md` の `resize-data --width 800 --height 608`、元データも 800×600）、
訓練分布と幾何が一致する。切り出し（800×576）は再標本化を避けられるが、
訓練時と幾何が変わるので採らない。

**実施済み (08:54)**: 62 枚を匿名連番 `image_%08d.png`・メタデータなし・RGB・800×608 へ変換し、
`$FRAPPE_LOCAL_DATA_ROOT/imagefolder/validation/` に配置した。`tools/prune_latent_channels.load_images`
が期待する `<root>/<split>/image_????????.png` の形。元ファイル名は保持せず、
SHA256 と匿名連番の対応表だけを `.private/data/manifest.json`（gitignore 済み）に置いた。
`git status` に現れないことを確認済み。学習はここで一切行わないので、62 枚すべてが未見データであり、
`validation` は単なるディレクトリ名。

### 4.5 作業中に main 側で ONNX 契約が変わった（`c881650` → `2426b37`）

3 コミットを `main` と `npu` に取り込んだ。**本作業の前提を直接変える変更**なので要点を記録する。

| コミット | 内容 |
| :--- | :--- |
| `5af15f9` | ビットストリーム整形を ONNX encoder グラフの中へ移動 |
| `773b38c` | 上記の作業記録 |
| `2426b37` | 長さ接頭辞に起因する 2 系統の bpp の食い違いを定量化 |

**新しいグラフ契約**（`tools/export_onnx.py`, `JOINT_PREFIX_TRAINING.md` の ONNX 節）:

| グラフ | 入力 | 出力 |
| :--- | :--- | :--- |
| `*_encoder.onnx` | `image (1, 3, 32h, 32w) uint8` | scale 群ごとに `uint8` 平面 `(1, n_s·32h/p_s, 32w/p_s)` |
| `*_decoder.onnx` | 同じ平面 | `reconstruction (1, 3, 32h, 32w) uint8` |

- 分割点が**エントロピー符号器の直前**に移った。encoder の出力は JPEG-LS が読むグレースケール画像
  そのもの。グラフの外に残るのは JPEG-LS 本体と 4 バイト長さ接頭辞だけ。
  → **配備側に算術が残らない**。OpenVINO ランタイムはテンソルを JPEG-LS に渡すだけでよくなり、
  この作業にとっては明確に楽になる。
- I/O が両側 uint8 に統一され、`/127.5 - 1` の正規化がグラフ内へ。`--io float` で従来形式も出る。
- **形状が真に動的**になった。旧エクスポートは `dynamic_axes` を宣言していたが einops が
  traced 形状を焼き込んでおり、800×608 以外では実行時エラーだった。
  `ops.adapt_to_decoder` が `F.pixel_unshuffle` / `repeat_interleave` に置換され（bit-identical、
  `test_adapt_to_decoder_matches_the_einops_formulation` が全 patch size で検証）、
  dynamo exporter + `torch.export.Dim` で `units_h/units_w` の関係式がグラフに残る。
- opset 18、onnx-simplifier 適用（encoder 91 ノード / decoder 192〜195 ノード）。

**この作業への影響:**

1. `$FRAPPE_BUNDLE_ROOT/onnx_cr40|cr50/` の ONNX は**旧契約**。チェックポイントから再エクスポートする。
2. NPU は静的形状を要求するので、動的グラフを 800×608 に reshape してからコンパイルする
   （旧グラフでの実測でこの手順は確認済み）。動的性は「1 つのチェックポイントから
   任意解像度の静的グラフを作れる」という形で効く。
3. uint8 I/O は NPU 側にとって有利（fp32 の host 転送量が 1/4、正規化の取り違えが起きない）。
4. `pixel_unshuffle` → `SpaceToDepth`、`repeat_interleave` → `Tile` 系に落ちるはずだが、
   NPU プラグインの対応は**実測で確かめる**（旧グラフは einops 由来の Reshape だったので、
   ここは新規リスク）。

**レート規約の食い違い（`2426b37`）**: `encode_latents` は scale 群ごとに 4 バイトの長さ接頭辞を
付けるが、`evaluate.py` / `tools/evaluate_joint_prefix.py` / notebook は生ペイロードのみ合計する。
差は `32·G/(T1·T2)` = 800×608・5 群で **3.29e-4 bpp**（レートの 0.07% 未満、CR で 0.035 ポイント）。
**本作業の bpp は接頭辞なし側（`evaluate_joint_prefix` 系）で統一する** — 既存の
CR 40/50 の数字がすべてそちらで、比較対象がそこにあるため。JSON には
`bpp_payload_only` と `bpp_with_length_prefix` の両方を出し、どちらの規約かを明示する。

---

## 5. 元論文（FRAPPE, Jacobellis & Yadwadkar, UT Austin）の要点

本作業に効く部分だけ:

- 解析変換は **スケール別の重なりなし線形射影のみ**。5 個の `Conv2d` + companding + 量子化に collapse する。
  再帰も量子化器チェーンもない DAG なので、**アクセラレータに載せる分には自明に載る**（実測でも確認済み）。
- 合成変換がパラメータと FLOPs のほぼ全部を持つ（ConvNeXt 系 12 ブロック、幅 768 が論文構成）。
  **非対称配備**（軽い符号化をセンサ側、重い復号をクラウド側）が設計思想。
  → 今回の「NUC 一台で両方」は論文の想定と逆で、**decoder 側が律速になる**のは構造上当然。
- エントロピー符号化は「4 関数契約」の外側に隔離されており、JPEG-LS は差し替え可能な一実装。
  各スケールの int8 潜在を `(C*H, W)` の 1 枚のグレースケール画像に整形して JPEG-LS にかける規約。
- 量子化前の活性は概ね Laplacian に従うので、Golomb-Rice 予測残差を持つ JPEG-LS がほぼ最適。
- 論文の CPU 符号化スループットは 74〜168 MPx/s（AVIF 比 47 倍速）。本作業の実測
  CPU 1.70 ms @ 800×608 = 286 MPx/s はこの桁と整合する。

---

## 6. これから作るもの（成果物の形）

0. **前提の再生成**: `pixi run export-onnx` を CR-40 / CR-50 のチェックポイントに対して走らせ、
   新契約の ONNX を作り直す。ここで `--io uint8` を既定にする（配備形）。
   旧 ONNX（配布アーカイブ同梱）は来歴として残すが配備には使わない。

1. `src/compressors/frappe/openvino_runtime.py`
   OpenVINO デバイス（CPU/GPU/NPU）で encoder / decoder を実行する薄い層。
   デバイス指定は明示のみ、暗黙 fallback なし。静的形状を要求し、コンパイルをキャッシュする。
2. `tools/roundtrip_openvino.py`
   画像 → encode(device) → int8 符号 → JPEG-LS → decode(device) → 再構成 の一気通貫 CLI。
   bpp / CR / PSNR / デバイス別レイテンシを JSON で出す。
3. `tools/benchmark_devices.py`
   CPU / iGPU / NPU × encoder / decoder のレイテンシ表を出す。
4. （副目標）JPEG-LS の MED 予測 + コンテキスト ID を OpenVINO グラフとして iGPU/NPU で計算し、
   CPU の Golomb 符号化へ渡す経路。参考実装の `IntelNPUTileGraph` を OpenVINO へ置換する形。
   参考実装が定めた契約は `prediction: int16[R,C]` と `context_id: int16[R,C]` の 2 出力だけなので、
   前処理バックエンドの差し替えだけで済む。
5. テスト一式（ビット一致・デバイス指定の失敗が失敗として現れること・ラウンドトリップ）。

---

## 7. 未決事項（作業しながら判定する / 判定できないならユーザーに問う）

| # | 論点 | 既定の判断 |
| :--- | :--- | :--- |
| U-1 | どちらの動作点を主とするか | **CR 50 (16ch)** を主、CR 40 (17ch) を従。理由: 圧縮率優先の用途想定。両方通す |
| U-2 | encoder を NPU に置くか CPU に置くか | 両方実装して測る。既定は **CPU encoder + iGPU decoder**（実測で最速）。`--device` で上書き可 |
| U-3 | JPEG-LS の OpenVINO 化をどこまでやるか | 主目標（1〜3）完了後に着手。副目標なので主目標を人質にしない |
| U-4 | fp16 / NPU_TURBO 等の最適化 | 主目標完了後の改善フェーズ。まず正しさ、次に速度 |
