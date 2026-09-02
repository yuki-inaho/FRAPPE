# アイデアメモ: 学習済みFRAPPE decoderの量子化フレンドリー化

状態: **研究アイデア。未実装・未検証**

このメモはONNXのDeconv多相分解とは別件である。目的は、まず
`LayerNorm + GELU`を持つ高品質なFP32 teacherを学習し、その重みを出発点として
decoderだけを量子化・推論しやすいstudent構造へ移植することである。

## 結論

実施可能で、FRAPPEでは試す価値が高い。ただし「活性化関数だけを置換し、ほぼ全重みを
freezeすれば自動的に回復する」とは限らない。最初は正規化層とaffineだけを学習し、回復が
不足したときに置換block内のdepthwise Conv、2個のpointwise Conv、LayerScaleを段階的に
unfreezeするのが安全である。

encoder、compander、整数符号生成は最後までfreezeできる。したがって、decoder surgeryが
正しく分離されていれば、JPEG-LS符号とbppはteacherと完全に同一であり、最適化対象は
reconstruction品質とdecoder速度に限定できる。

## 現在のFRAPPE blockと高速化余地

`ConvBlockND`は次のConvNeXt型残差blockである。

```text
input
  ├───────────────────────────────────────────────┐
  └─ depthwise Conv ─ LayerNormND ─ Conv 1x1 expand
       ─ GELU ─ Conv 1x1 shrink ─ [LayerScale] ─ add
```

`LayerNormND`はNCHWをchannels-lastへ移動し、LayerNorm後にNCHWへ戻す。現在のCR-50
ONNX decoderでは6個のLayerNormと、それらに付随する12個のTransposeが存在する。
GELUは6個ある。

OpenVINO CPUは既に各GELUを直前のConvと実行時融合している。そのため、GELUをReLUへ
変えるだけでは大きなCPU高速化は期待しにくい。より大きな候補はLayerNormを
`BatchNorm2d`へ置換し、推論時に直前のdepthwise Convへfoldすることである。これが成功すれば、
LayerNorm、前後Transpose、BN自体を実行グラフから除去できる可能性がある。

## 候補student

一度に全要素を変えず、差分を分離する。

| ID | 正規化 | 活性化 | 目的 | リスク |
|---|---|---|---|---|
| T0 | LayerNorm | GELU | FP32 teacher / 基準 | 現行構造 |
| S1 | LayerNorm | ReLU | 活性化置換だけの影響を見る | TransposeとLayerNormは残る |
| S2 | BatchNorm2d | GELU | 正規化置換だけの影響を見る | GELUは正斉次でない |
| S3 | BatchNorm2d | LeakyReLU | 負値経路を残し、正斉次性を得る | teacherとの差が残る |
| S4 | BatchNorm2d | ReLU | 最も単純なINT8・fold構造 | dead activationと品質低下 |
| S5 | 静的channel affine | GELU/ReLU | calibration回帰だけでLNを近似 | 入力依存の正規化を失う |

第一候補はS3、最大速度候補はS4である。LeakyReLUは正のchannel scale `s` に対して

```text
LeakyReLU(s*x) = s*LeakyReLU(x),  s > 0
```

を満たし、負の小信号を完全には捨てない。一方GELUはこの等式を満たさない。

BatchNormもそれ自体はoffsetを含むため正斉次ではないが、推論時には固定affineである。

```text
BN(x) = a*x + c
W_fold = a*W
b_fold = a*b + c
```

として直前Convへ厳密にfoldできる。scale移送は、BN fold後に残るReLUまたはLeakyReLUの
正斉次性に対して利用する。

## teacherからstudentへの初期化

### Convと残差経路

- depthwise Conv、expand 1x1 Conv、shrink 1x1 Convはteacherからコピーする。
- LayerScaleもコピーする。
- skip connectionは変更しない。
- ReLU/LeakyReLUには学習パラメータがないため、初期差は蒸留で吸収する。

### LayerNormからBatchNorm

LayerNormは各空間位置でchannel方向を正規化し、BatchNormはchannelごとにbatch・空間方向の
統計を使う。したがって、重みとbiasをコピーするだけでは等価にならない。

初期化にはcalibration画像上の回帰を使う。各blockについてdepthwise Conv出力 `x[c]` と、
teacher LayerNorm出力 `y[c]` を保存し、channelごとに

```text
y[c] ≈ a[c] * x[c] + c[c]
```

を最小二乗またはrobust regressionで求める。BatchNormのrunning mean `mu`、variance `var`
を同じcalibration集合から計算し、推論時BNがこのaffineに近くなるよう、

```text
gamma[c] = a[c] * sqrt(var[c] + eps)
beta[c]  = c[c] + a[c] * mu[c]
```

で初期化する。この回帰はLayerNormの入力依存性を再現できないが、ランダム初期化より
teacherに近い開始点になる。

NNCFには圧縮モデルへ統計を合わせるBatchNorm adaptationと、既定で2000 sampleを使う設定が
ある。実装時は独自回帰後のBN adaptation有無をablationする。

## 段階的fine-tuning

### Phase 0: teacherを固定

- 学習済みFP32 checkpointをteacherとしてロードする。
- `eval()`、`requires_grad_(False)`とし、teacher出力は必ず`no_grad()`で生成する。
- teacherとstudentは同じ整数codesをdecoder入力として受け取る。
- full-prefixだけでなく、運用する全prefixを蒸留対象にする。

### Phase 1: 統計・affineだけを適応

学習対象:

- 新しいBatchNormの`weight`、`bias`、running statistics
- prefix別`prefix_scale`、`prefix_bias`は最初はfreezeし、必要時のみ後半で開放

その他はfreezeする。まず数百〜数千iterationで、構造置換そのものによる差がaffineだけで
どこまで回復するかを見る。

### Phase 2: 置換blockを局所的に開放

Phase 1で足りなければ、各置換blockについて次をunfreezeする。

- depthwise Conv
- expand 1x1 Conv
- shrink 1x1 Conv
- LayerScale
- 各Convのbias

最終headとfirst Conv、encoderはまだfreezeする。全blockを同時に変える方法と、出力に近い
最後のblockから1個ずつ置換・回復する方法を比較する。後者は失敗位置を特定しやすい。

### Phase 3: decoder全体の低学習率調整

品質差が残る場合に限り、`first`、trunk全体、prefix affine、headを低学習率で開放する。
encoderとcompanderは開放しない。これにより整数codesとJPEG-LS streamを維持する。

学習率は、例として新規BN/affineを`1.0`、置換block Convを`0.1`、既存first/headを`0.01`
という倍率に分ける。具体値は短いsweepで決める。

## 蒸留・再構成loss

単一lossだけに依存しない。

```text
L = lambda_rgb     * Charbonnier(student_rgb, target_rgb)
  + lambda_teacher * L1(student_rgb, teacher_rgb)
  + lambda_feature * sum(normalized_MSE(student_block_i, teacher_block_i))
  + lambda_edge    * gradient_loss(student_rgb, target_rgb)
```

- target RGBは本来の再構成目標を維持する。
- teacher RGB lossは構造置換による急な出力変化を抑える。
- block feature lossは、後段だけで誤差を帳尻合わせするのを防ぐ。
- featureはchannel数と空間サイズで正規化し、巨大な初期blockがlossを独占しないようにする。
- rate lossはencoderをfreezeする限り不要。ただしcodes・JPEG-LS bytesの一致を回帰テストする。

## FP32回復後のINT8 QAT

構造置換と量子化を同時に開始しない。先にFP32 studentを十分回復させ、その後NNCFの
fake quantizerを挿入する。

1. FP32 teacherを学習済みcheckpointからロード。
2. student surgeryとFP32 distillationを実施。
3. FP32 studentが品質gateを通過したcheckpointを固定基準として保存。
4. calibration dataでNNCF quantizer rangeとBN statisticsを初期化。
5. 最初はquantizer range、BN affine、LayerScaleのみを更新。
6. 必要に応じて置換block Convを低学習率で開放。
7. QAT済みPyTorch → ONNX → OpenVINO IRの順でexport。
8. BNがConvへfoldされたこと、LayerNorm/Transposeが消えたこと、INT8 Convになったことを
   OpenVINO runtime graphで確認。

NNCFはactivation/weight quantizationの開始時期を分けられるため、最初にweight fake quant、
次にactivation fake quantという段階導入も比較する。

## 最小実験行列

各構造についてFP32 surgery後とINT8 QAT後を測る。

| 実験 | 構造 | 学習範囲 |
|---|---|---|
| E0 | LN + GELU | teacher、学習なし |
| E1 | LN + ReLU | norm/affineのみ → block Conv |
| E2 | BN + GELU | BNのみ → block Conv |
| E3 | BN + LeakyReLU | BNのみ → block Conv |
| E4 | BN + ReLU | BNのみ → block Conv |
| E5 | E3のINT8 QAT | quantizer → block Conv |
| E6 | E4のINT8 QAT | quantizer → block Conv |

最初は1 blockだけ置換するsmoke testを行い、その後6/12 blockへ広げる。全置換を最初から
長時間学習するのは避ける。

## 評価と暫定gate

### 必須の不変条件

- teacherとstudentで全prefixの整数codesがbit-exact。
- JPEG-LS payload bytesがbit-exact。
- 画像名、元データ名、絶対private pathをreportへ保存しない。
- FP32、QAT PyTorch、ONNX Runtime、OpenVINOで同じ評価集合を使う。

### 暫定的な品質・速度gate

- FP32 surgery: teacher比PSNR低下0.05 dB以内。
- INT8 QAT: 対応するFP32 student比PSNR低下0.10 dB以内。
- 最悪prefixでもPSNR低下0.15 dB以内。
- target device decoder中央値がteacher OpenVINO比10%以上改善。
- BN fold後にLayerNorm 0、不要Transpose 0を構造検査で確認。

これらは論文値ではなく、最初の継続・中止判断用のengineering gateである。速度改善がないなら、
品質gateを通っても構造変更を配備する理由は弱い。

## 主なリスク

1. **LayerNormとBatchNormの正規化軸が違う。** affineだけでは回復せず、block Convの調整が
   必要になる可能性が高い。
2. **小batchのBN。** 空間次元を含むため統計は取れるが、データ分布依存性が強い。学習後は
   running statisticsを固定し、calibration/holdoutの分離が必要。
3. **ReLUのdead channel。** teacher GELUの負側情報を失う。LeakyReLUを先に試す理由である。
4. **full-prefixだけの過適合。** prefixごとに入力分布が違うため、全運用prefixをsampleする。
5. **FP32回復とINT8回復の混同。** 構造差と量子化差を別checkpoint・別reportで測る。
6. **CPUとNPUの最適構造差。** CPUでのBN fold・ReLU融合がNPUの速度を保証しない。

## 採用判断

この案の本質はGELUをReLUへ近似することだけではない。

```text
高品質なLN+GELU teacher
  -> decoder-only architecture surgery
  -> teacher distillationでFP32品質回復
  -> BN fold可能・正斉次activationのstudent
  -> NNCF QAT
  -> ONNX/OpenVINO target-device検証
```

というteacher/student変換である。encoderを完全に維持できるFRAPPEでは、符号互換性を
壊さずdecoderだけを大胆に最適化できる点が強い。

関連する別方向の手法として、RepQ-ViTはpost-LayerNorm activationのchannel間range差を
scale reparameterizationで扱う。構造置換の品質回復が難しい場合は、LayerNormを残したまま
量子化scale表現だけを変えるbaselineとして比較する。

参考:

- [RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers](https://arxiv.org/abs/2212.08254)
- [NNCF quantization configuration and BatchNorm adaptation](https://openvinotoolkit.github.io/nncf/schema/)
- [NNCF documentation](https://openvinotoolkit.github.io/nncf/)
