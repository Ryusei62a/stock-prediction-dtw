# Stock Prediction with DTW / DDTW and Transformer

過去の株価チャートから類似パターンを探索し、その後の値動きを特徴量として Transformer に入力する、株価方向予測の実験プロジェクトです。

現在は Apple (`AAPL`) の終値データを対象に、直近30日間の値動きから「今後10日間で上昇するか」を2値分類します。

## 概要

このプロジェクトでは、単純な移動平均や前日比だけではなく、「今のチャートに似ていた過去の局面では、その後どう動いたか」をモデルに与えることを目的にしています。

主な処理の流れは以下です。

```text
Yahoo Finance から株価データを取得
↓
直近30日間の終値パターンを作成
↓
DTW / DDTW で過去の類似パターンを探索
↓
類似上位5件の距離・その後リターンを特徴量化
↓
Transformer で今後10日間の上昇/非上昇を分類
↓
評価指標と最新データに対する予測確率を出力
```

## 作成目的

就職活動用のポートフォリオとして、以下を示すために作成しました。

- 時系列データに対する特徴量設計
- DTW / DDTW を用いた類似パターン探索
- PyTorch による Transformer モデル実装
- GPU を使った計算処理
- Accuracy だけに依存しない評価指標の確認
- 実験結果から改善点を考察する姿勢

## 技術スタック

- Python
- NumPy
- yfinance
- PyTorch
- scikit-learn
- CUDA 対応 GPU

## 主な特徴

### 1. DTW による類似チャート探索

DTW (Dynamic Time Warping) を使い、現在の30日間の値動きと過去の30日間の値動きを比較します。

通常の距離計算ではタイミングのズレに弱いですが、DTW を使うことで、多少タイミングがずれた類似パターンも評価できます。

### 2. DDTW による変化率パターンの比較

DDTW (Derivative DTW) では、価格そのものではなく差分系列を比較します。

これにより、価格水準ではなく「上がり方・下がり方」の形も特徴として使います。

### 3. 類似パターンのその後リターンを特徴量化

類似度が高い過去パターン上位5件について、以下を特徴量として使います。

- DTW と DDTW を組み合わせたスコア
- DTW 距離
- DDTW 距離
- 類似パターンのその後10日間リターン

これらを直近30日間の正規化済み終値と結合し、Transformer に入力します。

### 4. Transformer による2値分類

入力データは以下の形になります。

```text
サンプル数 x 30日 x 特徴量数
```

Transformer Encoder により、30日間の時系列特徴量から今後10日間で上昇するかを分類します。

### 5. GPU 対応

PyTorch が CUDA 対応 GPU を認識できる場合、以下の処理で GPU を使用します。

- DTW / DDTW のバッチ計算
- Transformer の学習
- Transformer の評価
- 最新データの推論

実行時には以下のように使用デバイスを表示します。

```text
DEVICE: cuda
GPU: NVIDIA GeForce RTX 3060 Ti
```

## 評価指標

モデルの評価では、Accuracy だけでなく以下も出力します。

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- 常に上昇と予測した場合のベースライン
- 多数派クラスを常に予測した場合のベースライン

株価データは上昇/下落ラベルが偏ることがあるため、Accuracy だけではモデルの有効性を判断しにくいです。そのため、ベースラインと比較しながら予測力を確認します。

## 出力例

```text
DEVICE: cuda
GPU: NVIDIA GeForce RTX 3060 Ti
close_data shape: (753,)
X shape: (678, 30, 21)
y shape: (678,)
label distribution: {0: 294, 1: 384}
positive ratio: 0.5664
Baseline always up | Acc: 0.6324, Precision: 0.6324, Recall: 1.0000, F1: 0.7748
Epoch 10/10 | Train Loss: 0.6830, Train Acc: 0.5517 | Test Loss: 0.6738, Test Acc: 0.6103, Precision: 0.6260, Recall: 0.9535, F1: 0.7558, AUC: 0.4660

Latest prediction
Ticker: AAPL
Up probability over next 10 days: 0.6248
Prediction: UP
```

`Up probability` はモデルが出した最新データに対する上昇確率であり、モデル全体の正答率ではありません。モデルの性能は `Test Acc` や `AUC`、ベースライン比較で判断します。

## 実行方法

```bash
pip install yfinance numpy torch scikit-learn
python benkyou.py
```

CUDA 対応版の PyTorch を使用する場合は、環境に合わせて公式のインストール手順で導入してください。

例:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 現在の課題

現時点では、モデルが「上昇」と予測しやすく、単純なベースラインに勝てないケースがあります。

特に、AUC が 0.5 前後またはそれ以下の場合、モデルは上昇しやすい日をうまく順位付けできていないと判断できます。

## 今後の改善案

- 目的変数を `future_change > 0` ではなく、`+2%以上上昇` などに変更する
- 上昇 / 下落 / 見送り の3クラス分類にする
- 出来高、RSI、移動平均乖離率、ボラティリティなどの特徴量を追加する
- `SPY` や `QQQ`、`VIX` など市場全体の情報を加える
- LightGBM や Random Forest などのモデルと比較する
- walk-forward validation による時系列検証を行う
- バックテストを追加し、売買ルールとして有効か評価する

## 注意事項

このプロジェクトは学習・研究目的の実験コードです。投資判断を目的としたものではありません。

株式投資には損失リスクがあり、モデルの出力は将来の値動きを保証するものではありません。
