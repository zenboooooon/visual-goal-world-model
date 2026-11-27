# ViZDoom World Model

4枚の観測画像と4枚の目標画像から次の操作を予測するPyTorchモデルです。E1M1実験の標準設定はRGB 192×192、潜在幅512、WM幅512、再帰9段、潜在行動16次元、操作8ボタンです。パッケージはゲーム本体・録画・実験ディレクトリから独立しています。

## インストールと推論

```bash
python -m pip install .
```

```python
import torch
from vizdoom_world_model import Config, WorldModel

model = WorldModel(Config(temporal_mode="pooled")).eval()
current = torch.rand(1, 4, 3, 192, 192)
goal = torch.rand_like(current)
with torch.inference_mode():
    logits = model(current, goal)  # [batch, action_dim]
```

| `temporal_mode` | 時系列Transformerの入力と出力 |
|---|---|
| `"pooled"` | 各時刻の空間ViTのCLSを1個ずつ入力し、現在時刻の出力を使います。 |
| `"patch"` | 現在時刻の各パッチをクエリ、4時刻の全パッチをキー・値として注意を計算し、現在時刻の最終パッチを平均します。 |

両方式とも空間ViTは8×8のパッチを使います。パッチ時系列版は `WorldModel(Config(temporal_mode="patch"))` で初期化します。設定は保存ファイルにも記録されます。他の環境に渡すときは `action_dim` と入力サイズ・チャネル数を指定できます。画像の一辺はパッチサイズの倍数を使います。

入力は古い順の4枚で、値域は0〜1です。E1M1では35 Hzの映像から4 tic間隔で取り、4:3のRGB画像を192×144に縮小して上下に24画素ずつ黒帯を付けます。短い履歴は左を埋め、`current_valid` と `goal_valid` にそれぞれ形状 `[B,4]` のboolマスクを渡します。末尾のフレームは常に有効です。出力は独立なボタンごとのlogitです。

## 自分のデータでの共同学習

1. 現在と目標の画像履歴を `model.model.encoder` で潜在に変換します。WM教師の各時刻は `model.model.target_encoder` で変換します。
2. ラベルの有無にかかわらず、`model.world_loss(current_z, goal_z, stage_targets)` と `model.transition_loss(current_z, future_z)` を計算します。E1M1では目標を1〜10秒先から選び、段の教師は元の時間差の89、78、67、56、45、34、23、12%に最も近いtic（下限8、上限は目標の1 tic前）、最終段は8 tic先です。遷移の教師も8 tic先です。
3. 操作ログのあるサンプルだけを選んで `model.action_loss(labeled_current_z, labeled_goal_z, buttons)` を加えます。ボタン教師は形状 `[B,action_dim]` の0/1です。E1M1では成功軌跡の参照ウェイポイントを操作の目標画像に使います。教師がないバッチではこの損失をスキップできます。
4. 合計損失を逆伝播し、optimizer更新後に `model.model.update_target()` を呼びます。既定のEMA係数は0.996です。

E1M1ではオンラインエンコーダーに学習率3e-5、WMと行動ヘッドに5e-4のAdamWを使っています。推論時はWMが予測した最終段と現在潜在を行動エンコーダーに渡し、行動デコーダーでボタンを予測します。

## 保存と復元

```python
model.save("model.pt")
restored = WorldModel.load("model.pt")
# E1M1 RGB・非パッチ型の既存policy.ptも同じAPIで読み込めます。
# ボタン順は restored.buttons で確認できます。
```

新規モデルのボタン順はデータ提供側で決め、必要に応じて保存前に `model.buttons` に設定します。既存E1M1重みの読み込みではチェックポイントのボタン順を引き継ぎます。

```bash
python -m unittest discover -s tests
```
