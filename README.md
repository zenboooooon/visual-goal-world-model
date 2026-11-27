# Visual Goal World Model

現在と目標の画像履歴を受け取り、9段の潜在状態と次の行動を予測するPyTorchパッケージです。必要な外部ライブラリはPyTorchだけです。映像の取得、前処理、教師にする時刻、操作ラベルの意味は呼び出し側のプロジェクトで決めます。

## インストールと使い方

```bash
python -m pip install .
```

```python
import torch
from visual_goal_world_model import Config, WorldModel

# 例として単眼グレースケール映像と、同時に選べる5種類の行動を使う
cfg = Config(temporal_mode="patch", image_size=64, channels=1, action_dim=5)
model = WorldModel(cfg).eval()
current = torch.rand(2, 4, 1, 64, 64)
goal = torch.rand_like(current)
with torch.inference_mode():
    logits = model(current, goal)       # [2, 5]
    probabilities = logits.sigmoid()
```

| `temporal_mode` | 時系列Transformerに入れるもの |
|---|---|
| `"pooled"`（既定） | 各時刻の空間ViTのCLSトークン。直近の時刻の出力を使います。 |
| `"patch"` | 現在画像の各パッチが4枚すべての画像のパッチにattentionし、現在画像の出力パッチを平均します。 |

両方式で入力形式は同じです。現在・目標ともに古い順の4枚を `[B, 4, C, H, W]`、値域0〜1の浮動小数点テンソルで渡します。辺の長さは `image_size` で、`patch_size` の倍数を使います。短い履歴は左側を埋めて、`current_valid` と `goal_valid` に形状 `[B, 4]` のboolマスクを渡します。末尾は有効な観測にします。組み込みの操作損失は複数の行動が同時に成立する二値ラベルを扱います。

## ラベル付き映像とラベルなし映像の共同学習

1. 現在と目標の履歴を `model.model.encoder` で符号化し、各予測段に対応する将来の履歴を `model.model.target_encoder` で符号化します。教師の順序は遠い時刻から近い時刻で、9段の形状は `[B, 9, width]` です。教師フレームの選択はデータローダーが行います。
2. すべての映像から `model.world_loss(current_z, goal_z, stage_targets)` と `model.transition_loss(current_z, future_z)` を計算します。
3. 操作教師がある行だけを選び、`model.action_loss(labeled_current_z, labeled_goal_z, targets)` を加えます。`targets` は `[B_labeled, action_dim]` の0/1です。ラベルのないバッチでは映像の損失だけを使えます。
4. optimizer更新後に `model.model.update_target()` を呼びます。EMAの既定係数は0.996です。

## 保存・復元と既存重みの移行

```python
model.action_names = ["a", "b", "c", "d", "e"]  # 任意のラベル順
model.save("model.pt")
restored = WorldModel.load("model.pt")
```

他プロジェクトの同一構成の重みは、元のチャネル数、画像サイズ、時系列方式、各層の幅、出力数を指定して読み込みます。外部ファイルの辞書構造は呼び出し側で処理します。

```python
saved = torch.load("policy.pt", map_location="cpu", weights_only=True)
model = WorldModel(Config(temporal_mode="pooled", image_size=192, channels=3,
                          width=512, wm_hidden=512, latent_action_dim=16, action_dim=8))
model.load_state_dict(saved["policy"], strict=True)
model.eval()
```

テストは `python -m unittest discover -s tests` で実行できます。
