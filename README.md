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

## 高速版の共同学習

`visual_goal_world_model.training` に、実験で使った高速化をまとめた1更新分のAPIがあります。モデルと重み形式は両方式で共通です。通常の `model(current, goal)` と検証時の計算は従来どおりです。

```python
from visual_goal_world_model.training import FrameBank, joint_step, make_optimizer

bank = FrameBank(images)  # CPU uint8 [N,C,H,W]
optimizer = make_optimizer(model, encoder_lr=3e-5, wm_lr=5e-4, head_lr=5e-4)
losses = joint_step(model, optimizer, bank,
                    world=(current_ids, goal_ids, stage_ids),
                    transitions=(before_ids, future_ids),
                    actions=(labeled_ids, reference_goal_ids, labels),
                    fast=True, microbatch=32)
```

| 引数 | データの渡し方 |
|---|---|
| `world` | 現在と目標は `[B,4]`、9段の教師は遠い順の `[B,9,4]`。 |
| `transitions` | 現在と実際の未来の履歴をそれぞれ `[T,4]` で渡す。 |
| `actions` | 操作ラベルのある現在・目標の履歴は `[A,4]`、ラベルはCPUテンソル `[A,action_dim]`。映像だけの更新では `None`。 |

履歴の各要素はフレーム表に対するCPUの `torch.int64` 行番号です。複数の軌跡を扱う場合は全軌跡で一意の番号を割り当て、各履歴を同じ軌跡の中で作ります。左パディングは `-1`、最後の要素は有効な番号にします。同じ番号には同じ前処理済み画像を割り当てます。異なるデータ拡張を適用した画像には別の番号を使います。

### 実際に省ける計算

- **同じ目標画像の再利用**。サンプラーが選んだ操作行を、目標履歴が同じ行からまとめて処理します。対応する操作ラベルも一緒に並べ替えます。各マイクロバッチで現在・目標をまとめ、重複フレームの空間ViTを一度だけ計算します。EMA教師の特徴とオンラインの特徴は別々に計算します。
- **画像と二値マスクの転送**。`FrameBank(images, mask_bits)` は、画像と展開済みマスクを同じuint8出力バッファに詰めてからデバイスに転送します。`mask_bits` は `[N,H,ceil(W/8)]` で、上位ビットから画素を割り当てます。モデルの `channels` は画像のチャネル数に1を加えます。RGBだけの場合は `FrameBank(images)` が使えます。
- **最終層のCLS計算**。`pooled` では最後の空間attentionのクエリをCLSの1個に絞り、キー・値には全パッチを残します。`patch` は時系列層で全パッチを使います。VRAMが足りない場合は `checkpoint_patches=True` で逆伝播時に活性を再計算できます。
- **マイクロバッチ**。`fast=True` は既定32、`fast=False` は既定16です。操作・遷移損失は各分割の行数で重み付けして勾配を蓄積します。WMの分散・共分散は元のWMバッチ全体で計算し、optimizerとEMAをそれぞれ1回更新します。

対応CUDA環境では自動でBF16を使います。`bf16=False` でFP32を選べます。実験で用いた一例はWMバッチ20、操作バッチ128、勾配ノルム上限1、EMA係数0.996です。実際に動く合成データ例でモードを切り替えられます。

```bash
python examples/joint_update.py --mode pooled
python examples/joint_update.py --mode pooled --baseline
python examples/joint_update.py --mode patch --device cuda
```

### 元の実験での速度

RTX 5090で別の学習ジョブが動いている条件で、同じ保存点と5組のバッチを使い、各条件を1回ウォームアップして測定しました。元のRGB・深度・二値マスク付き `pooled` 実験の値です。表はWM・遷移・操作の順伝播と逆伝播の合計時間の中央値、および5回中の最大GPU割当量です。optimizer更新、EMA、検証、保存の時間は計測外です。

| 条件 | 秒/バッチ | ピーク割当量 |
|---|---:|---:|
| 元の通常版、マイクロバッチ16 | 0.960 | 4.23 GiB |
| 目標順ソート・バッファへの格納・マイクロバッチ32 | 0.774 | 5.16 GiB |
| 最終層のCLS計算のみ適用 | 0.795 | 2.76 GiB |
| 4つすべて | 0.542 | 3.42 GiB |

この測定では全手法の併用で約1.77倍でした。入力・目標の重複率、GPU、時系列方式で効果は変わります。元のBF16比較では最初のバッチの通常版と高速版の勾配のコサイン類似度は0.99978でした。移植版のテストでは、両時系列方式についてCPU・FP32で出力と勾配も比較しています。
