"""Pinned E1M1 seed-17 joint-training protocol. Paths are relative to this recipe."""

import argparse
from copy import deepcopy
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from visual_goal_world_model import Config, WorldModel
from visual_goal_world_model.encoder import HistoryEncoder, ModelConfig
from visual_goal_world_model.heads import LatentActionHead, ReverseWM
from visual_goal_world_model.training import make_optimizer
from common import BUTTONS, DATA, HERE, json_write
from dataset import MixedDataset, inference_frames
from sampling import Cycle, file_hash

RUNS = Path(os.environ.get("E1M1_RUNS", HERE / "runs")).resolve()
MANIFEST = DATA / "e1m1_mixed_rgb/manifest.json"
BUDGET = {"base": 25000, "control": 50000, "higher": 50000}


class Policy(WorldModel):
    rgb_only = True
    buttons = BUTTONS

    def __init__(self):
        # Reproduce the original initialization draws, including the original
        # RGBD projection and seven-output head that were resized before training.
        with torch.random.fork_rng(devices=[]):
            super().__init__(Config())
            torch.manual_seed(17)
            enc = HistoryEncoder(
                ModelConfig(channels=4, image_size=192, width=512, spatial_layers=2, temporal_layers=2),
                "pooled",
            )
            spatial = enc.encoder.spatial
            old = spatial.patch_embed
            new = nn.Conv2d(3, 512, 8, stride=8)
            with torch.no_grad():
                new.weight.copy_(old.weight[:, :3])
                new.bias.copy_(old.bias)
            spatial.patch_embed = new
            spatial.config = replace(spatial.config, channels=3)
            enc.encoder.config = spatial.config
            self.model.encoder = enc
            self.model.target_encoder = deepcopy(enc).eval().requires_grad_(False)
            torch.manual_seed(17 + 17019)
            self.model.wm = ReverseWM(512, 512)
            torch.manual_seed(17 + 20019)
            self.head = LatentActionHead(512, 16, 7)
            self.head.decoder[-1] = nn.Linear(256, 8)

    @torch.inference_mode()
    def act(self, frames, _depth, goal, _goal_depth):
        device = next(self.parameters()).device
        images, valid = [], []
        for hist in (frames, goal):
            x = inference_frames(hist)
            pad = 4 - len(x)
            images.append(torch.cat([x[:1].expand(pad, -1, -1, -1), x]))
            valid.append([False] * pad + [True] * len(x))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = self.model.encoder(
                torch.stack(images).to(device, torch.float32) / 255.0, torch.tensor(valid, device=device)
            )
            logits = self.logits(z[:1], z[1:])[0].float().cpu().numpy()
        result = (logits > 0).astype(np.uint8)
        for i, j in ((4, 5), (2, 3), (0, 1)):
            if result[i] and result[j]:
                result[j if logits[i] > logits[j] else i] = 0
        return result


def load_policy(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = Policy()
    model.load_state_dict(state["policy"])
    return model.to(device).eval()


def encode(data, encoder, points, chunk, device, fast):
    frames, ix, valid = data.pack(points)
    features = torch.cat(
        [encoder.spatial(x.to(device, torch.float32) / 255.0, cls_only=fast) for x in frames.split(chunk)]
    )
    return encoder.from_spatial(features[ix.to(device)], valid.to(device))


def world(data, policy, windows, device, fast):
    now, goal, targets = [], [], []
    for i, j, h in windows:
        tic = data.episodes[i].tics[j]
        gap = 35 * h
        now.append((i, tic))
        goal.append((i, tic + gap))
        offsets = [min(gap - 1, max(8, (gap * p + 50) // 100)) for p in policy.model.wm.schedule[:-1]]
        targets.extend((i, tic + d) for d in [*offsets, 8])
    chunk = 32 if fast else 16
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        a, b = encode(data, policy.model.encoder, now + goal, chunk, device, fast).chunk(2)
        with torch.no_grad():
            tgt = encode(data, policy.model.target_encoder, targets, chunk, device, fast)
        return policy.world_loss(a, b, tgt.reshape(len(now), 9, -1))


def other_loss(data, policy, rows, device, kind, fast):
    chunk = 32 if fast else 16
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if kind == "transition":
            now = [(i, data.episodes[i].tics[j]) for i, j in rows]
            z = encode(data, policy.model.encoder, now, chunk, device, fast)
            with torch.no_grad():
                target = encode(
                    data, policy.model.target_encoder, [(i, t + 8) for i, t in now], chunk, device, fast
                )
            return policy.transition_loss(z, target)
        now, goal, labels = data.action_batch(rows)
        a, b = encode(data, policy.model.encoder, now + goal, chunk, device, fast).chunk(2)
        return policy.action_loss(a, b, torch.from_numpy(labels).to(device))


def validation_windows(data):
    result = []
    for idx in data.dev:
        by_h = {}
        for h in range(1, 11):
            rows = data.rows([idx], 35 * h)
            by_h[h] = rows[np.linspace(0, len(rows) - 1, 4, dtype=int)]
        for j in (0, 2):
            result.append([(i, t, h) for h in by_h for i, t in by_h[h][j : j + 2]])
    return result


def save(path, obj):
    tmp = path.with_suffix(".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def train(phase):
    run = RUNS / phase
    goal = BUDGET[phase]
    lr = 7.5e-4 if phase == "higher" else 5e-4
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if file_hash(MANIFEST) != file_hash(HERE / "reference/manifest.json"):
        raise ValueError("Dataset manifest differs from the pinned experiment")
    data = MixedDataset(MANIFEST, 192)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires a BF16-capable CUDA device")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model = Policy().to(device)
    opt = make_optimizer(model, wm_lr=lr)
    windows = {h: Cycle(data.rows(data.train, 35 * h), 17, h) for h in range(1, 11)}
    transitions = Cycle(data.rows(data.train, 8), 17, 101)
    actions = Cycle(data.rows(data.train, labels=True), 17, 102)
    valid = validation_windows(data)
    dev = data.rows(data.dev, labels=True)
    last = run / "resume.pt"
    start, best_step, best_score, best = 0, None, None, None
    if last.exists() or phase != "base":
        new_branch = not last.exists()
        source = RUNS / "base/resume.pt" if new_branch else last
        state = load(source)
        if state["phase"] != ("base" if new_branch else phase) or state["dataset"] != data.fingerprint:
            raise ValueError("Resume identity mismatch")
        start = state["step"]
        if new_branch and start != 25000:
            raise ValueError("Base must reach exactly 25,000")
        model.load_state_dict(state["policy"])
        opt.load_state_dict(state["optimizer"])
        opt.param_groups[1]["lr"] = lr
        for h, pool in windows.items():
            pool.position = state["window_positions"][h]
        transitions.position = state["transition_position"]
        actions.position = state["action_position"]
        if not (new_branch and phase == "higher"):
            best_step, best_score, best = state["best_step"], state["best_score"], state["best"]
        if best is not None:
            save(run / "best.pt", dict(step=best_step, policy=best))
        del state
    if start > goal:
        raise ValueError("Checkpoint exceeds requested phase")
    for step in range(start + 1, goal + 1):
        opt.zero_grad(set_to_none=True)
        batch = [(i, j, h) for h, pool in windows.items() for i, j in pool.draw(2)]
        w = world(data, model, batch, device, True)
        w.backward()
        values = {}
        for kind, rows in (
            ("transition", transitions.draw(20)),
            ("action", data.sort_action_rows(actions.draw(128))),
        ):
            total = torch.zeros((), device=device)
            for j in range(0, len(rows), 32):
                sub = rows[j : j + 32]
                loss = other_loss(data, model, sub, device, kind, True) * (len(sub) / len(rows))
                loss.backward()
                total += loss.detach()
            values[kind] = total
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        model.model.update_target()
        if step % 100:
            continue
        model.eval()
        with torch.inference_mode():
            score = torch.stack([world(data, model, b, device, False) for b in valid]).mean().item()
            bce = None
            if step == 100 or step % 1000 == 0:
                total = torch.zeros((), device=device)
                for j in range(0, len(dev), 16):
                    rows = dev[j : j + 16]
                    total += other_loss(data, model, rows, device, "action", False) * len(rows)
                bce = total.item() / len(dev)
        model.train()
        if best_score is None or score < best_score:
            best_step, best_score = step, score
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        record = dict(
            step=step,
            phase=phase,
            wm_lr=lr,
            train_wm=w.item(),
            train_transition=values["transition"].item(),
            train_action=values["action"].item(),
            dev_wm=score,
            dev_action=bce,
        )
        snapshot = dict(
            phase=phase,
            step=step,
            dataset=data.fingerprint,
            policy=model.state_dict(),
            optimizer=opt.state_dict(),
            best=best,
            best_step=best_step,
            best_score=best_score,
            window_positions={h: p.position for h, p in windows.items()},
            transition_position=transitions.position,
            action_position=actions.position,
        )
        if last.exists():
            link = run / "previous.link"
            link.unlink(missing_ok=True)
            os.link(last, link)
            os.replace(link, run / "previous.pt")
        save(last, snapshot)
        if step == best_step:
            save(run / "best.pt", dict(step=step, policy=best))
        if bce is not None:
            json_write(run / "validation_action.json", dict(step=step, bce=bce))
        json_write(run / "progress.json", record)
        print(json.dumps(record), flush=True)
    # Also repairs the export after interruption between final checkpoint and export.
    save(run / "exact.pt", dict(step=goal, policy=model.state_dict(), buttons=list(BUTTONS)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=BUDGET)
    train(p.parse_args().phase)
