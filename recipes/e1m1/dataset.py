import hashlib
import json

import cv2
import numpy as np

from common import BUTTONS, TOP, HEIGHT
from sampling import Dataset, Episode

SIZE, HEIGHT_OUT = 192, 144
FORMAT = "e1m1-mixed-rgb-v1"
COUNTS = {"success": 50, "death": 100, "human": 89}


def resize_rgb(rgb):
    """Accept a native 4:3 view or the collector's 196px letterbox."""
    if rgb.shape == (196, 196, 3):
        rgb = rgb[TOP : TOP + HEIGHT]
    if rgb.shape not in ((240, 320, 3), (HEIGHT, 196, 3)):
        raise ValueError(f"Unexpected E1M1 image shape {rgb.shape}")
    return cv2.resize(rgb, (SIZE, HEIGHT_OUT), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)


def inference_frames(history):
    import torch

    out = np.zeros((len(history), 3, SIZE, SIZE), dtype=np.uint8)
    for i, frame in enumerate(history):
        if frame.shape == (SIZE, SIZE, 3):
            out[i] = frame.transpose(2, 0, 1)
        else:
            out[i, :, 24:168] = resize_rgb(frame)
    return torch.from_numpy(out)


class MixedDataset(Dataset):
    def __init__(self, path, size):
        raw = path.read_bytes()
        manifest = json.loads(raw)
        rows = manifest["episodes"]
        self.buttons = tuple(manifest["buttons"])
        if (
            manifest["format"] != FORMAT
            or size != SIZE
            or self.buttons != BUTTONS
            or {s: sum(r["source"] == s for r in rows) for s in COUNTS} != COUNTS
            or len(set(r["sha256"] for r in rows)) != sum(COUNTS.values())
        ):
            raise ValueError("Unexpected mixed dataset")
        self.train, self.dev, self.episodes, self.groups = [], [], [], []
        self.ref = next(
            i for i, r in enumerate(rows) if r["id"] == manifest["reference_id"] and r["source"] == "success"
        )
        if "dev_start" in rows[self.ref]:
            raise ValueError("Visual goal reference must be entirely in training")
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        for split in ("train", "dev"):
            for row in rows:
                if split == "dev" and "dev_start" not in row:
                    continue
                group = row["source"]
                if ("labels" in row) != (group == "success"):
                    raise ValueError("Only successful auto-player trajectories may supply action labels")
                frames = np.load(path.parent / row["bank"], mmap_mode="r")
                if frames.dtype != np.uint8 or frames.shape != (row["frames"], 3, SIZE, SIZE):
                    raise ValueError("RGB bank mismatch")
                cut = row.get("dev_start", len(frames))
                start, end = (0, cut) if split == "train" else (cut, len(frames))
                tics = np.arange(0, end - start, 4)
                actions = expert = goals = None
                if group == "success":
                    with np.load(path.parent / row["labels"]) as labels:
                        a, g = labels["actions"], labels["goals"]
                    if hashlib.sha256(a.tobytes() + g.tobytes()).hexdigest() != row["label_digest"]:
                        raise ValueError("Supervision checksum mismatch")
                    actions, goals = a[(start + tics) // 4], g[(start + tics) // 4]
                    if goals.min() < 0 or goals.max() >= rows[self.ref]["frames"]:
                        raise ValueError("Goal outside training reference")
                    expert = np.ones(len(tics), bool)
                indices = self.train if split == "train" else self.dev
                indices.append(len(self.episodes))
                self.episodes.append(Episode(frames[start:end], tics, actions, goals, expert))
                self.groups.append(group)
        self.ref_for_episode = [self.ref] * len(self.episodes)

    def rows(self, indices, gap=0, labels=False):
        if labels:
            indices = [i for i in indices if self.groups[i] == "success"]
        return super().rows(indices, gap, labels)
