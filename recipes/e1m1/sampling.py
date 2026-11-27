from dataclasses import dataclass
import hashlib

import numpy as np
import torch


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(8 << 20), b""):
            result.update(piece)
    return result.hexdigest()


@dataclass
class Episode:
    frames: np.ndarray
    tics: np.ndarray
    actions: np.ndarray | None
    goals: np.ndarray | None
    expert: np.ndarray | None


class Dataset:
    def rows(self, indices, gap=0, labels=False):
        chunks = []
        for idx in indices:
            ep = self.episodes[idx]
            valid = ep.tics + gap < len(ep.frames)
            if labels:
                if ep.expert is None:
                    raise ValueError("Action supervision is disabled")
                valid &= ep.expert
            decisions = np.flatnonzero(valid)
            chunks.append(np.column_stack((np.full(len(decisions), idx), decisions)))
        return np.concatenate(chunks)

    def _action_goal(self, idx, decision, horizon=None):
        ep = self.episodes[idx]
        if horizon is None:
            return self.ref_for_episode[idx], ep.goals[decision]
        return idx, ep.tics[decision] + 35 * horizon

    def sort_action_rows(self, rows):
        """Group already drawn actions so each microbatch can reuse its repeated goal frames."""
        rows = np.asarray(rows)
        order = sorted(range(len(rows)), key=lambda i: self._action_goal(*rows[i]))
        return rows[order]

    def action_batch(self, rows):
        current, goals, actions = [], [], []
        for row in rows:
            idx, decision = row[:2]
            ep = self.episodes[idx]
            current.append((idx, ep.tics[decision]))
            goals.append(self._action_goal(*row))
            if ep.actions is None:
                raise ValueError("Action supervision is disabled")
            actions.append(ep.actions[decision])
        return current, goals, np.stack(actions)

    def pack(self, points):
        """Deduplicate frames across histories; early histories use the temporal mask."""
        unique, indices, masks = {}, [], []
        for idx, tic in points:
            ticks = (tic - 12, tic - 8, tic - 4, tic)
            masks.append([t >= 0 for t in ticks])
            indices.append([unique.setdefault((idx, max(0, t)), len(unique)) for t in ticks])
        frames = np.stack([self.episodes[idx].frames[tic] for idx, tic in unique])
        return torch.from_numpy(frames), torch.tensor(indices), torch.tensor(masks)


class Cycle:
    """Absolute cursor determines the permutation on resume or budget extension."""

    def __init__(self, rows, seed, stream):
        if not len(rows):
            raise ValueError(f"No eligible examples for sampling stream {stream}")
        self.rows, self.seed, self.stream = rows, seed, stream
        self.position, self.epoch, self.order = 0, None, None

    def draw(self, count):
        parts = []
        while count:
            epoch, offset = divmod(self.position, len(self.rows))
            if epoch != self.epoch:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.stream, epoch]))
                self.order, self.epoch = rng.permutation(len(self.rows)), epoch
            n = min(count, len(self.rows) - offset)
            parts.append(self.rows[self.order[offset : offset + n]])
            self.position += n
            count -= n
        return np.concatenate(parts)
