"""One joint optimizer update over indexed image histories. See README for batch shapes."""
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class FrameBank:
    """CPU uint8 [N,C,H,W] frames, optionally with one MSB-first packed binary channel."""
    images: torch.Tensor
    mask_bits: torch.Tensor | None = None

    def __post_init__(self):
        if self.images.ndim != 4 or self.images.device.type != "cpu" or self.images.dtype != torch.uint8:
            raise ValueError("images must be a CPU uint8 tensor [N,C,H,W]")
        n, _, height, width = self.images.shape
        if self.mask_bits is not None:
            mask = self.mask_bits
            if (mask.shape != (n, height, (width + 7) // 8)
                    or mask.device.type != "cpu" or mask.dtype != torch.uint8):
                raise ValueError("mask_bits must be CPU uint8 [N,H,ceil(W/8)]")

    def gather(self, ids):
        if self.mask_bits is None:
            return self.images.index_select(0, ids)
        _, channels, height, width = self.images.shape
        batch = self.images.new_empty((len(ids), channels + 1, height, width))
        torch.index_select(self.images, 0, ids, out=batch[:, :channels])
        packed = self.mask_bits.index_select(0, ids)
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
        bits = (packed.unsqueeze(-1) >> shifts).bitwise_and_(1).flatten(-2)
        batch[:, channels].copy_(bits[..., :width]).mul_(255)
        return batch


def encode_histories(encoder, bank, histories, *, microbatch=32, fast=True, checkpoint_patches=False):
    """Encode CPU int64 [B,4] frame IDs. Use -1 for missing observations."""
    if (histories.ndim != 2 or histories.shape[1] != 4 or not len(histories)
            or histories.dtype != torch.int64 or histories.device.type != "cpu" or microbatch < 1):
        raise ValueError("expected nonempty CPU int64 histories [B,4] and a positive microbatch")
    valid = histories >= 0
    if not valid[:, -1].all() or histories.min() < -1 or histories.max() >= len(bank.images):
        raise ValueError("invalid frame index or missing current observation")
    # A masked position still needs an in-bounds index for the gather.
    ids = torch.where(valid, histories, histories[:, -1:]).flatten()
    if fast:
        ids, gather = ids.unique(return_inverse=True)
    else:
        gather = torch.arange(len(ids))
    device = next(encoder.parameters()).device
    patch = encoder.temporal_mode == "patch"
    recompute = patch and checkpoint_patches and torch.is_grad_enabled()

    def spatial(frames):
        return encoder.spatial(frames, cls_only=fast and not patch)

    features = []
    for part in ids.split(microbatch):
        frames = bank.gather(part).to(device=device, dtype=torch.float32).div_(255.)
        features.append(checkpoint(spatial, frames, use_reentrant=False) if recompute else spatial(frames))
    features = torch.cat(features)
    gather = gather.view_as(histories).to(device)
    valid = valid.to(device)
    if not patch:
        return encoder.from_spatial(features[gather], valid)

    def fuse(bank_features, indices, mask):
        return encoder.from_spatial(bank_features[indices], mask)

    outputs = []
    for ix, mask in zip(gather.split(microbatch), valid.split(microbatch)):
        outputs.append(checkpoint(fuse, features, ix, mask, use_reentrant=False)
                       if recompute else fuse(features, ix, mask))
    return torch.cat(outputs)


def make_optimizer(model, *, encoder_lr=3e-5, wm_lr=5e-4, head_lr=5e-4):
    return torch.optim.AdamW([
        {"params": model.model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.model.wm.parameters(), "lr": wm_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ])


def joint_step(model, optimizer, bank, *, world, transitions, actions=None, fast=True,
               microbatch=None, bf16=None, checkpoint_patches=False):
    """Perform one joint update; actions may contain only labeled rows or be omitted."""
    if microbatch is None:
        microbatch = 32 if fast else 16
    device = next(model.parameters()).device
    if bf16 is None:
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    now, goal, targets = world
    if targets.shape != (len(now), len(model.model.wm.schedule), 4) or now.shape != goal.shape:
        raise ValueError("world requires [B,4], [B,4], [B,9,4] histories")
    if len(transitions) != 2 or transitions[0].shape != transitions[1].shape or not len(transitions[0]):
        raise ValueError("transitions requires two equally sized nonempty sets of histories")
    if actions is not None:
        obs, goals, labels = actions
        if (not len(obs) or obs.shape != goals.shape or labels.shape != (len(obs), model.config.action_dim)
                or labels.device.type != "cpu"):
            raise ValueError("actions requires aligned histories and CPU action labels; use None for no labels")
        if fast:
            # Sort the already sampled rows, keeping targets and current histories paired.
            keys = goals.tolist()
            order = sorted(range(len(keys)), key=keys.__getitem__)
            obs, goals, labels = obs[order], goals[order], labels[order]

    def encode(encoder, histories):
        return encode_histories(encoder, bank, histories, microbatch=microbatch,
                                fast=fast, checkpoint_patches=checkpoint_patches)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    online, teacher = model.model.encoder, model.model.target_encoder
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        current_z, goal_z = encode(online, torch.cat((now, goal))).chunk(2)
        with torch.no_grad():
            stage_z = encode(teacher, targets.flatten(0, 1)).reshape(len(now), targets.size(1), -1)
        world_loss = model.world_loss(current_z, goal_z, stage_z)
    world_loss.backward()

    transition_total = torch.zeros((), device=device)
    before, after = transitions
    for src, dst in zip(before.split(microbatch), after.split(microbatch)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            with torch.no_grad():
                future_z = encode(teacher, dst)
            loss = model.transition_loss(encode(online, src), future_z) * (len(src) / len(before))
        loss.backward()
        transition_total += loss.detach()

    action_total = torch.zeros((), device=device)
    if actions is not None:
        for src, dst, truth in zip(obs.split(microbatch), goals.split(microbatch), labels.split(microbatch)):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                a, b = encode(online, torch.cat((src, dst))).chunk(2)
                loss = model.action_loss(a, b, truth.to(device)) * (len(src) / len(labels))
            loss.backward()
            action_total += loss.detach()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.model.update_target()
    return {"world": world_loss.detach(), "transition": transition_total, "action": action_total}
