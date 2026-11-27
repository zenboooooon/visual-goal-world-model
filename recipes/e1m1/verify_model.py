"""Compare the recipe's initialization or original 25k checkpoint against recorded measurements."""

import argparse
import json
from pathlib import Path

import torch

from dataset import MixedDataset
from experiment import MANIFEST, Policy, load_policy, other_loss, validation_windows, world
from sampling import Cycle


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-25000", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    data = MixedDataset(MANIFEST, 192)
    if args.original_25000:
        model = load_policy(args.original_25000, device)
        losses = [world(data, model, b, device, False) for b in validation_windows(data)]
        dev = data.rows(data.dev, labels=True)
        total = torch.zeros((), device=device)
        for i in range(0, len(dev), 16):
            rows = dev[i : i + 16]
            total += other_loss(data, model, rows, device, "action", False) * len(rows)
        values = [torch.stack(losses).mean().item(), total.item() / len(dev)]
        reference = [0.66967, 0.9028775772958432]
    else:
        model = Policy().to(device)
        pools = {h: Cycle(data.rows(data.train, 35 * h), 17, h) for h in range(1, 11)}
        batch = [(i, t, h) for h, pool in pools.items() for i, t in pool.draw(2)]
        transitions = Cycle(data.rows(data.train, 8), 17, 101).draw(20)
        actions = data.sort_action_rows(Cycle(data.rows(data.train, labels=True), 17, 102).draw(128))
        action = sum(
            other_loss(data, model, actions[i : i + 32], device, "action", True) * 0.25
            for i in range(0, 128, 32)
        )
        values = [
            world(data, model, batch, device, True).item(),
            other_loss(data, model, transitions, device, "transition", True).item(),
            action.item(),
        ]
        reference = [0.8934366106987, 0.42705535888671875, 0.7020484209060669]
    print(json.dumps(dict(measured=values, reference=reference)))
    torch.testing.assert_close(torch.tensor(values), torch.tensor(reference), rtol=0, atol=1e-4)


if __name__ == "__main__":
    main()
