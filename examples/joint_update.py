"""Run a single joint-update example on generated video frames."""
import argparse

import torch

from visual_goal_world_model import Config, WorldModel
from visual_goal_world_model.training import FrameBank, joint_step, make_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pooled", "patch"], default="pooled")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = WorldModel(Config(temporal_mode=args.mode, image_size=32, width=32,
                              wm_hidden=48, action_dim=3)).to(args.device)
    optimizer = make_optimizer(model)
    # Each immutable frame ID belongs to one episode. This example uses one episode.
    bank = FrameBank(torch.randint(256, (72, 3, 32, 32), dtype=torch.uint8))
    def histories(t):
        return (t[..., None] + torch.arange(-3, 1)).clamp_min(-1)

    t = torch.tensor([0, 8, 16, 24])
    goal = t + 36
    stage_offsets = torch.tensor([32, 28, 24, 20, 16, 12, 8, 5, 2])
    # Only these two current histories have action labels. Both share a reference goal.
    labeled = torch.tensor([0, 16])
    losses = joint_step(
        model, optimizer, bank,
        world=(histories(t), histories(goal), histories(t[:, None] + stage_offsets)),
        transitions=(histories(t), histories(t + 2)),
        actions=(histories(labeled), histories(torch.tensor([56, 56])), torch.tensor([[1,0,0], [0,1,1]])),
        fast=not args.baseline, checkpoint_patches=args.mode == "patch",
    )
    print({name: round(loss.item(), 6) for name, loss in losses.items()})


if __name__ == "__main__":
    main()
