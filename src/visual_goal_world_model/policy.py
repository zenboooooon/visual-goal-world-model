"""Goal-conditioned reverse world model for partially labeled videos."""
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import HistoryEncoder, ModelConfig
from .heads import LatentActionHead, ReverseWM, context_regularization


@dataclass(frozen=True)
class Config:
    temporal_mode: str = "pooled"
    image_size: int = 192
    channels: int = 3
    width: int = 512
    wm_hidden: int = 512
    latent_action_dim: int = 16
    action_dim: int = 8
    patch_size: int = 8
    heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 2


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision = ModelConfig(image_size=config.image_size, channels=config.channels,
                             patch_size=config.patch_size, width=config.width, heads=config.heads,
                             spatial_layers=config.spatial_layers, temporal_layers=config.temporal_layers)
        self.encoder = HistoryEncoder(vision, config.temporal_mode)
        self.target_encoder = deepcopy(self.encoder).eval().requires_grad_(False)
        self.wm = ReverseWM(config.width, config.wm_hidden)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(momentum).add_(online, alpha=1-momentum)


class WorldModel(nn.Module):
    """Inputs are float image histories [B,4,C,H,W] in [0,1]. Outputs are logits."""

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.model = Backbone(config)
        self.head = LatentActionHead(config.width, config.latent_action_dim, config.action_dim)
        self.action_names = None

    def logits(self, current, goal):
        stages = self.model.wm.unroll(current, goal)
        action = self.head.encode_action(current, stages[:, -1])
        return self.head.decode_action(action)

    def forward(self, current, goal, current_valid=None, goal_valid=None):
        return self.logits(self.model.encoder(current, current_valid), self.model.encoder(goal, goal_valid))

    def world_loss(self, current, goal, stage_targets):
        """Inputs are online [B,D] and EMA [B,9,D] embeddings ordered far to near."""
        prediction = self.model.wm.unroll(current, goal)
        if stage_targets.shape != prediction.shape:
            raise ValueError("Expected one target embedding per prediction stage")
        std, cov = context_regularization(torch.cat((current, goal)))
        return F.smooth_l1_loss(prediction.float(), stage_targets.detach().float()) + std + 0.01*cov

    def transition_loss(self, current, future):
        future = future.detach()
        action = self.head.encode_action(current, future)
        return F.smooth_l1_loss(self.head.predict_future(current, action).float(), future.float())

    def action_loss(self, current, goal, targets):
        """Pass only labeled examples to this objective, using multi-label binary targets."""
        return F.binary_cross_entropy_with_logits(self.logits(current, goal).float(), targets.float())

    def save(self, path):
        torch.save(dict(format="visual-goal-world-model-v1", config=asdict(self.config),
                        action_names=self.action_names, policy=self.state_dict()), path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Restore a model saved with WorldModel.save."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != "visual-goal-world-model-v1":
            raise ValueError("For external weights, construct a model with the matching Config and use load_state_dict")
        model = cls(Config(**saved["config"]))
        model.load_state_dict(saved["policy"], strict=True)
        model.action_names = saved["action_names"]
        return model.to(device).eval()
