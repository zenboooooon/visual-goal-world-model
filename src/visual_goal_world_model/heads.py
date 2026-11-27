import torch
from torch import nn
from torch.nn import functional as F

SCHEDULE = (89, 78, 67, 56, 45, 34, 23, 12, 1)


class ReverseWM(nn.Module):
    def __init__(self, width, hidden=128):
        super().__init__()
        self.width, self.hidden = width, hidden
        self.schedule = SCHEDULE
        self.fractions = tuple(p / previous for p, previous in zip(SCHEDULE, (100, *SCHEDULE[:-1])))
        self.midpoint = nn.Sequential(nn.Linear(width * 3, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, width))

    def predict_stage(self, current, endpoint, stage_index):
        features = torch.cat((current, endpoint, endpoint - current), dim=-1)
        return torch.lerp(current, endpoint, self.fractions[stage_index]) + self.midpoint(features)

    def unroll(self, current, goal):
        endpoint, stages = goal, []
        for stage in range(len(self.schedule)):
            endpoint = self.predict_stage(current, endpoint, stage)
            stages.append(endpoint)
        return torch.stack(stages, dim=1)


def context_regularization(features, std_target=1.):
    with torch.autocast(features.device.type, enabled=False):
        x = features.float()
        if x.ndim != 2 or len(x) < 2 or x.shape[1] < 2:
            raise ValueError("Expected at least two multidimensional latent states")
        d = x.shape[1]
        centered = x - x.mean(0)
        covariance = centered.T @ centered / (len(x) - 1)
        variance_loss = F.relu(std_target - torch.sqrt(covariance.diagonal() + 1e-4)).mean()
        off = covariance - torch.diag_embed(covariance.diagonal())
        # Mean over ordered dimension pairs, calibrated to the existing d=256 loss.
        covariance_loss = off.square().sum() * (255. / (d * (d - 1)))
    return variance_loss, covariance_loss


class LatentActionHead(nn.Module):
    """Encode a transition, predict its future latent, and decode action logits."""

    def __init__(self, width, latent_width, action_dim):
        super().__init__()
        if latent_width < 1:
            raise ValueError("Latent action width must be positive")
        self.encoder = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(),
                                     nn.Linear(256, latent_width), nn.Tanh())
        self.transition = nn.Sequential(nn.Linear(width + latent_width, 256), nn.GELU(),
                                        nn.Linear(256, width))
        self.decoder = nn.Sequential(nn.Linear(latent_width, 256), nn.GELU(), nn.Linear(256, action_dim))

    def encode_action(self, current, future):
        return self.encoder(torch.cat((current, future), dim=-1))

    def predict_future(self, current, action):
        return self.transition(torch.cat((current, action), dim=-1))

    def decode_action(self, action):
        return self.decoder(action)

    def forward(self, features):
        return self.decode_action(self.encoder(features))
