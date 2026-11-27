"""Spatial ViT and temporal encoder for a four-frame image history."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 96
    channels: int = 3
    patch_size: int = 8
    width: int = 128
    spatial_layers: int = 4
    temporal_layers: int = 2
    heads: int = 4
    mlp_ratio: int = 4
    history_images: int = 4
    history_actions: int = 0


class SpatialViT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image size must be divisible by patch size")
        self.config = config
        patch_count = (config.image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            config.channels, config.width, config.patch_size, stride=config.patch_size
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, config.width))
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * config.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.spatial_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.width)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        expected = (self.config.channels, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"expected (N, {expected}), got {tuple(images.shape)}")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position
        return self.output_norm(self.blocks(tokens)[:, 0])


class ContextEncoder(nn.Module):
    """Exact NC ObservationEncoder with a reusable spatial-feature entry point."""

    def __init__(self, config, spatial_type):
        super().__init__()
        if config.history_images != 4 or config.history_actions != 0:
            raise ValueError("NC context requires four images and zero past actions")
        self.config = config
        self.spatial = spatial_type(config)
        self.position = nn.Parameter(torch.zeros(1, 4, config.width))
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * config.mlp_ratio,
            dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(
            layer, config.temporal_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.width)
        self.register_buffer("causal_mask", torch.triu(torch.ones(4, 4, dtype=torch.bool), 1))
        nn.init.normal_(self.position, std=0.02)

    def from_spatial(self, features: Tensor, valid: Tensor) -> Tensor:
        batch = features.shape[0]
        if tuple(features.shape) != (batch, 4, self.config.width):
            raise ValueError("expected four spatial features per history at the configured width")
        if tuple(valid.shape) != (batch, 4) or valid.dtype != torch.bool:
            raise ValueError("expected boolean validity mask (B, 4)")
        if not bool(valid[:, -1].all()):
            raise ValueError("current frame must be valid")
        tokens = features + self.position
        mask = self.causal_mask.unsqueeze(0) | ~valid[:, None, :]
        diagonal = torch.arange(4, device=features.device)
        mask[:, diagonal, diagonal] = False
        temporal = self.temporal(tokens, mask=mask.repeat_interleave(self.config.heads, 0))
        return self.output_norm(temporal[:, -1])


class HistoryEncoder(nn.Module):
    """Encode four aligned observations in frame-token or patch-token mode."""

    def __init__(self, config, temporal_mode):
        super().__init__()
        if temporal_mode not in ("pooled", "patch"):
            raise ValueError("temporal_mode must be pooled or patch")
        self.encoder = ContextEncoder(config, SpatialViT)
        self.temporal_mode = temporal_mode

    def spatial(self, images):
        if self.temporal_mode == "pooled":
            return self.encoder.spatial(images)
        spatial = self.encoder.spatial
        patches = spatial.patch_embed(images).flatten(2).transpose(1, 2)
        cls = spatial.cls.expand(len(images), -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        tokens = tokens + spatial.position.to(tokens.dtype)
        return spatial.output_norm(spatial.blocks(tokens)[:, 1:])

    def from_spatial(self, features, valid):
        if self.temporal_mode == "pooled":
            return self.encoder.from_spatial(features, valid)
        tokens = features + self.encoder.position.to(features.dtype)[:, :, None]
        past = tokens[:, :-1].flatten(1, 2)
        current = tokens[:, -1]
        padding = None
        if not valid.all():
            padding = (~valid).repeat_interleave(features.shape[2], dim=1)
        for layer in self.encoder.temporal.layers:
            memory = torch.cat((past, current), dim=1)
            memory = layer.norm1(memory)
            query = memory[:, -current.size(1):]
            delta = layer.self_attn(query, memory, memory, need_weights=False,
                                    key_padding_mask=padding)[0]
            current = current + layer.dropout1(delta)
            hidden = layer.linear1(layer.norm2(current))
            current = current + layer.dropout2(layer.linear2(layer.dropout(layer.activation(hidden))))
        return self.encoder.output_norm(current.mean(dim=1))

    def forward(self, images, valid=None):
        cfg = self.encoder.config
        if images.ndim != 5 or images.shape[1:] != (4,cfg.channels,cfg.image_size,cfg.image_size):
            raise ValueError("Expected image histories of shape [batch,4,channels,size,size]")
        if valid is None:
            valid = torch.ones(images.shape[:2], device=images.device, dtype=torch.bool)
        if valid.shape != images.shape[:2] or valid.dtype != torch.bool or not valid[:, -1].all():
            raise ValueError("Expected a boolean history mask with a valid current frame")
        frames = self.spatial(images.flatten(0,1))
        return self.from_spatial(frames.unflatten(0,images.shape[:2]), valid)
