"""MLP probe for handcrafted polyp-size features."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PhotometryMLP(nn.Module):
    """Two-layer MLP classifier for precomputed diagnostic features."""

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dims: List[int] | None = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.net(features)


def create_photometry_mlp_from_config(config: dict) -> PhotometryMLP:
    model_cfg = config.get("model", {})
    return PhotometryMLP(
        input_dim=model_cfg.get("photometry_input_dim", 51),
        hidden_dims=model_cfg.get("photometry_hidden", [128, 64]),
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("dropout", 0.3),
    )
