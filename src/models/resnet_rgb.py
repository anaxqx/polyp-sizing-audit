"""ResNet RGB probe for polyp size classification."""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Dict


def _create_resnet_backbone(model_name: str, pretrained: bool) -> nn.Module:
    """Build a torchvision ResNet backbone with version-safe pretrained loading."""
    model_name = str(model_name).lower()
    builders = {
        'resnet18': (tv_models.resnet18, getattr(tv_models, 'ResNet18_Weights', None)),
        'resnet34': (tv_models.resnet34, getattr(tv_models, 'ResNet34_Weights', None)),
        'resnet50': (tv_models.resnet50, getattr(tv_models, 'ResNet50_Weights', None)),
        'resnet101': (tv_models.resnet101, getattr(tv_models, 'ResNet101_Weights', None)),
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported ResNet model '{model_name}'. Use one of: {sorted(builders.keys())}")

    builder, weights_enum = builders[model_name]

    if pretrained:
        if weights_enum is not None:
            try:
                return builder(weights=weights_enum.DEFAULT)
            except TypeError:
                pass
        try:
            return builder(weights='DEFAULT')
        except Exception:
            return builder(pretrained=True)

    try:
        return builder(weights=None)
    except TypeError:
        return builder(pretrained=False)


class ResNetRGB(nn.Module):
    """ResNet RGB classifier with optional temporal aggregation."""

    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 2,
        input_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.0,
        temporal_enabled: bool = False,
        temporal_method: str = 'mean',
        temporal_num_layers: int = 2,
        temporal_num_heads: int = 4,
        temporal_dropout: float = 0.1,
        temporal_max_len: int = 16,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_method = str(temporal_method)
        self.temporal_max_len = int(temporal_max_len)

        backbone = _create_resnet_backbone(model_name, pretrained=pretrained)
        
        # Adjust first layer for non-RGB input
        if self.input_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                self.input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize new weights: copy RGB weights, average for extra channels
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if self.input_channels > 3:
                    # Initialize extra channels with average of RGB weights
                    avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:] = avg_weight.repeat(1, self.input_channels - 3, 1, 1)
            backbone.conv1 = new_conv

        self.feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.head = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.temporal_pos_embed = None
        self.temporal_encoder = None
        self.temporal_attn = None
        if self.temporal_enabled:
            if self.temporal_method == 'transformer':
                layer = nn.TransformerEncoderLayer(
                    d_model=self.feature_dim,
                    nhead=int(temporal_num_heads),
                    dropout=float(temporal_dropout),
                    batch_first=True,
                )
                self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=int(temporal_num_layers))
                self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.temporal_max_len, self.feature_dim))
                nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
            elif self.temporal_method == 'attn':
                self.temporal_attn = nn.Sequential(
                    nn.Linear(self.feature_dim, self.feature_dim),
                    nn.Tanh(),
                    nn.Linear(self.feature_dim, 1),
                )
            elif self.temporal_method in {'mean', 'max'}:
                pass
            else:
                raise ValueError(f"Unknown temporal_method: {self.temporal_method}")

    def _aggregate_temporal(self, feats: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal features [B, T, D] -> [B, D]."""
        method = self.temporal_method
        if method == 'mean':
            return feats.mean(dim=1)
        if method == 'max':
            return feats.max(dim=1).values
        if method == 'attn':
            weights = self.temporal_attn(feats)
            weights = torch.softmax(weights, dim=1)
            return (feats * weights).sum(dim=1)
        if method == 'transformer':
            if self.temporal_pos_embed is not None:
                if feats.size(1) > self.temporal_pos_embed.size(1):
                    raise ValueError(
                        f"Temporal length {feats.size(1)} exceeds max_len {self.temporal_pos_embed.size(1)}"
                    )
                feats = feats + self.temporal_pos_embed[:, :feats.size(1), :]
            feats = self.temporal_encoder(feats)
            return feats.mean(dim=1)
        raise ValueError(f"Unknown temporal_method: {method}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled appearance features from RGB frames."""
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            if c != self.input_channels:
                raise ValueError(f"Expected input channels {self.input_channels}, got {c}")
            feats = self.backbone(x.reshape(b * t, c, h, w))
            feats = feats.reshape(b, t, -1)
            if self.temporal_enabled:
                feats = self._aggregate_temporal(feats)
            else:
                feats = feats.mean(dim=1)
            return feats

        if x.dim() != 4:
            raise ValueError(f"Expected input [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected input channels {self.input_channels}, got {x.shape[1]}")
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.head(self.dropout(feats))

def create_resnet_rgb_from_config(config: Dict) -> ResNetRGB:
    """Create ResNet RGB classifier from config."""
    mc = config['model']
    temporal_cfg = mc.get('temporal', {}) or {}
    return ResNetRGB(
        model_name=mc.get('name', mc.get('backbone', 'resnet18')),
        num_classes=mc.get('num_classes', 2),
        input_channels=mc.get('input_channels', 3),
        pretrained=mc.get('pretrained', True),
        dropout=mc.get('dropout', 0.0),
        temporal_enabled=temporal_cfg.get('enabled', False),
        temporal_method=temporal_cfg.get('method', 'mean'),
        temporal_num_layers=temporal_cfg.get('num_layers', 2),
        temporal_num_heads=temporal_cfg.get('num_heads', 4),
        temporal_dropout=temporal_cfg.get('dropout', 0.1),
        temporal_max_len=temporal_cfg.get('max_len', temporal_cfg.get('clip_len', 16)),
    )
