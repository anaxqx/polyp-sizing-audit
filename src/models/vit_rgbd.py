"""
Vision Transformer for RGB+D classification

Modified ViT-B/16 with 4-channel input (RGB + Depth)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple


class ViTRGBD(nn.Module):
    """
    Vision Transformer with 4-channel input (RGB + Depth)

    Based on ViT-B/16 from timm, modified for RGBD input
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_384',
                 num_classes: int = 3,
                 pretrained: bool = True,
                 depth_channel_init: str = 'mean',
                 input_channels: int = 4,
                 temporal_enabled: bool = False,
                 temporal_method: str = 'mean',
                 temporal_num_layers: int = 2,
                 temporal_num_heads: int = 4,
                 temporal_dropout: float = 0.1,
                 temporal_max_len: int = 16,
                 mask_head: Optional[Dict] = None):
        """
        Args:
            model_name: timm model name (e.g., 'vit_base_patch16_384')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            depth_channel_init: How to initialize depth channel weights
                - 'mean': Average of RGB weights
                - 'duplicate_red': Copy red channel weights
                - 'zeros': Initialize with zeros
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.depth_channel_init = depth_channel_init
        self.input_channels = int(input_channels)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_method = temporal_method
        self.temporal_max_len = int(temporal_max_len)
        self.mask_head_cfg = mask_head or {}

        # Load pretrained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # No head

        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim

        # Modify patch embedding for 4-channel input (RGB+D)
        if self.input_channels == 4:
            self._modify_patch_embedding()
        elif self.input_channels == 3:
            # Standard RGB input; no patch embedding change needed
            pass
        else:
            raise ValueError(f"Unsupported input_channels={self.input_channels} (expected 3 or 4)")

        # Create classification head
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Initialize head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Optional segmentation mask head (predicts 1-channel mask)
        self.mask_head = None
        self.mask_upsample = str(self.mask_head_cfg.get('upsample', 'bilinear'))
        self.mask_out_size = self.mask_head_cfg.get('out_size')
        if bool(self.mask_head_cfg.get('enabled', False)):
            hidden_dim = int(self.mask_head_cfg.get('hidden_dim', max(32, self.embed_dim // 2)))
            self.mask_head = nn.Sequential(
                nn.Conv2d(self.embed_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )

        # Optional temporal aggregator (operates on per-frame features)
        self.temporal_pos_embed = None
        self.temporal_encoder = None
        self.temporal_attn = None
        if self.temporal_enabled:
            method = self.temporal_method
            if method == 'transformer':
                layer = nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=int(temporal_num_heads),
                    dropout=float(temporal_dropout),
                    batch_first=True,
                )
                self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=int(temporal_num_layers))
                # Learnable positional embedding for temporal tokens
                self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.temporal_max_len, self.embed_dim))
                nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
            elif method == 'attn':
                # Simple attention pooling over time
                self.temporal_attn = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.Tanh(),
                    nn.Linear(self.embed_dim, 1),
                )
            elif method in {'mean', 'max'}:
                pass
            else:
                raise ValueError(f"Unknown temporal_method: {method}")

    def _forward_tokens(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward through ViT and return patch tokens (if available) and pooled features.

        Returns:
            patch_tokens: [B, N, D] or None if not available
            features: [B, D]
        """
        tokens = self.vit.forward_features(x)
        if tokens.dim() == 2:
            # Some timm variants return pooled features directly
            return None, tokens

        num_prefix = int(getattr(self.vit, 'num_prefix_tokens', 1))
        if num_prefix > 0:
            patch_tokens = tokens[:, num_prefix:, :]
            features = tokens[:, 0, :]
        else:
            patch_tokens = tokens
            features = tokens.mean(dim=1)

        # Respect global_pool if present
        global_pool = getattr(self.vit, 'global_pool', None)
        if global_pool in {'avg', 'mean'} and patch_tokens is not None:
            features = patch_tokens.mean(dim=1)

        return patch_tokens, features

    def _tokens_to_mask(self, patch_tokens: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Convert patch tokens to a spatial mask prediction.

        Args:
            patch_tokens: [B, N, D]
            image_hw: (H, W) of input image

        Returns:
            mask_logits: [B, 1, H, W]
        """
        if patch_tokens is None:
            raise ValueError("Patch tokens unavailable; cannot produce mask.")

        b, n, d = patch_tokens.shape
        grid_size = getattr(self.vit.patch_embed, 'grid_size', None)
        if grid_size is None:
            grid_h = grid_w = int(math.sqrt(n))
        else:
            grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
        if grid_h * grid_w != n:
            grid_h = grid_w = int(math.sqrt(n))
        feats = patch_tokens.transpose(1, 2).reshape(b, d, grid_h, grid_w)
        mask_logits = self.mask_head(feats)
        target_h, target_w = image_hw
        if self.mask_out_size is not None:
            target_h = target_w = int(self.mask_out_size)
        if mask_logits.shape[-2:] != (target_h, target_w):
            mask_logits = F.interpolate(mask_logits, size=(target_h, target_w), mode=self.mask_upsample, align_corners=False)
        return mask_logits

    def _modify_patch_embedding(self):
        """
        Modify patch embedding to accept 4 channels instead of 3

        Strategy:
        1. Get pretrained 3-channel conv weights
        2. Create new 4-channel conv
        3. Initialize RGB channels with pretrained weights
        4. Initialize depth channel (mean/duplicate/zeros)
        """
        # Get original patch embedding
        orig_patch_embed = self.vit.patch_embed.proj

        # Original conv parameters
        out_channels = orig_patch_embed.out_channels
        kernel_size = orig_patch_embed.kernel_size
        stride = orig_patch_embed.stride
        padding = orig_patch_embed.padding

        # Create new conv with 4 input channels
        new_patch_embed = nn.Conv2d(
            in_channels=4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Get pretrained RGB weights
        with torch.no_grad():
            pretrained_weight = orig_patch_embed.weight.clone()  # [out, 3, patch, patch]

            # Initialize RGB channels
            new_patch_embed.weight[:, :3, :, :] = pretrained_weight

            # Initialize depth channel
            if self.depth_channel_init == 'mean':
                # Average of RGB channels
                depth_init = pretrained_weight.mean(dim=1, keepdim=True)  # [out, 1, patch, patch]
            elif self.depth_channel_init == 'duplicate_red':
                # Copy red channel
                depth_init = pretrained_weight[:, 0:1, :, :]  # [out, 1, patch, patch]
            elif self.depth_channel_init == 'zeros':
                # Initialize with zeros
                depth_init = torch.zeros_like(pretrained_weight[:, 0:1, :, :])
            else:
                raise ValueError(f"Unknown depth_channel_init: {self.depth_channel_init}")

            new_patch_embed.weight[:, 3:4, :, :] = depth_init

            # Copy bias
            if orig_patch_embed.bias is not None:
                new_patch_embed.bias = orig_patch_embed.bias

        # Replace patch embedding
        self.vit.patch_embed.proj = new_patch_embed

        print(f"Modified patch embedding: 3 channels -> 4 channels (depth init: {self.depth_channel_init})")

    def _aggregate_temporal(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features.

        Args:
            feats: [B, T, D]
        Returns:
            [B, D]
        """
        method = self.temporal_method
        if method == 'mean':
            return feats.mean(dim=1)
        if method == 'max':
            return feats.max(dim=1).values
        if method == 'attn':
            weights = self.temporal_attn(feats)  # [B, T, 1]
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

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 4, H, W] (RGB + Depth)

        Returns:
            Logits [B, num_classes] (and optional mask logits)
        """
        # Support temporal input: [B, T, C, H, W]
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            assert c == self.input_channels, f"Expected {self.input_channels} channels, got {c}"
            x = x.reshape(b * t, c, h, w)
            patch_tokens, features = self._forward_tokens(x)
            features = features.reshape(b, t, -1)  # [B, T, D]
            if self.temporal_enabled:
                features = self._aggregate_temporal(features)
            else:
                features = features.mean(dim=1)
            mask_logits = None
            if self.mask_head is not None:
                if patch_tokens is None:
                    raise ValueError("Mask head enabled but patch tokens unavailable.")
                mask_logits = self._tokens_to_mask(patch_tokens, (h, w)).reshape(b, t, 1, h, w)
        else:
            # Check input shape
            assert x.shape[1] == self.input_channels, f"Expected {self.input_channels} channels, got {x.shape[1]}"
            patch_tokens, features = self._forward_tokens(x)
            mask_logits = None
            if self.mask_head is not None:
                if patch_tokens is None:
                    raise ValueError("Mask head enabled but patch tokens unavailable.")
                mask_logits = self._tokens_to_mask(patch_tokens, (x.shape[-2], x.shape[-1]))

        # Classification head
        logits = self.head(features)  # [B, num_classes]

        if self.mask_head is not None:
            return logits, mask_logits
        return logits

    def get_features(self, x):
        """
        Extract features without classification head

        Args:
            x: Input tensor [B, 4, H, W]

        Returns:
            Features [B, embed_dim]
        """
        return self.vit(x)


def create_vit_rgbd_from_config(config: Dict) -> ViTRGBD:
    """
    Create ViT RGBD model from config

    Args:
        config: Configuration dictionary

    Returns:
        ViT RGBD model
    """
    model_config = config['model']

    temporal_cfg = model_config.get('temporal', {}) or {}

    model = ViTRGBD(
        model_name=model_config.get('name', 'vit_base_patch16_384'),
        num_classes=model_config.get('num_classes', 3),
        pretrained=model_config.get('pretrained', True),
        depth_channel_init=model_config.get('depth_channel_init', 'mean'),
        input_channels=model_config.get('input_channels', 4),
        temporal_enabled=temporal_cfg.get('enabled', False),
        temporal_method=temporal_cfg.get('method', 'mean'),
        temporal_num_layers=temporal_cfg.get('num_layers', 2),
        temporal_num_heads=temporal_cfg.get('num_heads', 4),
        temporal_dropout=temporal_cfg.get('dropout', 0.1),
        temporal_max_len=temporal_cfg.get('max_len', temporal_cfg.get('clip_len', 16)),
        mask_head=model_config.get('mask_head'),
    )

    return model


# Example usage and testing
if __name__ == "__main__":
    # Quick self-test with default config values
    config = {
        "data": {"image_size": 224, "num_classes": 2},
        "model": {
            "architecture": "vit_rgbd",
            "name": "vit_base_patch16_224",
            "input_channels": 4,
            "num_classes": 2,
            "pretrained": False,
        },
    }

    print("Creating ViT RGBD model...")
    model = create_vit_rgbd_from_config(config)

    print(f"\nModel: {model.model_name}")
    print(f"Embedding dim: {model.embed_dim}")
    print(f"Number of classes: {model.num_classes}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nTesting forward pass...")
    batch_size = 2
    image_size = config['data']['image_size']
    dummy_input = torch.randn(batch_size, 4, image_size, image_size)
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")

    with torch.no_grad():
        features = model.get_features(dummy_input)
    print(f"Features shape: {features.shape}")

    print("\nModel creation and forward pass successful!")
