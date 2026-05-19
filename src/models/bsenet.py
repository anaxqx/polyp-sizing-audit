"""
BseNet: Binary Size Estimation Network

Lightweight depth-only CNN for binary polyp size classification.
Input: Single-channel depth map (179x179).
Output: 2-class logits (Small vs Large).

Architecture:
  3 Conv blocks (Conv2d -> LeakyReLU -> MaxPool2d -> BatchNorm2d)
  Classification head: Dropout -> GAP -> LeakyReLU -> Linear

Spatial reduction: 179 -> 57 -> 18 -> 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union


class BseNet(nn.Module):
    """Binary Size Estimation Network for depth-only polyp classification."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 1ch, 179x179 -> 64ch, 57x57
        # Conv: (179 - 9) / 1 + 1 = 171 ; Pool: 171 / 3 = 57
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.BatchNorm2d(64),
        )

        # Block 2: 64ch, 57x57 -> 128ch, 18x18
        # Conv: (57 - 4) / 1 + 1 = 54 ; Pool: 54 / 3 = 18
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.BatchNorm2d(128),
        )

        # Block 3: 128ch, 18x18 -> 256ch, 5x5
        # Conv: (18 - 9) / 1 + 1 = 10 ; Pool: 10 / 2 = 5
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=9, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
        )

        # Classification head: Dropout -> GAP -> LeakyReLU -> Linear
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_activation = nn.LeakyReLU(negative_slope=0.3)
        self.fc = nn.Linear(256, num_classes)

        # Weight initialization (He / Kaiming)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.3
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.3
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 179, 179] depth map tensor
        Returns:
            logits: [B, num_classes] raw logits (no softmax)
        """
        x = self.block1(x)   # -> [B, 64, 57, 57]
        x = self.block2(x)   # -> [B, 128, 18, 18]
        x = self.block3(x)   # -> [B, 256, 5, 5]

        x = self.dropout(x)
        x = self.gap(x)      # -> [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # -> [B, 256]
        x = self.head_activation(x)
        x = self.fc(x)       # -> [B, num_classes]
        return x


def create_bsenet_from_config(config: dict) -> BseNet:
    """Factory function to create BseNet from a config dict."""
    model_cfg = config.get('model', {})
    return BseNet(
        num_classes=model_cfg.get('num_classes', 2),
        dropout_rate=model_cfg.get('dropout_rate', 0.2),
    )


@torch.no_grad()
def sequential_voting_inference(
    model: BseNet,
    frames: Union[torch.Tensor, List[torch.Tensor]],
    threshold: float = 0.5,
    device: str = 'cuda',
    batch_size: int = 64,
) -> Dict:
    """
    Run inference on a list of depth frames and return majority-vote prediction.

    Args:
        model: Trained BseNet model.
        frames: Tensor [N, 1, 179, 179] or list of [1, 179, 179] tensors.
        threshold: Probability threshold for class 1 (default 0.5).
        device: Device to run inference on.
        batch_size: Batch size for inference (memory management).

    Returns:
        dict with keys:
            prediction (int): Final class (0 or 1) from majority vote.
            confidence (float): Fraction of votes for the winning class.
            per_frame_probs (np.ndarray): [N, 2] softmax probabilities.
            per_frame_votes (np.ndarray): [N] binary votes.
            vote_counts (dict): {0: count, 1: count}.
    """
    model.eval()

    if isinstance(frames, list):
        frames = torch.stack(frames, dim=0)
    frames = frames.to(device)

    # Batched inference
    all_probs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()  # [N, 2]

    # Threshold on class-1 probability to get binary votes
    per_frame_votes = (all_probs[:, 1] >= threshold).astype(int)

    # Majority vote
    vote_counts = {0: int((per_frame_votes == 0).sum()),
                   1: int((per_frame_votes == 1).sum())}
    prediction = 1 if vote_counts[1] > vote_counts[0] else 0
    confidence = vote_counts[prediction] / len(per_frame_votes)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'per_frame_probs': all_probs,
        'per_frame_votes': per_frame_votes,
        'vote_counts': vote_counts,
    }


if __name__ == '__main__':
    # Verify architecture: instantiate model, check spatial dims, print summary
    model = BseNet(num_classes=2, dropout_rate=0.2)
    print("BseNet Architecture:")
    print(model)
    print()

    # Verify spatial dimensions with a dummy input
    x = torch.randn(1, 1, 179, 179)
    print(f"Input shape: {x.shape}")

    # Trace through blocks
    out1 = model.block1(x)
    print(f"After Block 1: {out1.shape}")  # expect [1, 64, 57, 57]

    out2 = model.block2(out1)
    print(f"After Block 2: {out2.shape}")  # expect [1, 128, 18, 18]

    out3 = model.block3(out2)
    print(f"After Block 3: {out3.shape}")  # expect [1, 256, 5, 5]

    logits = model(x)
    print(f"Output logits: {logits.shape}")  # expect [1, 2]
    print()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Initialize optimizer and scheduler per spec
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: StepLR(step_size=15, gamma=0.1)")
    print(f"  LR at epoch 0: {scheduler.get_last_lr()}")
