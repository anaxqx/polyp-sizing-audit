"""
Data augmentation transforms for RGB+D polyp classification

Key principle: Color jitter and noise only apply to RGB channels, NOT depth!
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from typing import Tuple, Optional


class RGBDTransform:
    """Base class for RGBD transforms that handles 4-channel input"""

    def __init__(self):
        pass

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transform to RGB and depth separately

        Args:
            rgb: RGB tensor (3, H, W)
            depth: Depth tensor (1, H, W)
            mask: Optional mask tensor (1, H, W)

        Returns:
            Transformed (rgb, depth) tuple or (rgb, depth, mask)
        """
        raise NotImplementedError


class RandomHorizontalFlip(RGBDTransform):
    """Random horizontal flip for both RGB and depth"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
            if mask is not None:
                mask = TF.hflip(mask)
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class RandomRotation(RGBDTransform):
    """Random rotation (small angle) for both RGB and depth"""

    def __init__(self, degrees=5):
        super().__init__()
        self.degrees = degrees

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = random.uniform(-self.degrees, self.degrees)
        rgb = TF.rotate(rgb, angle, interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.rotate(depth, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class RandomResizedCrop(RGBDTransform):
    """Random crop and resize for both RGB and depth"""

    def __init__(self, size, scale=(0.8, 1.0), ratio=(0.9, 1.1)):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get same crop parameters for both
        i, j, h, w = T.RandomResizedCrop.get_params(rgb, self.scale, self.ratio)

        rgb = TF.resized_crop(rgb, i, j, h, w, (self.size, self.size),
                              interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.resized_crop(depth, i, j, h, w, (self.size, self.size),
                                interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resized_crop(mask, i, j, h, w, (self.size, self.size),
                                   interpolation=TF.InterpolationMode.NEAREST)
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class ColorJitter(RGBDTransform):
    """Color jitter for RGB ONLY (not depth!)"""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        super().__init__()
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only apply to RGB
        rgb = self.color_jitter(rgb)
        # Depth unchanged
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class GaussianNoise(RGBDTransform):
    """Add Gaussian noise to RGB ONLY (not depth!)"""

    def __init__(self, std=0.02):
        super().__init__()
        self.std = std

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only add noise to RGB
        if self.std > 0:
            noise = torch.randn_like(rgb) * self.std
            rgb = rgb + noise
            rgb = torch.clamp(rgb, 0, 1)
        # Depth unchanged
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class GrayWorldColorBalance(RGBDTransform):
    """Simple gray-world white balance for RGB only."""

    def __init__(self, strength=1.0, max_scale=2.5):
        super().__init__()
        self.strength = float(strength)
        self.max_scale = float(max_scale)

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.strength <= 0:
            if mask is None:
                return rgb, depth
            return rgb, depth, mask

        # rgb shape: [3, H, W], value range expected in [0, 1]
        ch_mean = rgb.view(3, -1).mean(dim=1)  # [3]
        target = ch_mean.mean()
        scales = target / (ch_mean + 1e-6)
        scales = torch.clamp(scales, 1.0 / self.max_scale, self.max_scale).view(3, 1, 1)

        balanced = torch.clamp(rgb * scales, 0.0, 1.0)
        rgb = rgb * (1.0 - self.strength) + balanced * self.strength
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class Normalize(RGBDTransform):
    """Normalize RGB and depth with different stats"""

    def __init__(self,
                 rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225),
                 depth_mean=0.0,
                 depth_std=1.0,
                 depth_norm_method='zscore'):
        """
        Args:
            rgb_mean: ImageNet mean for RGB
            rgb_std: ImageNet std for RGB
            depth_mean: Mean for depth normalization
            depth_std: Std for depth normalization
            depth_norm_method: 'zscore', 'minmax', or 'none'
        """
        super().__init__()
        self.rgb_mean = torch.tensor(rgb_mean).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std).view(3, 1, 1)
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.depth_norm_method = depth_norm_method

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize RGB with ImageNet stats
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        # Normalize depth
        if self.depth_norm_method == 'zscore':
            depth = (depth - self.depth_mean) / (self.depth_std + 1e-8)
        elif self.depth_norm_method == 'minmax':
            # Assume depth already in [0, 1], map to [-1, 1]
            depth = (depth * 2) - 1
        # elif 'none', keep as is

        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class ToTensor(RGBDTransform):
    """Convert PIL Image to Tensor"""

    def __init__(self):
        super().__init__()
        self.to_tensor = T.ToTensor()

    def __call__(self, rgb, depth, mask: Optional[torch.Tensor] = None):
        """
        Args:
            rgb: PIL Image or numpy array
            depth: PIL Image or numpy array
        """
        if not isinstance(rgb, torch.Tensor):
            rgb = self.to_tensor(rgb)
        if not isinstance(depth, torch.Tensor):
            depth = self.to_tensor(depth)

        if mask is not None and not isinstance(mask, torch.Tensor):
            # Handle PIL or numpy mask
            mask = self.to_tensor(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                mask = mask[:1]

        # Ensure depth is single channel
        if depth.ndim == 3 and depth.shape[0] == 3:
            depth = depth.mean(dim=0, keepdim=True)
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0)

        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class Resize(RGBDTransform):
    """Resize both RGB and depth to target size"""

    def __init__(self, size):
        super().__init__()
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = TF.resize(rgb, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.resize(depth, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        if mask is None:
            return rgb, depth
        return rgb, depth, mask


class Compose:
    """Compose multiple RGBD transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, depth, mask: Optional[torch.Tensor] = None):
        if mask is None:
            for t in self.transforms:
                rgb, depth = t(rgb, depth)
            return rgb, depth

        for t in self.transforms:
            rgb, depth, mask = t(rgb, depth, mask)
        return rgb, depth, mask


def get_train_transforms(config):
    """
    Create training transforms from config

    Args:
        config: Configuration dict with augmentation settings

    Returns:
        Composed transform
    """
    aug_config = config.get('augmentation', {})

    transforms = []

    # Convert to tensor first
    transforms.append(ToTensor())

    # Resize
    image_size = config.get('data', {}).get('image_size', 384)
    transforms.append(Resize(image_size))

    # Optional color balancing to reduce device/domain color cast.
    if aug_config.get('color_balance', False):
        transforms.append(GrayWorldColorBalance(
            strength=aug_config.get('color_balance_strength', 1.0),
            max_scale=aug_config.get('color_balance_max_scale', 2.5),
        ))

    # Random horizontal flip
    if aug_config.get('horizontal_flip', True):
        transforms.append(RandomHorizontalFlip(p=aug_config.get('flip_prob', 0.5)))

    # Random rotation
    if aug_config.get('rotation', True):
        transforms.append(RandomRotation(degrees=aug_config.get('rotation_degrees', 5)))

    # Random resized crop
    if aug_config.get('random_crop_resize', True):
        transforms.append(RandomResizedCrop(
            size=image_size,
            scale=aug_config.get('crop_scale', [0.8, 1.0]),
            ratio=aug_config.get('crop_ratio', [0.9, 1.1])
        ))

    # Color jitter (RGB only!)
    if aug_config.get('color_jitter', True):
        transforms.append(ColorJitter(
            brightness=aug_config.get('brightness', 0.2),
            contrast=aug_config.get('contrast', 0.2),
            saturation=aug_config.get('saturation', 0.2),
            hue=aug_config.get('hue', 0.1)
        ))

    # Gaussian noise (RGB only!)
    if aug_config.get('gaussian_noise', True):
        transforms.append(GaussianNoise(std=aug_config.get('noise_std', 0.02)))

    # Normalization (will be done in dataset after depth rescaling if needed)
    # So we don't add it here

    return Compose(transforms)


def get_val_transforms(config):
    """
    Create validation/test transforms from config

    Args:
        config: Configuration dict

    Returns:
        Composed transform
    """
    image_size = config.get('data', {}).get('image_size', 384)

    transforms = []

    # Convert to tensor
    transforms.append(ToTensor())

    # Resize
    transforms.append(Resize(image_size))

    # Keep color balancing consistent across train/val/test if enabled.
    aug_config = config.get('augmentation', {})
    if aug_config.get('color_balance', False):
        transforms.append(GrayWorldColorBalance(
            strength=aug_config.get('color_balance_strength', 1.0),
            max_scale=aug_config.get('color_balance_max_scale', 2.5),
        ))

    # No augmentations for validation
    # Normalization will be done in dataset

    return Compose(transforms)


def compute_depth_normalization_stats(depth_maps, method='zscore'):
    """
    Compute normalization statistics from a list of depth maps

    Args:
        depth_maps: List of depth tensors
        method: 'zscore' or 'minmax'

    Returns:
        (mean, std) tuple
    """
    if method == 'zscore':
        all_values = torch.cat([d.flatten() for d in depth_maps])
        mean = all_values.mean().item()
        std = all_values.std().item()
        return mean, std
    elif method == 'minmax':
        # Depth already in [0, 1], no stats needed
        return 0.0, 1.0
    else:
        return 0.0, 1.0


# Example usage
if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "scenario1_baseline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    # Test with dummy data
    rgb = torch.rand(3, 224, 224)
    depth = torch.rand(1, 224, 224)

    print("Testing train transforms:")
    rgb_aug, depth_aug = train_transform(rgb, depth)
    print(f"  RGB shape: {rgb_aug.shape}")
    print(f"  Depth shape: {depth_aug.shape}")
    print(f"  RGB range: [{rgb_aug.min():.3f}, {rgb_aug.max():.3f}]")
    print(f"  Depth range: [{depth_aug.min():.3f}, {depth_aug.max():.3f}]")

    print("\nTesting val transforms:")
    rgb_val, depth_val = val_transform(rgb, depth)
    print(f"  RGB shape: {rgb_val.shape}")
    print(f"  Depth shape: {depth_val.shape}")
    print(f"  RGB range: [{rgb_val.min():.3f}, {rgb_val.max():.3f}]")
    print(f"  Depth range: [{depth_val.min():.3f}, {depth_val.max():.3f}]")
