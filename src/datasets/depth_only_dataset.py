"""
Lightweight depth-only dataset for BseNet polyp size classification.

Loads single-channel PPSNet depth maps and resizes to target resolution (179x179).
Supports video-stratified splitting and precomputed split files.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import random
import os
import time
import hashlib
from PIL import Image


class DepthOnlyPolypDataset(Dataset):
    """Dataset that loads only depth maps for BseNet training."""

    def __init__(
        self,
        csv_path: str,
        depth_dir: str,
        split: str = 'train',
        image_size: int = 179,
        precomputed_split_file: str = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_strategy: str = 'video_stratified',
        augment: bool = False,
        flip_prob: float = 0.5,
        max_samples: int = None,
        sun_path: str = None,
        depth_source: str = 'ppsnet',
        depth_norm_method: str = 'minmax',
        scale_factors_file: str = None,
        scale_factor_col: str = 'scale_factor',
        real_colon_segmentation_dir: str = None,
        sun_segmentation_dir: str = None,
        prefer_predicted_mask_bbox: bool = False,
        mask_bbox_scale_factor: float = 1.0,
    ):
        """
        Args:
            csv_path: Path to labels CSV (must have: image_name, video_id, label, depth_idx).
            depth_dir: Directory containing {depth_idx:06d}.pt files.
            split: 'train', 'val', or 'test'.
            image_size: Target spatial resolution (default 179).
            precomputed_split_file: Optional CSV with (image_name, split) columns.
            split_seed: Random seed for split generation.
            train_ratio: Train split ratio (default 0.7).
            val_ratio: Val split ratio (default 0.15).
            split_strategy: 'video_stratified' or 'frame_stratified'.
            augment: Whether to apply augmentations (flip).
            flip_prob: Horizontal flip probability.
            max_samples: Optional cap on number of samples (for smoke tests).
            depth_source: 'ppsnet' or 'metriccol'.
        """
        self.depth_dir = Path(depth_dir)
        self.csv_path = Path(csv_path)
        self.sun_path = Path(sun_path) if sun_path else None
        self.image_size = image_size
        self.augment = augment
        self.flip_prob = flip_prob
        self.depth_source = str(depth_source).lower()
        self.depth_norm_method = str(depth_norm_method).lower()
        self.scale_factors = None
        self.scale_factor_col = str(scale_factor_col)
        self.real_colon_seg_dir = Path(real_colon_segmentation_dir) if real_colon_segmentation_dir else None
        self.sun_seg_dir = Path(sun_segmentation_dir) if sun_segmentation_dir else None
        self.prefer_predicted_mask_bbox = bool(prefer_predicted_mask_bbox)
        self.mask_bbox_scale_factor = float(mask_bbox_scale_factor)

        # Load labels CSV
        df = pd.read_csv(csv_path)
        required_cols = {'image_name', 'video_id', 'label', 'depth_idx',
                         'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                         'image_width', 'image_height'}
        # 'dataset' is optional for backward compatibility but recommended for RC+SUN
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        self.has_depth_path_col = 'depth_path' in df.columns

        self.has_dataset_col = 'dataset' in df.columns

        # Optional per-frame depth scaling (e.g., oracle scales)
        if scale_factors_file:
            sf_df = pd.read_csv(scale_factors_file)
            if self.scale_factor_col not in sf_df.columns:
                raise ValueError(
                    f"Scale factors file missing column '{self.scale_factor_col}': {scale_factors_file}"
                )
            if 'image_name' not in sf_df.columns:
                raise ValueError(
                    f"Scale factors file missing 'image_name' column: {scale_factors_file}"
                )

            if 'dataset' in sf_df.columns:
                self.scale_factors = {
                    (str(r['dataset']), str(r['image_name'])): float(r[self.scale_factor_col])
                    for _, r in sf_df.iterrows()
                    if pd.notna(r.get(self.scale_factor_col))
                }
            else:
                self.scale_factors = {
                    str(r['image_name']): float(r[self.scale_factor_col])
                    for _, r in sf_df.iterrows()
                    if pd.notna(r.get(self.scale_factor_col))
                }

        df_all_for_cache = df.copy()

        # Assign splits
        if precomputed_split_file is not None:
            split_df = pd.read_csv(precomputed_split_file)
            df = self._assign_precomputed_splits(df, split_df)
        else:
            df = self._make_split(df, split_strategy, split_seed, train_ratio, val_ratio)

        # Filter to requested split
        df = df[df['split'] == split].reset_index(drop=True)

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=split_seed).reset_index(drop=True)

        # Optionally replace GT bbox with bbox inferred from predicted segmentation masks.
        # Do this after split filtering to avoid unnecessary work on non-used rows.
        if self.prefer_predicted_mask_bbox:
            df = self._apply_predicted_bbox_overrides(df, cache_source_df=df_all_for_cache)

        self.samples = df
        self.labels = df['label'].values

        # Compute class statistics
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
        total = counts.sum()
        # Inverse frequency weights for CrossEntropyLoss
        self.class_weights = torch.tensor(
            [total / (len(unique_labels) * c) for c in counts], dtype=torch.float32
        )

    def _find_mask_path(self, dataset: str, image_name: str) -> Optional[Path]:
        if dataset == 'sun':
            root = self.sun_seg_dir
        elif dataset == 'real_colon':
            root = self.real_colon_seg_dir
        else:
            root = self.real_colon_seg_dir
        if root is None:
            return None
        stem = Path(image_name).stem
        image_path = Path(image_name)
        candidates = [
            root / f"{stem}_binary.png",
            root / f"{stem}_mask.png",
            root / f"{stem}.png",
        ]
        # Keep relative-path support (real-colon) but avoid absolute SUN jpg paths.
        if not image_path.is_absolute():
            candidates.insert(0, root / image_name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _bbox_from_mask(mask_arr: np.ndarray) -> Optional[tuple]:
        bin_mask = mask_arr > 0
        if not np.any(bin_mask):
            return None
        ys, xs = np.where(bin_mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _scale_bbox_xyxy(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float,
        height: float,
        factor: float,
    ):
        if (not np.isfinite(factor)) or factor <= 0 or abs(float(factor) - 1.0) < 1e-8:
            return x1, y1, x2, y2
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(1e-6, x2 - x1) * float(factor)
        bh = max(1e-6, y2 - y1) * float(factor)
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        return x1, y1, x2, y2

    def _pred_bbox_cache_path(self) -> Path:
        key = "|".join([
            str(self.csv_path.resolve()),
            str(self.real_colon_seg_dir.resolve()) if self.real_colon_seg_dir else "",
            str(self.sun_seg_dir.resolve()) if self.sun_seg_dir else "",
        ])
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        cache_dir = self.csv_path.parent
        return cache_dir / f".pred_bbox_cache_{self.csv_path.stem}_{digest}.csv"

    def _assign_precomputed_splits(self, df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
        """Assign split labels from split_df with SUN path-remap compatibility."""
        if 'split' not in split_df.columns or 'image_name' not in split_df.columns:
            raise ValueError("Precomputed split file must have 'image_name' and 'split' columns")

        # Legacy exact merge path (single-dataset or no dataset column).
        if not ('dataset' in split_df.columns and self.has_dataset_col):
            return df.merge(split_df[['image_name', 'split']], on='image_name', how='inner')

        split_map = {
            (str(r['dataset']), str(r['image_name'])): str(r['split'])
            for _, r in split_df[['dataset', 'image_name', 'split']].iterrows()
        }

        # Some split CSVs were exported before SUN path normalization. Try basename, then
        # guarded row-order fallback for SUN to preserve original split semantics.
        sun_basename_to_split = {}
        sun_ambiguous = set()
        for (dataset_name, image_name), split_name in split_map.items():
            if dataset_name != 'sun':
                continue
            base = Path(image_name).name
            prev = sun_basename_to_split.get(base)
            if prev is None:
                sun_basename_to_split[base] = split_name
            elif prev != split_name:
                sun_ambiguous.add(base)
        for base in sun_ambiguous:
            sun_basename_to_split.pop(base, None)

        sun_split_sequence = split_df.loc[split_df['dataset'].astype(str) == 'sun', 'split'].astype(str).tolist()
        sun_total = int((df['dataset'].astype(str) == 'sun').sum())
        can_use_sun_row_fallback = len(sun_split_sequence) == sun_total and sun_total > 0

        assigned = []
        missing = []
        recovered_sun_by_basename = 0
        recovered_sun_by_row = 0
        sun_row_order_conflicts = 0
        sun_pos = 0

        for row in df.itertuples(index=False):
            dataset = str(getattr(row, 'dataset'))
            image_name = str(getattr(row, 'image_name'))
            split = split_map.get((dataset, image_name))
            if dataset == 'sun':
                if split is None and sun_basename_to_split:
                    split = sun_basename_to_split.get(Path(image_name).name)
                    if split is not None:
                        recovered_sun_by_basename += 1
                expected_by_pos = sun_split_sequence[sun_pos] if sun_pos < len(sun_split_sequence) else None
                if split is not None and expected_by_pos is not None and str(split) != str(expected_by_pos):
                    sun_row_order_conflicts += 1
                if split is None and can_use_sun_row_fallback and expected_by_pos is not None:
                    split = expected_by_pos
                    recovered_sun_by_row += 1
                sun_pos += 1
            if split is None:
                missing.append((dataset, image_name))
            assigned.append(split)

        if sun_row_order_conflicts:
            raise ValueError(
                f"Precomputed SUN split row order no longer matches current SUN labels "
                f"({sun_row_order_conflicts} row-order conflicts). Regenerate fold*_split.csv."
            )
        if missing:
            preview = ", ".join([f"{d}:{n}" for d, n in missing[:5]])
            raise ValueError(f"Missing {len(missing)} samples in precomputed split map. Examples: {preview}")

        out = df.copy()
        out['split'] = assigned
        print("\nUsing precomputed split assignments")
        if recovered_sun_by_basename:
            print(f"Recovered {recovered_sun_by_basename} SUN split assignments by filename fallback (path remap compatibility)")
        if recovered_sun_by_row:
            print(f"Recovered {recovered_sun_by_row} SUN split assignments by row-index fallback (stale split CSV path compatibility)")
        return out

    def _acquire_lock(self, lock_path: Path, timeout_sec: int = 600) -> bool:
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                if time.time() - start > timeout_sec:
                    return False
                time.sleep(0.5)

    def _load_or_build_pred_bbox_cache(self, source_df: pd.DataFrame) -> pd.DataFrame:
        cache_path = self._pred_bbox_cache_path()
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

        if cache_path.exists():
            return pd.read_csv(cache_path)

        got_lock = self._acquire_lock(lock_path)
        if not got_lock and cache_path.exists():
            return pd.read_csv(cache_path)

        try:
            # Re-check after lock in case another worker finished first.
            if cache_path.exists():
                return pd.read_csv(cache_path)

            work = source_df.copy()
            key_cols = ["image_name"]
            if "dataset" in work.columns:
                key_cols.insert(0, "dataset")
            work = work[key_cols].drop_duplicates().reset_index(drop=True)

            rows = []
            for _, row in work.iterrows():
                dataset = str(row.get("dataset", "real_colon"))
                image_name = str(row["image_name"])
                mask_path = self._find_mask_path(dataset, image_name)
                if mask_path is None:
                    continue
                try:
                    mask_arr = np.array(Image.open(mask_path).convert("L"))
                except Exception:
                    continue
                pred_bbox = self._bbox_from_mask(mask_arr)
                if pred_bbox is None:
                    continue
                rec = {
                    "image_name": image_name,
                    "pred_bbox_xmin": pred_bbox[0],
                    "pred_bbox_ymin": pred_bbox[1],
                    "pred_bbox_xmax": pred_bbox[2],
                    "pred_bbox_ymax": pred_bbox[3],
                }
                if "dataset" in work.columns:
                    rec["dataset"] = dataset
                rows.append(rec)

            out = pd.DataFrame(rows)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            out.to_csv(tmp, index=False)
            os.replace(tmp, cache_path)
            return out
        finally:
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass

    def _apply_predicted_bbox_overrides(
        self,
        df: pd.DataFrame,
        cache_source_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        source_df = cache_source_df if cache_source_df is not None else df
        cache_df = self._load_or_build_pred_bbox_cache(source_df)
        if cache_df.empty:
            print(f"Predicted-mask bbox overrides applied: 0/{len(df)}")
            return df

        key_cols = ["image_name"]
        if "dataset" in df.columns and "dataset" in cache_df.columns:
            key_cols = ["dataset", "image_name"]

        updated = df.merge(cache_df, on=key_cols, how="left")
        has_pred = updated["pred_bbox_xmin"].notna()
        n_overridden = int(has_pred.sum())

        for tgt, pred in (
            ("bbox_xmin", "pred_bbox_xmin"),
            ("bbox_ymin", "pred_bbox_ymin"),
            ("bbox_xmax", "pred_bbox_xmax"),
            ("bbox_ymax", "pred_bbox_ymax"),
        ):
            updated.loc[has_pred, tgt] = updated.loc[has_pred, pred].astype(int)

        updated = updated.drop(
            columns=["pred_bbox_xmin", "pred_bbox_ymin", "pred_bbox_xmax", "pred_bbox_ymax"],
            errors="ignore",
        )
        print(f"Predicted-mask bbox overrides applied: {n_overridden}/{len(updated)}")
        return updated

    @staticmethod
    def _make_split(
        df: pd.DataFrame,
        strategy: str,
        seed: int,
        train_ratio: float,
        val_ratio: float,
    ) -> pd.DataFrame:
        """Generate train/val/test split assignments.

        Delegates to overnight_experiments.make_split to ensure identical
        splits (same group-label logic, count allocation, and seed handling).
        """
        from overnight_experiments import make_split
        return make_split(df, strategy, seed, train_ratio, val_ratio)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_depth_path(self, dataset: str, depth_idx: int) -> Path:
        """Resolve depth map file path based on dataset and depth_source."""
        if self.depth_source == 'metriccol':
            if dataset == 'real_colon':
                path = self.depth_dir.parent / 'depth_maps_metriccol' / 'real_colon' / f"{depth_idx:06d}.pt"
                if path.exists():
                    return path
            elif dataset == 'polyp_size':
                path = self.depth_dir.parent / 'depth_maps_metriccol' / 'polyp_size' / f"{depth_idx:06d}.pt"
                if path.exists():
                    return path
            elif dataset == 'sun' and self.sun_path:
                path = self.sun_path / 'depth_maps_metriccol' / f"{depth_idx:06d}.pt"
                if path.exists():
                    return path

        # PPSNet (default or fallback)
        if dataset == 'real_colon':
            depth_path = self.depth_dir / 'real_colon' / f"{depth_idx:06d}.pt"
        elif dataset == 'polyp_size':
            depth_path = self.depth_dir / 'polyp_size' / f"{depth_idx:06d}.pt"
        elif dataset == 'sun':
            depth_path = self.depth_dir / 'sun' / f"{depth_idx:06d}.pt"
            if not depth_path.exists() and self.sun_path:
                depth_path = self.sun_path / 'depth_maps_ppsnet' / f"{depth_idx:06d}.pt"
            if not depth_path.exists():
                depth_path = self.depth_dir / f"{depth_idx:06d}.pt"
        else:
            depth_path = self.depth_dir / f"{depth_idx:06d}.pt"
        return depth_path

    def _load_depth_tensor(self, depth_path: Path) -> torch.Tensor:
        """Load depth map from .pt/.pth or .npy and return tensor [C,H,W] or [H,W]."""
        suffix = depth_path.suffix.lower()
        if suffix == '.npy':
            arr = np.load(depth_path)
            depth = torch.from_numpy(arr)
        else:
            try:
                depth = torch.load(depth_path, map_location='cpu', weights_only=True)
            except TypeError:
                # Older torch versions do not support weights_only.
                depth = torch.load(depth_path, map_location='cpu')
        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth)
        return depth.float()

    def __getitem__(self, idx: int) -> Dict:
        row = self.samples.iloc[idx]
        depth_idx = int(row['depth_idx'])
        label = int(row['label'])
        dataset = row.get('dataset', 'real_colon')

        # Load depth map
        if self.has_depth_path_col and pd.notna(row.get('depth_path', None)):
            depth_path = Path(str(row['depth_path']))
        else:
            depth_path = self._resolve_depth_path(dataset, depth_idx)

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth map not found: {depth_path} (dataset={dataset}, depth_source={self.depth_source})")

        depth = self._load_depth_tensor(depth_path)

        # Ensure shape [1, H, W]
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        elif depth.dim() == 3 and depth.shape[0] != 1:
            depth = depth[:1]

        # Optional per-frame scale multiplication before bbox masking.
        if self.scale_factors is not None:
            if self.has_dataset_col and isinstance(self.scale_factors, dict):
                key = (str(dataset), str(row['image_name']))
                scale = self.scale_factors.get(key)
                if scale is None:
                    scale = self.scale_factors.get(str(row['image_name']))
            else:
                scale = self.scale_factors.get(str(row['image_name']))
            if scale is not None and np.isfinite(scale):
                depth = depth * float(scale)

        # Apply polyp bounding box mask: zero out everything outside the bbox.
        _, dh, dw = depth.shape
        orig_w = float(row['image_width'])
        orig_h = float(row['image_height'])

        if self.depth_source == 'metriccol':
            # MetricCol: depth is a direct resize of the original image (no center crop)
            # Simple aspect-ratio-preserving mapping: original coords -> depth coords
            scale_x = dw / orig_w
            scale_y = dh / orig_h
            bbox_x1 = float(row['bbox_xmin']) * scale_x
            bbox_y1 = float(row['bbox_ymin']) * scale_y
            bbox_x2 = float(row['bbox_xmax']) * scale_x
            bbox_y2 = float(row['bbox_ymax']) * scale_y
        else:
            # PPSNet: depth was generated by:
            #   1. Resize shortest side to 518 (preserving aspect ratio)
            #   2. CenterCrop(518) to get a square
            #   3. PPSNet inference -> output resized to (384, 384)
            resize_target = 518
            scale = resize_target / min(orig_h, orig_w)
            resized_w = orig_w * scale
            resized_h = orig_h * scale
            crop_x_offset = (resized_w - resize_target) / 2.0
            crop_y_offset = (resized_h - resize_target) / 2.0
            depth_scale = dw / resize_target
            bbox_x1 = (float(row['bbox_xmin']) * scale - crop_x_offset) * depth_scale
            bbox_y1 = (float(row['bbox_ymin']) * scale - crop_y_offset) * depth_scale
            bbox_x2 = (float(row['bbox_xmax']) * scale - crop_x_offset) * depth_scale
            bbox_y2 = (float(row['bbox_ymax']) * scale - crop_y_offset) * depth_scale

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self._scale_bbox_xyxy(
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, dw, dh, self.mask_bbox_scale_factor
        )

        x1 = max(0, int(bbox_x1))
        y1 = max(0, int(bbox_y1))
        x2 = min(dw, int(bbox_x2))
        y2 = min(dh, int(bbox_y2))

        mask = torch.zeros_like(depth)
        if x2 > x1 and y2 > y1:
            mask[:, y1:y2, x1:x2] = 1.0
        depth = depth * mask

        # Resize to target resolution
        if depth.shape[1] != self.image_size or depth.shape[2] != self.image_size:
            depth = F.interpolate(
                depth.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        # Normalize depth based on stored parameter
        depth_norm_method = self.depth_norm_method
        if depth_norm_method in ('none', 'identity', 'raw'):
            # No normalization — preserve absolute metric depth values
            pass
        else:
            # Min-max normalization to [0, 1] (computed only over non-zero polyp region)
            polyp_vals = depth[depth > 0]
            if polyp_vals.numel() > 0:
                d_min, d_max = polyp_vals.min(), polyp_vals.max()
                if d_max - d_min > 1e-8:
                    depth = torch.where(
                        depth > 0,
                        (depth - d_min) / (d_max - d_min),
                        depth,
                    )

        # Augmentation (training only)
        if self.augment and random.random() < self.flip_prob:
            depth = torch.flip(depth, dims=[2])  # horizontal flip

        return {
            'depth': depth,          # [1, 179, 179]
            'label': label,
            'image_name': row['image_name'],
            'video_id': row['video_id'],
            'dataset': str(dataset),
        }


def create_depth_dataloaders(config: dict) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders for depth-only BseNet training.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'class_weights' tensor.
    """
    data_cfg = config['data']
    aug_cfg = config.get('augmentation', {})

    common = dict(
        csv_path=data_cfg['csv_path'],
        depth_dir=data_cfg['depth_dir'],
        image_size=data_cfg.get('image_size', 179),
        precomputed_split_file=data_cfg.get('precomputed_split_file'),
        split_seed=data_cfg.get('split_seed', 42),
        train_ratio=data_cfg.get('train_split', 0.7),
        val_ratio=data_cfg.get('val_split', 0.15),
        split_strategy=data_cfg.get('split_strategy', 'video_stratified'),
        max_samples=data_cfg.get('max_samples_per_split'),
        sun_path=data_cfg.get('sun_path'),
        depth_source=data_cfg.get('depth_source', 'ppsnet'),
        depth_norm_method=aug_cfg.get('depth_norm_method', data_cfg.get('depth_norm_method', 'minmax')),
        scale_factors_file=data_cfg.get('scale_factors_file'),
        scale_factor_col=data_cfg.get('scale_factor_col', 'scale_factor'),
        real_colon_segmentation_dir=data_cfg.get('real_colon_segmentation_dir'),
        sun_segmentation_dir=data_cfg.get('sun_segmentation_dir'),
        prefer_predicted_mask_bbox=data_cfg.get('prefer_predicted_mask_bbox', False),
        mask_bbox_scale_factor=data_cfg.get('mask_bbox_scale_factor', 1.0),
    )

    train_ds = DepthOnlyPolypDataset(
        split='train', augment=True,
        flip_prob=aug_cfg.get('flip_prob', 0.5),
        **common,
    )
    val_ds = DepthOnlyPolypDataset(split='val', augment=False, **common)
    test_ds = DepthOnlyPolypDataset(split='test', augment=False, **common)

    batch_size = data_cfg.get('batch_size', 64)
    num_workers = data_cfg.get('num_workers', 8)
    pin_memory = data_cfg.get('pin_memory', True)

    # Weighted sampler for training (handle class imbalance)
    use_sampler = config.get('training', {}).get('use_weighted_sampler', False)
    train_sampler = None
    if use_sampler and len(train_ds) > 0:
        labels = train_ds.labels
        class_counts = np.bincount(labels, minlength=2)
        weight_per_class = 1.0 / (class_counts + 1e-8)
        sample_weights = weight_per_class[labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )

    loaders = {
        'train': DataLoader(
            train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=True,
        ),
        'val': DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        'test': DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        'class_weights': train_ds.class_weights,
        'class_counts': train_ds.class_counts,
        'datasets': {'train': train_ds, 'val': val_ds, 'test': test_ds},
    }
    return loaders
