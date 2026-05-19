"""
Unified RGBD dataset loader for polyp size classification

Supports:
- Dataset A: Polyp_Size (frame-level annotations)
- Dataset B: real-colon-dataset (video-level annotations)
- Patient/video-wise splits (no leakage)
- Scenario toggles (baseline, depth rescaling, copy-paste)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
from tqdm import tqdm

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class RGBDPolypDataset(Dataset):
    """Unified dataset for RGB+D polyp classification"""

    def __init__(self,
                 config: Dict,
                 split: str = 'train',
                 transform=None,
                 scenario_id: int = 1):
        """
        Args:
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
            transform: Data transforms (should handle RGB and depth separately)
            scenario_id: 1 (baseline), 2 (depth rescaling), or 3 (copy-paste)
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.scenario_id = scenario_id

        self.data_config = config['data']
        self.image_size = self.data_config['image_size']
        self.num_classes = self.data_config['num_classes']
        self.min_mask_px = self.data_config.get('min_mask_px', 100)
        # Input composition: list of channel names, e.g. ['rgb', 'mask', 'depth', 'pps', 'masked_rgb']
        self.input_composition = self.data_config.get('input_composition', None)
        if self.input_composition:
            self.input_channels = sum(
                3 if c in ('rgb', 'masked_rgb') else 1 for c in self.input_composition
            )
            # Auto-set flags based on composition
            self._needs_depth = any(c in ('depth', 'pps') for c in self.input_composition)
            self._needs_mask_channel = 'mask' in self.input_composition
            self._needs_masked_rgb = 'masked_rgb' in self.input_composition
            self._needs_pps = 'pps' in self.input_composition
        else:
            self.input_channels = int(
                self.data_config.get('input_channels', config.get('model', {}).get('input_channels', 4))
            )
            self._needs_depth = self.input_channels >= 4
            self._needs_mask_channel = False
            self._needs_masked_rgb = False
            self._needs_pps = False
        self.use_mask = bool(self.data_config.get('use_mask', False))
        self.mask_depth = bool(self.data_config.get('mask_depth', self.use_mask))
        self.mask_rgb = bool(self.data_config.get('mask_rgb', False))
        self.force_bbox_mask = bool(self.data_config.get('force_bbox_mask', False))
        self.return_mask = bool(self.data_config.get('return_mask', False)) or self.use_mask or self._needs_mask_channel or self._needs_masked_rgb
        self.return_unmasked = bool(self.data_config.get('return_unmasked', False))
        self.mask_resize_target = int(self.data_config.get('mask_resize_target', 518))
        self.depth_fallback = bool(self.data_config.get('depth_fallback', True))
        self.depth_source = str(self.data_config.get('depth_source', 'ppsnet')).lower()  # 'ppsnet' or 'metriccol'
        self.real_colon_seg_dir = Path(self.data_config.get('real_colon_segmentation_dir') or '')
        self.sun_seg_dir = Path(self.data_config.get('sun_segmentation_dir') or '')
        self.prefer_predicted_mask_bbox = bool(self.data_config.get('prefer_predicted_mask_bbox', True))
        self.mask_bbox_scale_factor = float(self.data_config.get('mask_bbox_scale_factor', 1.0))
        # PPS camera intrinsics (needed when input_composition includes 'pps')
        pps_cfg = self.data_config.get('pps', {}) or {}
        self._pps_fx = float(pps_cfg.get('fx', 227.60416))
        self._pps_fy = float(pps_cfg.get('fy', 237.5))
        self._pps_cx = float(pps_cfg.get('cx', 227.60416))
        self._pps_cy = float(pps_cfg.get('cy', 237.5))
        self.temporal_cfg = self.data_config.get('temporal', {}) or {}
        self.temporal_enabled = bool(self.temporal_cfg.get('enabled', False))
        mask_consistency_weight = float(self.config.get('training', {}).get('mask_consistency_weight', 0.0))
        if mask_consistency_weight > 0:
            self.return_mask = True
            self.return_unmasked = True

        # Paths
        self.polyp_size_images = Path(self.data_config['polyp_size_images'])
        self.polyp_size_labels = Path(self.data_config['polyp_size_labels'])
        self.real_colon_path = Path(self.data_config['real_colon_path'])
        self.depth_dir = Path(self.data_config['depth_maps_dir'] or '')
        self.seg_dir = Path(self.data_config.get('segmentation_dir') or '')
        self.real_colon_allowlist = self._load_real_colon_allowlist()

        # Load size measurements if available
        self.measurements = None
        measurements_file = self.data_config.get('size_measurements_file')
        if measurements_file and Path(measurements_file).exists():
            with open(measurements_file) as f:
                measurements_list = json.load(f)
                self.measurements = {m['image_name']: m for m in measurements_list}

        # Load scale factors if available (Scenario 2)
        self.scale_factors = None
        if scenario_id == 2:
            scale_file = self.data_config.get('scale_factors_file')
            if scale_file and Path(scale_file).exists():
                df = pd.read_csv(scale_file)
                self.scale_factors = dict(zip(df['image_name'], df['scale_factor']))

        # Load samples
        all_samples = self._load_samples()

        # Initialize copy-paste augmentation for Scenario 3
        # Note: Build pools from ALL samples (before split filtering) so we can paste from any source
        # IMPORTANT: Only initialize if copy_paste is enabled (on-the-fly mode)
        # For pre-generated augmented samples, this is not needed
        self.copy_paste_aug = None
        if scenario_id == 3 and config.get('copy_paste', {}).get('enabled', False):
            if self.input_channels == 3:
                print("Warning: RGB-only input; disabling copy-paste augmentation (requires depth).")
                self.copy_paste_aug = None
            else:
                self.copy_paste_aug = self._init_copy_paste_augmentation(all_samples, config)

        # Filter to current split
        self.samples = [s for s in all_samples if s['split'] == split]

        # Optional sample cap (useful for smoke tests)
        max_samples = self.data_config.get('max_samples_per_split')
        if max_samples is not None and len(self.samples) > int(max_samples):
            split_offset = {'train': 1, 'val': 2, 'test': 3}.get(split, 0)
            rng = random.Random(self.data_config.get('split_seed', 42) + split_offset)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            keep = set(indices[:int(max_samples)])
            self.samples = [s for i, s in enumerate(self.samples) if i in keep]
            print(f"Limited {split} split to {len(self.samples)} samples (max_samples_per_split={max_samples})")

        print(f"Loaded {len(self.samples)} samples for {split} split")

        # Build temporal clip groups if enabled
        self.clip_groups = None
        self.clip_labels = None
        if self.temporal_enabled:
            self.clip_groups, self.clip_labels = self._build_temporal_groups()
            print(f"Built {len(self.clip_groups)} temporal clips for {split} split")

        # Compute class weights
        self._compute_class_stats()

        # Load photometry features (optional)
        self.photometry_features = None
        self.photometry_mean = None
        self.photometry_std = None
        self.photometry_cols = None
        self.photometry_col_to_idx = {}
        self.photo_image_cfg = self.data_config.get('photo_image', {}) or {}
        self.photo_image_enabled = bool(self.photo_image_cfg.get('enabled', False))
        self.photo_size = int(self.photo_image_cfg.get('size', self.image_size))
        self.photo_channels = int(self.photo_image_cfg.get('channels', 3))
        self.photo_include_rgb_ratio = bool(
            self.photo_image_cfg.get('include_rgb_ratio', self.photo_channels >= 6)
        )
        self.photo_v_thresh = float(self.photo_image_cfg.get('specular_v_thresh', 0.95))
        self.photo_s_thresh = float(self.photo_image_cfg.get('specular_s_thresh', 0.10))
        self.photo_polyp_axis_ratio = float(self.photo_image_cfg.get('polyp_axis_ratio', 0.45))
        self.photo_ring_inner = float(self.photo_image_cfg.get('ring_inner_ratio', 0.55))
        self.photo_ring_outer = float(self.photo_image_cfg.get('ring_outer_ratio', 0.85))
        if self.photo_image_enabled and cv2 is None:
            raise ImportError("photo_image.enabled=true requires OpenCV (cv2), but cv2 is not available.")
        photometry_csv = self.data_config.get('photometry_csv')
        if photometry_csv and Path(photometry_csv).exists():
            self._load_photometry_features(photometry_csv)

    def _load_photometry_features(self, csv_path: str):
        """Load precomputed photometry features and build lookup by image_path."""
        df = pd.read_csv(csv_path)

        # Identify feature columns (exclude metadata)
        meta_cols = {'dataset', 'video_id', 'lesion_id', 'frame_idx', 'image_path',
                     'size_mm', 'label'}
        self.photometry_cols = [c for c in df.columns if c not in meta_cols]
        self.photometry_col_to_idx = {c: i for i, c in enumerate(self.photometry_cols)}

        # Build lookup dict keyed by image_path (normalized)
        feat_lookup = {}
        for _, row in df.iterrows():
            key = str(row.get('image_path', ''))
            if key:
                feat_lookup[key] = np.array([row[c] for c in self.photometry_cols],
                                            dtype=np.float32)

        # Match features to samples
        matched = 0
        for sample in self.samples:
            key = str(sample['image_path'])
            if key in feat_lookup:
                sample['_photometry'] = feat_lookup[key]
                matched += 1
            else:
                sample['_photometry'] = None

        print(f"Photometry features: matched {matched}/{len(self.samples)} samples "
              f"({len(self.photometry_cols)} features)")

        # Compute standardization stats from training set only
        if self.split == 'train':
            all_feats = []
            for s in self.samples:
                if s.get('_photometry') is not None:
                    f = s['_photometry']
                    if not np.any(np.isnan(f)):
                        all_feats.append(f)
            if all_feats:
                feats_arr = np.stack(all_feats)
                self.photometry_mean = feats_arr.mean(axis=0)
                self.photometry_std = feats_arr.std(axis=0)
                self.photometry_std[self.photometry_std < 1e-8] = 1.0
        self.photometry_features = True  # Flag that features are loaded

    def set_photometry_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set standardization stats (called on val/test sets using train stats)."""
        self.photometry_mean = mean
        self.photometry_std = std

    def _load_samples(self) -> List[Dict]:
        """Load and combine samples from selected datasets"""
        samples = []

        # Determine which datasets to load (default: both)
        dataset_types = self.data_config.get('dataset_types', ['polyp_size', 'real_colon'])

        # Load Dataset A: Polyp_Size
        if 'polyp_size' in dataset_types:
            samples_a = self._load_polyp_size_dataset()
            samples.extend(samples_a)

        # Load Dataset B: real-colon (if exists and selected)
        if 'real_colon' in dataset_types and self.real_colon_path.exists():
            samples_b = self._load_real_colon_dataset()
            samples.extend(samples_b)

        # Load Dataset C: SUN-SEG
        if 'sun' in dataset_types:
            samples_c = self._load_sun_dataset()
            samples.extend(samples_c)

        # Load precomputed split map if provided
        split_map = self._load_precomputed_split_map()
        if split_map is not None:
            samples = self._assign_splits_from_map(samples, split_map)
        else:
            # Create splits from strategy
            samples = self._create_splits(samples)

        return samples

    def _load_polyp_size_dataset(self) -> List[Dict]:
        """Load Polyp_Size dataset from frame_labels.csv"""
        samples = []

        # Load labels CSV
        if not self.polyp_size_labels.exists():
            print(f"Warning: Labels file not found: {self.polyp_size_labels}")
            return samples

        df = pd.read_csv(self.polyp_size_labels)
        max_label = df['label'].max() if 'label' in df.columns and not df.empty else None

        for idx, row in df.iterrows():
            # Parse from CSV columns: image_name, video_id, depth_idx, size_mm, label, has_mask, has_depth
            image_name = row.get('image_name', '')
            video_id = row.get('video_id', 'unknown')
            depth_idx = row.get('depth_idx', idx)
            size_mm = row.get('size_mm', 0.0)
            label = int(row.get('label', 1))
            if self.num_classes == 2:
                if max_label is not None and max_label <= 1:
                    # Already binary labels using the 5 mm threshold: keep as-is (0/1)
                    pass
                else:
                    # Legacy multiclass labels are not part of the public 5 mm task.
                    if label == 0:
                        continue
                    if label in (1, 2):
                        label = label - 1
            has_mask = row.get('has_mask', True)
            has_depth = row.get('has_depth', True)

            # Skip if no depth available (only for RGBD)
            if self.input_channels == 4 and not has_depth:
                continue

            sample = {
                'image_name': image_name,
                'image_path': self.polyp_size_images / image_name,
                'label': label,
                'size_mm': float(size_mm),
                'video_id': video_id,
                'lesion_id': row.get('lesion_id', row.get('polyp_idx', video_id)),
                'dataset': 'polyp_size',
                'depth_idx': depth_idx,
                'has_mask': has_mask,
                'split': None  # Will be assigned later
            }

            samples.append(sample)

        return samples

    def _load_real_colon_dataset(self) -> List[Dict]:
        """Load Real-Colon dataset from real_colon_frame_labels.csv"""
        samples = []

        # Look for Real-Colon frame labels CSV
        real_colon_labels = self.data_config.get('real_colon_labels')
        if real_colon_labels is None:
            # Try default location
            real_colon_labels = Path(self.data_config['polyp_size_labels']).parent / 'real_colon_frame_labels.csv'
        else:
            real_colon_labels = Path(real_colon_labels)

        if not real_colon_labels.exists():
            print(f"Info: Real-Colon labels CSV not found: {real_colon_labels}")
            return samples

        # Load Real-Colon labels CSV
        df = pd.read_csv(real_colon_labels)

        # Optional allowlist filtering (by image_name and/or frame_path)
        allow_names = self.real_colon_allowlist.get('image_names', set())
        allow_frames = self.real_colon_allowlist.get('frame_paths', set())
        if allow_names or allow_frames:
            before = len(df)
            mask = pd.Series(False, index=df.index)
            if allow_names and 'image_name' in df.columns:
                mask |= df['image_name'].isin(allow_names)
            if allow_frames and 'frame_path' in df.columns:
                mask |= df['frame_path'].isin(allow_frames)
            df = df[mask]
            print(f"Filtered Real-Colon labels by allowlist: {before} -> {len(df)}")
        max_label = df['label'].max() if 'label' in df.columns and not df.empty else None

        # Get dataset directory
        dataset_dir = self.real_colon_path / 'dataset'
        if not dataset_dir.exists():
            print(f"Warning: Real-Colon dataset directory not found: {dataset_dir}")
            return samples

        for idx, row in df.iterrows():
            # Parse from CSV columns: image_name, video_id, frame_path, lesion_id, size_mm, label,
            # bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, image_width, image_height, ...
            image_name = row.get('image_name', '')
            video_id = row.get('video_id', 'unknown')
            frame_path = row.get('frame_path', '')
            size_mm = row.get('size_mm', 0.0)
            label = int(row.get('label', 1))
            if self.num_classes == 2:
                if max_label is not None and max_label <= 1:
                    # Already binary labels using the 5 mm threshold: keep as-is (0/1)
                    pass
                else:
                    # Legacy multiclass labels are not part of the public 5 mm task.
                    if label == 0:
                        continue
                    if label in (1, 2):
                        label = label - 1

            # Check if this is an augmented sample (Scenario 3)
            is_augmented = row.get('is_augmented', False)

            bbox = self._extract_bbox_from_row(row)

            image_width = int(row.get('image_width', 0))
            image_height = int(row.get('image_height', 0))

            # Construct full path to frame
            if is_augmented:
                # Augmented samples: use base_dir / frame_path (already contains augmented_copy_paste/...)
                base_dir = self.real_colon_path.parent  # Go up to data/Size/
                full_frame_path = base_dir / frame_path
                dataset_type = 'augmented'
            else:
                # Original samples
                full_frame_path = dataset_dir / frame_path
                dataset_type = 'real_colon'

            if not full_frame_path.exists():
                continue

            sample = {
                'image_name': image_name,
                'image_path': full_frame_path,
                'label': label,
                'size_mm': float(size_mm),
                'video_id': video_id,
                'lesion_id': row.get('lesion_id', video_id),
                'dataset': dataset_type,  # 'real_colon' or 'augmented'
                'depth_idx': idx,  # Use CSV index for depth file lookup
                'is_augmented': is_augmented,
                'bbox': bbox,
                'image_width': image_width,
                'image_height': image_height,
                'split': None  # Will be assigned later
            }
            sample = self._apply_predicted_bbox_override(sample)

            samples.append(sample)

        print(f"Loaded {len(samples)} frames from Real-Colon dataset")
        return samples

    def _load_sun_dataset(self) -> List[Dict]:
        """Load SUN-SEG dataset from sun_labels CSV."""
        samples = []

        sun_labels = self.data_config.get('sun_labels')
        if sun_labels is None:
            print("Info: No sun_labels path in config, skipping SUN dataset")
            return samples

        sun_labels = Path(sun_labels)
        if not sun_labels.exists():
            print(f"Warning: SUN labels CSV not found: {sun_labels}")
            return samples

        sun_path = self.data_config.get('sun_path', '')
        depth_dir = self.data_config.get('depth_maps_dir', '')

        df = pd.read_csv(sun_labels)

        for idx, row in df.iterrows():
            image_name = row.get('image_name', '')
            if not image_name:
                continue

            image_path = Path(image_name)
            if not image_path.exists():
                continue

            video_id = str(row.get('video_id', 'unknown'))
            lesion_id = str(row.get('lesion_id', video_id))
            label = int(row['label'])
            size_mm = float(row.get('gt', row.get('size_mm', 0)))

            # Skip no-polyp frames
            if size_mm == 0 or pd.isna(size_mm):
                continue

            # Depth file (optional)
            depth_idx = row.get('depth_idx', idx)
            has_depth = False
            if depth_dir and Path(depth_dir).exists():
                has_depth = True

            if self.input_channels == 4 and not has_depth:
                continue

            # Bbox for mask generation (prefer CSV if present, else infer from mask if configured)
            bbox = self._extract_bbox_from_row(row)

            sample = {
                'image_name': image_name,
                'image_path': image_path,
                'label': label,
                'size_mm': size_mm,
                'video_id': video_id,
                'lesion_id': lesion_id,
                'dataset': 'sun',
                'depth_idx': depth_idx,
                'has_mask': bbox is not None,
                'bbox': bbox,
                'kfold_id': row.get('kfold_id', None),
                'split': None,
            }
            sample = self._apply_predicted_bbox_override(sample)
            samples.append(sample)

        print(f"Loaded {len(samples)} frames from SUN dataset")
        return samples

    def _extract_bbox_from_row(self, row: pd.Series) -> Optional[Dict[str, int]]:
        keys = ('bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax')
        values = [row.get(k) for k in keys]
        if any(pd.isna(v) for v in values):
            return None
        return {
            'xmin': int(values[0]),
            'ymin': int(values[1]),
            'xmax': int(values[2]),
            'ymax': int(values[3]),
        }

    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[Dict[str, int]]:
        if mask is None:
            return None
        bin_mask = mask > 0
        if not np.any(bin_mask):
            return None
        ys, xs = np.where(bin_mask)
        return {
            'xmin': int(xs.min()),
            'ymin': int(ys.min()),
            # Keep xmax/ymax as exclusive to match slicing behavior in _create_mask_from_bbox.
            'xmax': int(xs.max()) + 1,
            'ymax': int(ys.max()) + 1,
        }

    def _scale_bbox_xyxy(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float,
        height: float,
    ) -> Tuple[float, float, float, float]:
        factor = float(self.mask_bbox_scale_factor)
        if (not np.isfinite(factor)) or factor <= 0 or abs(factor - 1.0) < 1e-8:
            return x1, y1, x2, y2
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(1e-6, x2 - x1) * factor
        bh = max(1e-6, y2 - y1) * factor
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        return x1, y1, x2, y2

    def _apply_predicted_bbox_override(self, sample: Dict) -> Dict:
        """Optionally replace GT bbox by bbox inferred from predicted segmentation mask."""
        gt_bbox = sample.get('bbox')
        if not self.prefer_predicted_mask_bbox:
            sample['has_mask'] = gt_bbox is not None
            return sample

        mask_path = self._find_mask_path(sample)
        if mask_path and mask_path.exists():
            mask_arr = np.array(Image.open(mask_path).convert('L'))
            pred_bbox = self._bbox_from_mask(mask_arr)
            if pred_bbox is not None:
                sample['bbox'] = pred_bbox
                sample['has_mask'] = True
                return sample

        sample['has_mask'] = gt_bbox is not None
        return sample

    def _load_real_colon_allowlist(self) -> Dict[str, set]:
        """Load optional allowlist of Real-Colon frames from a dir of symlinks or a CSV."""
        allowlist = {'image_names': set(), 'frame_paths': set()}

        allow_dir = self.data_config.get('real_colon_allowlist_dir')
        if allow_dir:
            d = Path(allow_dir)
            if d.exists():
                for p in d.iterdir():
                    if not (p.is_file() or p.is_symlink()):
                        continue
                    name = p.name
                    if "__" in name:
                        name = name.split("__", 1)[1]
                    allowlist['image_names'].add(name)
            else:
                print(f"Warning: real_colon_allowlist_dir not found: {d}")

        allow_csv = self.data_config.get('real_colon_allowlist_csv')
        if allow_csv:
            csv_path = Path(allow_csv)
            if not csv_path.exists():
                print(f"Warning: real_colon_allowlist_csv not found: {csv_path}")
            else:
                with csv_path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        image_name = (row.get('image_name') or '').strip()
                        if image_name:
                            allowlist['image_names'].add(image_name)
                        frame_path = (row.get('frame_path') or '').strip()
                        if frame_path:
                            allowlist['frame_paths'].add(frame_path)
                        image_path = (row.get('image_path') or '').strip()
                        if image_path:
                            allowlist['image_names'].add(Path(image_path).name)

        if (allow_dir or allow_csv) and not (allowlist['image_names'] or allowlist['frame_paths']):
            print("Warning: real_colon allowlist provided but no entries were found.")

        return allowlist

    def _parse_xml_annotation(self, xml_path: Path) -> Tuple[int, float]:
        """Parse XML annotation file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Check if there are any objects
            objects = root.findall('.//object')

            if len(objects) == 0:
                return 0, 0.0

            # Get size if available (this depends on your XML format)
            # Assuming size is in lesion_info or similar
            # For now, default to unknown size
            size_mm = 0.0

            # If size is unknown, default to the >5 mm class for legacy XML inputs.
            # Or use measurements if available
            return 1, size_mm

        except Exception as e:
            return 0, 0.0

    def _create_splits(self, samples: List[Dict]) -> List[Dict]:
        """
        Create splits using configurable strategy:
          - video_stratified (default)
          - lesion_stratified
          - frame_stratified
        """
        strategy = self.data_config.get('split_strategy', 'video_stratified')
        split_seed = self.data_config.get('split_seed', 42)
        train_ratio = self.data_config.get('train_split', 0.7)
        val_ratio = self.data_config.get('val_split', 0.15)
        rng = random.Random(split_seed)

        # KFold: use pre-assigned kfold_id column (e.g. SUN official splits)
        if strategy == 'kfold':
            kfold_ids = [s.get('kfold_id') for s in samples]
            valid_ids = [k for k in kfold_ids if k is not None]
            if not valid_ids:
                raise ValueError("kfold strategy requires 'kfold_id' in samples")
            n_folds = int(max(valid_ids)) + 1
            fold_order = list(range(n_folds))
            rng.shuffle(fold_order)
            test_fold = fold_order[0]
            val_fold = fold_order[1] if n_folds > 1 else test_fold
            train_folds = set(fold_order[2:]) if n_folds > 2 else set()

            for sample in samples:
                fid = sample.get('kfold_id')
                if fid is None:
                    sample['split'] = 'train'
                elif int(fid) == test_fold:
                    sample['split'] = 'test'
                elif int(fid) == val_fold:
                    sample['split'] = 'val'
                else:
                    sample['split'] = 'train'

            n_train = sum(1 for s in samples if s['split'] == 'train')
            n_val = sum(1 for s in samples if s['split'] == 'val')
            n_test = sum(1 for s in samples if s['split'] == 'test')
            print(f"\nSplit strategy: kfold (seed={split_seed}, n_folds={n_folds})")
            print(f"  Test fold={test_fold}, Val fold={val_fold}, Train folds={sorted(train_folds)}")
            print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
            return samples

        if strategy == 'video_stratified':
            group_key = 'video_id'
        elif strategy == 'lesion_stratified':
            group_key = 'lesion_id'
        elif strategy == 'frame_stratified':
            group_key = None
        else:
            raise ValueError(f"Unknown split_strategy={strategy}")

        groups = defaultdict(list)
        if group_key is None:
            for i, sample in enumerate(samples):
                groups[f"frame_{i}"].append(sample)
        else:
            for sample in samples:
                key = str(sample.get(group_key, sample.get('video_id', 'unknown')))
                groups[key].append(sample)

        # Group ids by class label (group label = max frame label in that group)
        group_label = {gid: max(s['label'] for s in group_samples) for gid, group_samples in groups.items()}
        group_ids_by_class = defaultdict(list)
        for gid, lbl in group_label.items():
            group_ids_by_class[lbl].append(gid)

        for lbl in group_ids_by_class:
            rng.shuffle(group_ids_by_class[lbl])

        def allocate_counts(n_items: int):
            if n_items <= 1:
                return n_items, 0, 0
            if n_items == 2:
                return 1, 0, 1

            n_train = int(n_items * train_ratio)
            n_val = int(n_items * val_ratio)
            n_test = n_items - n_train - n_val

            # Keep all splits non-empty whenever possible
            if n_train == 0:
                n_train = 1
                n_test -= 1
            if n_val == 0:
                if n_test > 1:
                    n_val = 1
                    n_test -= 1
                elif n_train > 1:
                    n_val = 1
                    n_train -= 1
            if n_test == 0:
                if n_train > 1:
                    n_train -= 1
                    n_test = 1
                elif n_val > 1:
                    n_val -= 1
                    n_test = 1
            return n_train, n_val, n_test

        train_groups = []
        val_groups = []
        test_groups = []

        # Stratified split at group level
        for label, class_group_ids in group_ids_by_class.items():
            n_groups = len(class_group_ids)
            if n_groups == 0:
                continue
            n_train, n_val, n_test = allocate_counts(n_groups)
            train_groups.extend(class_group_ids[:n_train])
            val_groups.extend(class_group_ids[n_train:n_train + n_val])
            test_groups.extend(class_group_ids[n_train + n_val:n_train + n_val + n_test])

        train_set = set(train_groups)
        val_set = set(val_groups)
        for sample in samples:
            if group_key is None:
                # frame-level mode: iterate order defines split by synthetic group
                sample['split'] = None
            else:
                gid = str(sample.get(group_key, sample.get('video_id', 'unknown')))
                if gid in train_set:
                    sample['split'] = 'train'
                elif gid in val_set:
                    sample['split'] = 'val'
                else:
                    sample['split'] = 'test'

        if group_key is None:
            # Assign by synthetic frame groups
            frame_group_to_split = {}
            for gid in train_groups:
                frame_group_to_split[gid] = 'train'
            for gid in val_groups:
                frame_group_to_split[gid] = 'val'
            for gid in test_groups:
                frame_group_to_split[gid] = 'test'

            for i, sample in enumerate(samples):
                sample['split'] = frame_group_to_split[f"frame_{i}"]

        print(f"\nSplit strategy: {strategy} (seed={split_seed})")
        print(f"  Train groups: {len(train_groups)}, frames: {sum(len(groups[g]) for g in train_groups)}")
        print(f"  Val groups: {len(val_groups)}, frames: {sum(len(groups[g]) for g in val_groups)}")
        print(f"  Test groups: {len(test_groups)}, frames: {sum(len(groups[g]) for g in test_groups)}")

        class_names = self.config.get('evaluation', {}).get('class_names')
        if not class_names:
            class_names = ['le_5mm', 'gt_5mm'] if self.num_classes == 2 else [f'class_{i}' for i in range(self.num_classes)]

        print("\nGroups by max class:")
        for label in sorted(group_ids_by_class.keys()):
            group_ids = group_ids_by_class[label]
            n_train = sum(1 for g in train_groups if group_label[g] == label)
            n_val = sum(1 for g in val_groups if group_label[g] == label)
            n_test = sum(1 for g in test_groups if group_label[g] == label)
            cname = class_names[label] if label < len(class_names) else f'class_{label}'
            print(f"  {cname} ({label}): {len(group_ids)} groups -> train:{n_train}, val:{n_val}, test:{n_test}")

        self._export_split_artifacts(samples)
        return samples

    def _load_precomputed_split_map(self):
        """Load precomputed split file if provided in config."""
        split_file = self.data_config.get('precomputed_split_file')
        if not split_file:
            return None
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"precomputed_split_file not found: {split_path}")

        split_df = pd.read_csv(split_path)
        required = {'dataset', 'image_name', 'split'}
        if not required.issubset(set(split_df.columns)):
            raise ValueError(f"Split file must contain columns: {required}")

        split_map = {
            (str(row['dataset']), str(row['image_name'])): str(row['split'])
            for _, row in split_df.iterrows()
        }
        print(f"Loaded precomputed split map from {split_path} ({len(split_map)} rows)")
        return split_map

    def _assign_splits_from_map(self, samples: List[Dict], split_map: Dict) -> List[Dict]:
        """Assign split labels from precomputed split map."""
        # SUN image paths were remapped (directory-level corrections) after some split CSVs
        # were exported. Recover by basename for SUN only (filenames are unique).
        sun_basename_to_split: Dict[str, str] = {}
        sun_ambiguous_basenames = set()
        for (dataset_name, image_name), split_name in split_map.items():
            if str(dataset_name) != 'sun':
                continue
            base = Path(str(image_name)).name
            if not base:
                continue
            prev = sun_basename_to_split.get(base)
            if prev is None:
                sun_basename_to_split[base] = str(split_name)
            elif prev != str(split_name):
                sun_ambiguous_basenames.add(base)
        for base in sun_ambiguous_basenames:
            sun_basename_to_split.pop(base, None)

        sun_split_sequence = [
            str(split_name)
            for (dataset_name, _image_name), split_name in split_map.items()
            if str(dataset_name) == 'sun'
        ]
        sun_samples_total = sum(1 for s in samples if str(s.get('dataset')) == 'sun')
        can_use_sun_index_fallback = len(sun_split_sequence) == sun_samples_total and sun_samples_total > 0

        missing = []
        recovered_sun_by_basename = 0
        recovered_sun_by_row_index = 0
        sun_row_order_conflicts = 0
        sun_pos = 0
        for sample in samples:
            key = (str(sample['dataset']), str(sample['image_name']))
            split = split_map.get(key)
            if split is None and key[0] == 'sun' and sun_basename_to_split:
                base = Path(key[1]).name
                split = sun_basename_to_split.get(base)
                if split is not None:
                    recovered_sun_by_basename += 1
            if key[0] == 'sun':
                expected_by_pos = sun_split_sequence[sun_pos] if sun_pos < len(sun_split_sequence) else None
                if split is not None and expected_by_pos is not None and str(split) != str(expected_by_pos):
                    sun_row_order_conflicts += 1
                if split is None and can_use_sun_index_fallback and expected_by_pos is not None:
                    split = expected_by_pos
                    recovered_sun_by_row_index += 1
                sun_pos += 1
            if split is None:
                missing.append(key)
                continue
            sample['split'] = split

        if sun_row_order_conflicts:
            raise ValueError(
                f"Precomputed SUN split row order no longer matches current SUN labels "
                f"({sun_row_order_conflicts} row-order conflicts). Regenerate fold*_split.csv."
            )

        if missing:
            preview = ", ".join([f"{k[0]}:{k[1]}" for k in missing[:5]])
            raise ValueError(
                f"Missing {len(missing)} samples in precomputed split map. Examples: {preview}"
            )

        print("\nUsing precomputed split assignments")
        if recovered_sun_by_basename:
            print(f"Recovered {recovered_sun_by_basename} SUN split assignments by filename fallback (path remap compatibility)")
        if recovered_sun_by_row_index:
            print(f"Recovered {recovered_sun_by_row_index} SUN split assignments by row-index fallback (stale split CSV path compatibility)")
        self._export_split_artifacts(samples)
        return samples

    def _export_split_artifacts(self, samples: List[Dict]):
        """Optionally export split membership and split distributions."""
        split_artifact_dir = self.data_config.get('split_artifact_dir')
        if not split_artifact_dir:
            return

        out_dir = Path(split_artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        split_rows = []
        for s in samples:
            split_rows.append({
                'dataset': s.get('dataset', ''),
                'image_name': s.get('image_name', ''),
                'video_id': s.get('video_id', ''),
                'lesion_id': s.get('lesion_id', ''),
                'label': s.get('label', -1),
                'split': s.get('split', '')
            })
        split_df = pd.DataFrame(split_rows)
        split_df.to_csv(out_dir / 'split_assignments.csv', index=False)

        summary = []
        for split in ['train', 'val', 'test']:
            sdf = split_df[split_df['split'] == split]
            row = {
                'split': split,
                'frames': int(len(sdf)),
                'videos': int(sdf['video_id'].nunique()) if 'video_id' in sdf.columns else 0,
                'lesions': int(sdf['lesion_id'].nunique()) if 'lesion_id' in sdf.columns else 0,
            }
            for cls in sorted(sdf['label'].dropna().unique().tolist()):
                row[f'class_{int(cls)}_count'] = int((sdf['label'] == cls).sum())
            summary.append(row)

        pd.DataFrame(summary).to_csv(out_dir / 'split_summary.csv', index=False)

    def _compute_class_stats(self):
        """Compute class distribution and weights"""
        if self.temporal_enabled and self.clip_labels is not None:
            labels = list(self.clip_labels)
        else:
            labels = [s['label'] for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)

        self.class_counts = dict(zip(unique.tolist(), counts.tolist()))
        self.class_priors = {k: v / len(labels) for k, v in self.class_counts.items()}

        # Compute class weights: 1 / log(1.02 + p)
        self.class_weights = {}
        for k, p in self.class_priors.items():
            self.class_weights[k] = 1.0 / np.log(1.02 + p)

        print(f"\nClass distribution ({self.split}):")

        # Determine class names based on num_classes
        if self.num_classes == 2:
            class_names = ['le_5mm', 'gt_5mm']
            label_range = [0, 1]
        else:
            class_names = [f'class_{i}' for i in range(self.num_classes)]
            label_range = [0, 1, 2]

        for label in label_range:
            count = self.class_counts.get(label, 0)
            prior = self.class_priors.get(label, 0)
            weight = self.class_weights.get(label, 0)
            class_name = class_names[label] if label < len(class_names) else f'class_{label}'
            print(f"  {class_name}: {count} ({prior*100:.1f}%), weight={weight:.3f}")

    def __len__(self):
        if self.temporal_enabled and self.clip_groups is not None:
            return len(self.clip_groups)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.temporal_enabled:
            return self._get_temporal_item(idx)

        sample = self.samples[idx]
        image, label, mask, image_unmasked = self._load_single_sample(sample)
        output = {
            'image': image,
            'label': label,
            'image_name': sample['image_name'],
            'size_mm': sample['size_mm'],
            'video_id': sample['video_id'],
            'lesion_id': sample.get('lesion_id', sample['video_id']),
            'dataset': sample.get('dataset', ''),
        }
        if self.return_mask and mask is not None:
            output['mask'] = mask
        if self.return_unmasked and image_unmasked is not None:
            output['image_unmasked'] = image_unmasked
        # Photometry features (optional)
        if self.photometry_features:
            output['features'] = self._get_sample_photometry_tensor(sample)
        if self.photo_image_enabled:
            output['photo_img'] = self._get_sample_photo_tensor(sample)
        return output

    def _get_sample_photometry_tensor(self, sample: Dict) -> torch.Tensor:
        """Return standardized photometry tensor for one sample."""
        if sample.get('_photometry') is not None:
            feats = sample['_photometry'].copy()
            feats = np.nan_to_num(feats, nan=0.0)
            if self.photometry_mean is not None and self.photometry_std is not None:
                feats = (feats - self.photometry_mean) / self.photometry_std
            return torch.from_numpy(feats).float()
        n_feats = len(self.photometry_cols) if self.photometry_cols else 1
        return torch.zeros(n_feats, dtype=torch.float32)

    def _get_sample_lpx(self, sample: Dict) -> float:
        """Compute sqrt(area_px) proxy from bbox or raw photometry area."""
        bbox = sample.get('bbox')
        if bbox:
            w = max(1, int(bbox.get('xmax', 0)) - int(bbox.get('xmin', 0)) + 1)
            h = max(1, int(bbox.get('ymax', 0)) - int(bbox.get('ymin', 0)) + 1)
            return float(np.sqrt(w * h))
        feats = sample.get('_photometry')
        if feats is not None and self.photometry_col_to_idx:
            area_idx = self.photometry_col_to_idx.get('apparent_area_px')
            if area_idx is not None:
                area = float(feats[area_idx])
                if area > 0:
                    return float(np.sqrt(area))
        return 1.0

    def _empty_photo_tensor(self) -> torch.Tensor:
        return torch.zeros((self.photo_channels, self.photo_size, self.photo_size), dtype=torch.float32)

    def _infer_sun_mask_path(self, image_path: Path) -> Optional[Path]:
        path_str = str(image_path)
        candidates = []
        if "/Frame/" in path_str:
            candidates.append(path_str.replace("/Frame/", "/GT/"))
        if path_str.lower().endswith(".jpg") or path_str.lower().endswith(".jpeg"):
            candidates.append(str(Path(path_str).with_suffix(".png")))
        derived = []
        for c in candidates:
            p = Path(c)
            if p.suffix.lower() != ".png":
                p = p.with_suffix(".png")
            derived.append(p)
        for p in derived:
            if p.exists():
                return p
        return None

    def _pseudo_ellipse_regions(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[0:h, 0:w]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ax = max(1.0, self.photo_polyp_axis_ratio * w)
        ay = max(1.0, self.photo_polyp_axis_ratio * h)
        radius = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
        polyp_region = radius <= 1.0
        inner = float(min(self.photo_ring_inner, self.photo_ring_outer))
        outer = float(max(self.photo_ring_inner, self.photo_ring_outer))
        ring_region = (radius >= inner) & (radius <= outer)
        return polyp_region, ring_region

    def _get_photo_mask(self, sample: Dict, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        h, w = image_shape
        mask = None
        if not self.force_bbox_mask:
            mask_path = self._find_mask_path(sample)
            if mask_path and mask_path.exists():
                mask = np.array(Image.open(mask_path).convert('L'))

        if mask is not None:
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return (mask > 0).astype(np.uint8) * 255

        bbox = sample.get('bbox')
        if bbox:
            return self._create_mask_from_bbox(bbox, image_size=(h, w))
        return None

    def _get_sample_photo_tensor(self, sample: Dict) -> torch.Tensor:
        image_path = Path(sample.get('image_path', ''))
        if not image_path.exists():
            return self._empty_photo_tensor()

        rgb = np.array(Image.open(image_path).convert('RGB'))
        h, w = rgb.shape[:2]
        full_mask = self._get_photo_mask(sample, (h, w))

        bbox = sample.get('bbox')
        if bbox:
            x1 = int(max(0, min(w - 1, bbox.get('xmin', 0))))
            y1 = int(max(0, min(h - 1, bbox.get('ymin', 0))))
            x2 = int(max(x1, min(w - 1, bbox.get('xmax', w - 1))))
            y2 = int(max(y1, min(h - 1, bbox.get('ymax', h - 1))))
        elif full_mask is not None and np.any(full_mask > 0):
            ys, xs = np.where(full_mask > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
        else:
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1

        crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1]
        crop_mask = None
        if full_mask is not None:
            crop_mask = full_mask[y1:y2 + 1, x1:x2 + 1]

        crop_rgb = cv2.resize(crop_rgb, (self.photo_size, self.photo_size), interpolation=cv2.INTER_LINEAR)
        if crop_mask is not None:
            crop_mask = cv2.resize(crop_mask, (self.photo_size, self.photo_size), interpolation=cv2.INTER_NEAREST)

        if crop_mask is not None and np.any(crop_mask > 0):
            polyp_region = crop_mask > 0
            kernel_size = max(3, int(round(0.08 * self.photo_size)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(polyp_region.astype(np.uint8), kernel, iterations=1)
            ring_region = (dilated > 0) & (~polyp_region)
            if not np.any(ring_region):
                _, ring_region = self._pseudo_ellipse_regions(self.photo_size, self.photo_size)
        else:
            polyp_region, ring_region = self._pseudo_ellipse_regions(self.photo_size, self.photo_size)

        if not np.any(polyp_region):
            polyp_region, ring_region = self._pseudo_ellipse_regions(self.photo_size, self.photo_size)

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_l = lab[:, :, 0]

        if np.any(ring_region):
            gray_bg = float(np.median(gray[ring_region]))
            lab_bg = float(np.median(lab_l[ring_region]))
        else:
            bg_region = ~polyp_region
            if not np.any(bg_region):
                bg_region = np.ones_like(polyp_region, dtype=bool)
            gray_bg = float(np.median(gray[bg_region]))
            lab_bg = float(np.median(lab_l[bg_region]))
        gray_bg = max(gray_bg, 1.0)
        lab_bg = max(lab_bg, 1.0)

        gray_norm = np.clip(gray / gray_bg, 0.0, 4.0)
        lab_norm = np.clip(lab_l / lab_bg, 0.0, 4.0)

        v = hsv[:, :, 2] / 255.0
        s = hsv[:, :, 1] / 255.0
        v_max = float(v[polyp_region].max()) if np.any(polyp_region) else 1.0
        specular = (v > (self.photo_v_thresh * max(v_max, 1e-6))) & (s < self.photo_s_thresh) & polyp_region

        channels = [gray_norm.astype(np.float32), lab_norm.astype(np.float32), specular.astype(np.float32)]
        if self.photo_include_rgb_ratio:
            rgb_f = crop_rgb.astype(np.float32)
            denom = np.maximum(np.sum(rgb_f, axis=2, keepdims=True), 1.0)
            rgb_ratio = rgb_f / denom
            channels.extend([rgb_ratio[:, :, 0], rgb_ratio[:, :, 1], rgb_ratio[:, :, 2]])

        photo = np.stack(channels, axis=0).astype(np.float32)
        if photo.shape[0] > self.photo_channels:
            photo = photo[:self.photo_channels]
        elif photo.shape[0] < self.photo_channels:
            pad = np.zeros((self.photo_channels - photo.shape[0], self.photo_size, self.photo_size), dtype=np.float32)
            photo = np.concatenate([photo, pad], axis=0)
        return torch.from_numpy(photo)

    def _build_temporal_groups(self):
        group_by = self.temporal_cfg.get('group_by', 'video_id')
        mode = self.temporal_cfg.get('mode', 'single')
        clip_len = int(self.temporal_cfg.get('clip_len', 8))
        stride = int(self.temporal_cfg.get('stride', clip_len))
        pad_short = self.temporal_cfg.get('pad_short', 'repeat_last')
        cap_strategy = self.temporal_cfg.get('cap_strategy', 'uniform')

        cap_key = f"max_windows_per_group_{self.split}"
        max_windows = self.temporal_cfg.get(cap_key, self.temporal_cfg.get('max_windows_per_group', None))
        if max_windows is not None:
            max_windows = int(max_windows)

        split_offset = {'train': 1, 'val': 2, 'test': 3}.get(self.split, 0)
        rng = random.Random(self.data_config.get('split_seed', 42) + split_offset)

        groups = {}
        for idx, sample in enumerate(self.samples):
            gid = sample.get(group_by, None)
            if gid is None:
                gid = sample.get('video_id', 'unknown')
            groups.setdefault(str(gid), []).append(idx)

        clip_groups = []
        clip_labels = []

        for gid, indices in groups.items():
            def _sort_key(i):
                s = self.samples[i]
                return (s.get('depth_idx', 0), str(s.get('image_name', '')))

            indices = sorted(indices, key=_sort_key)
            if not indices:
                continue

            labels_all = [self.samples[i]['label'] for i in indices]
            group_label = self._aggregate_labels(labels_all)

            if mode == 'single':
                clip_groups.append({'group_id': gid, 'indices': indices, 'label': group_label, 'is_window': False})
                clip_labels.append(group_label)
                continue

            if mode != 'sliding':
                raise ValueError(f"Unknown temporal mode: {mode}")

            windows = []
            n = len(indices)
            if n < clip_len:
                if pad_short == 'drop':
                    continue
                window = indices + [indices[-1]] * (clip_len - n)
                windows = [window]
            else:
                for start in range(0, n - clip_len + 1, stride):
                    windows.append(indices[start:start + clip_len])

            if not windows:
                continue

            if max_windows is not None and len(windows) > max_windows:
                if cap_strategy == 'uniform':
                    if max_windows <= 1:
                        windows = [windows[0]]
                    else:
                        sel = np.linspace(0, len(windows) - 1, max_windows).round().astype(int).tolist()
                        windows = [windows[i] for i in sel]
                elif cap_strategy == 'random':
                    windows = rng.sample(windows, max_windows)
                else:
                    windows = windows[:max_windows]

            for window in windows:
                labels_w = [self.samples[i]['label'] for i in window]
                label_w = self._aggregate_labels(labels_w)
                clip_groups.append({'group_id': gid, 'indices': window, 'label': label_w, 'is_window': True})
                clip_labels.append(label_w)

        return clip_groups, clip_labels

    def _aggregate_labels(self, labels: List[int]) -> int:
        mode = self.temporal_cfg.get('label_mode', 'majority')
        if not labels:
            return 0
        if mode == 'first':
            return int(labels[0])
        if mode == 'max':
            return int(max(labels))
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[counts.argmax()])

    def _select_clip_indices(self, indices: List[int]) -> List[int]:
        clip_len = int(self.temporal_cfg.get('clip_len', 8))
        if clip_len <= 0:
            return indices

        sampling_train = self.temporal_cfg.get('sampling_train', 'random')
        sampling_eval = self.temporal_cfg.get('sampling_eval', 'uniform')
        if getattr(self, '_temporal_force_eval_sampling', False):
            sampling = sampling_eval
        else:
            sampling = sampling_train if self.split == 'train' else sampling_eval

        n = len(indices)
        if n == 0:
            return []

        if sampling == 'uniform':
            if n >= clip_len:
                import numpy as _np
                sel = _np.linspace(0, n - 1, clip_len).round().astype(int).tolist()
                return [indices[i] for i in sel]
            out = indices + [indices[-1]] * (clip_len - n)
            return out

        if sampling == 'center':
            if n >= clip_len:
                start = max(0, (n - clip_len) // 2)
                return indices[start:start + clip_len]
            out = indices + [indices[-1]] * (clip_len - n)
            return out

        # random contiguous segment
        if n >= clip_len:
            start = random.randint(0, n - clip_len)
            return indices[start:start + clip_len]
        out = indices[:]
        while len(out) < clip_len:
            out.append(random.choice(indices))
        return out

    @staticmethod
    def _compute_pps_map(depth_np: np.ndarray, fx: float, fy: float,
                         cx: float, cy: float) -> np.ndarray:
        """Compute dense PPS (Photometric Point-pair Similarity) map from depth.

        Replicates the logic from augment_photometry_with_pps.py but returns
        the full spatial map instead of summary statistics.
        """
        EPS = 1e-8
        h, w = depth_np.shape
        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing='ij',
        )
        z = depth_np
        x = (xx - cx) * z / max(float(fx), EPS)
        y = (yy - cy) * z / max(float(fy), EPS)
        positions = np.stack([x, y, z], axis=0)  # [3, H, W]

        p_u = np.gradient(positions, axis=2)
        p_v = np.gradient(positions, axis=1)
        n = np.cross(
            np.moveaxis(p_v, 0, -1),
            np.moveaxis(p_u, 0, -1),
        )  # [H, W, 3]
        n_norm = np.linalg.norm(n, axis=2, keepdims=True)
        n = n / np.clip(n_norm, EPS, None)
        normals = np.moveaxis(n, -1, 0)  # [3, H, W]

        light = -positions
        dist2 = np.sum(light * light, axis=0, keepdims=True)
        attenuation = 1.0 / np.clip(dist2, EPS, None)
        attenuation = attenuation / max(float(np.max(attenuation)), EPS)
        light_norm = light / np.clip(
            np.linalg.norm(light, axis=0, keepdims=True), EPS, None
        )
        cos_term = np.sum(normals * light_norm, axis=0, keepdims=True)
        pps = (attenuation * cos_term)[0]
        return pps.astype(np.float32, copy=False)

    def _load_single_sample(self, sample: Dict):
        # Load RGB image
        rgb = Image.open(sample['image_path']).convert('RGB')
        orig_w, orig_h = rgb.size
        true_orig_w, true_orig_h = orig_w, orig_h  # save before alignment overwrites

        # Load depth map
        needs_depth = self._needs_depth or self.input_channels >= 4
        if needs_depth:
            depth_path = self._find_depth_path(sample)
            if depth_path and depth_path.exists():
                depth = self._load_depth(depth_path)
            else:
                # If no depth, create dummy depth (zeros)
                depth = np.zeros((rgb.size[1], rgb.size[0]), dtype=np.float32)
        else:
            # RGB-only: use dummy depth for transforms (not returned)
            depth = np.zeros((rgb.size[1], rgb.size[0]), dtype=np.float32)

        # Align RGB to match Depth dimensions
        # Depth is [H, W], RGB is PIL Image (W, H)
        dh, dw = depth.shape
        rw, rh = rgb.size
        if (rh != dh) or (rw != dw):
            if dh == dw:
                # Square depth (e.g. PPSNet 384x384): center crop on short side, then resize
                short_side = min(rw, rh)
                left = (rw - short_side) // 2
                top = (rh - short_side) // 2
                rgb = rgb.crop((left, top, left + short_side, top + short_side))
                rgb = rgb.resize((dw, dh), resample=Image.BILINEAR)
            else:
                # Non-square depth (e.g. MetricCol 256x320): direct resize
                rgb = rgb.resize((dw, dh), resample=Image.BILINEAR)
            # Update orig sizes for mask logic
            orig_w, orig_h = dw, dh

        # Compute PPS map before converting depth to tensor (if needed)
        pps_map_np = None
        if self._needs_pps:
            valid = np.isfinite(depth) & (depth > 0)
            if valid.sum() >= 16:
                fill = float(np.median(depth[valid]))
                depth_filled = np.where(valid, depth, fill).astype(np.float32)
                pps_map_np = self._compute_pps_map(
                    depth_filled, self._pps_fx, self._pps_fy,
                    self._pps_cx, self._pps_cy,
                )
                # Normalize PPS map to roughly [-1, 1] range
                pps_finite = pps_map_np[np.isfinite(pps_map_np)]
                if pps_finite.size > 0:
                    pps_map_np = np.nan_to_num(pps_map_np, nan=0.0)
            if pps_map_np is None:
                pps_map_np = np.zeros_like(depth)

        # Build mask (if enabled) before transforms
        mask = None
        if self.use_mask or self.return_mask or self.return_unmasked or self._needs_mask_channel or self._needs_masked_rgb:
            mask_path = None if self.force_bbox_mask else self._find_mask_path(sample)
            if mask_path and mask_path.exists():
                mask_img = Image.open(mask_path).convert('L')
                mask = np.array(mask_img)
            else:
                bbox = sample.get('bbox')
                if bbox:
                    if needs_depth:
                        # Depth was loaded: map bbox through depth-alignment transform
                        mask = self._create_depth_mask_from_bbox(
                            bbox=bbox,
                            depth_shape=depth.shape,
                            orig_w=true_orig_w,
                            orig_h=true_orig_h,
                        )
                    else:
                        # No real depth loaded (dummy at orig resolution):
                        # create mask directly in original image coordinates
                        mask = self._create_mask_from_bbox(
                            bbox, (orig_h, orig_w)
                        )

        # Convert to tensors
        rgb = np.array(rgb).astype(np.float32) / 255.0  # [H, W, 3]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]

        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != depth.shape[1:]:
                mask = np.array(Image.fromarray(mask).resize((depth.shape[2], depth.shape[1]), resample=Image.NEAREST))
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)

        # Apply transforms
        if self.transform:
            if mask is None:
                rgb, depth = self.transform(rgb, depth)
            else:
                rgb, depth, mask = self.transform(rgb, depth, mask)

        if mask is not None and self.min_mask_px is not None and not self._needs_mask_channel and not self._needs_masked_rgb:
            if mask.sum().item() < float(self.min_mask_px):
                mask = None

        # Ensure mask key can be collated when requested, even if missing
        if mask is None and (self.return_mask or self._needs_mask_channel or self._needs_masked_rgb):
            mask = torch.zeros_like(depth)

        # Apply scenario-specific processing
        if self.scenario_id == 2 and needs_depth:
            depth = self._apply_depth_rescaling(depth, sample['image_name'])

        # Apply copy-paste augmentation for Scenario 3 (only during training)
        label = sample['label']  # Get label early
        if self.scenario_id == 3 and self.split == 'train' and self.copy_paste_aug is not None:
            rgb_clamped = torch.clamp(rgb, 0, 1)
            rgb_np = (rgb_clamped.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
            depth_np = depth.squeeze(0).cpu().numpy()  # [H, W]
            rgb_np, depth_np, label = self.copy_paste_aug(rgb_np, depth_np, label)
            rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0  # [3, H, W]
            depth = torch.from_numpy(depth_np).unsqueeze(0).float()  # [1, H, W]
            mask = None

        # Normalize depth (after rescaling if applicable)
        if needs_depth:
            depth = self._normalize_depth(depth)

        # Normalize RGB
        rgb = self._normalize_rgb(rgb)

        # Resize PPS map to match transformed depth size
        pps_tensor = None
        if self._needs_pps:
            pps_tensor = torch.from_numpy(pps_map_np).unsqueeze(0)  # [1, H, W]
            if pps_tensor.shape[1:] != depth.shape[1:]:
                pps_tensor = torch.nn.functional.interpolate(
                    pps_tensor.unsqueeze(0),
                    size=depth.shape[1:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

        # Preserve unmasked input for auxiliary losses if requested
        image_unmasked = None
        if mask is not None and self.return_unmasked:
            if needs_depth:
                image_unmasked = torch.cat([rgb, depth], dim=0)
            else:
                image_unmasked = rgb

        # Apply mask to inputs (after normalization)
        if mask is not None and self.use_mask:
            mask_bin = (mask > 0.5).float()
            if needs_depth and self.mask_depth:
                depth = depth * mask_bin
            if self.mask_rgb:
                rgb = rgb * mask_bin

        # Combine channels based on input_composition or legacy input_channels
        if self.input_composition:
            channels = []
            for comp in self.input_composition:
                if comp == 'rgb':
                    channels.append(rgb)  # [3, H, W]
                elif comp == 'masked_rgb':
                    # BSENet-style masked RGB (3ch): keep ROI, zero elsewhere.
                    mask_bin = (mask > 0.5).float() if mask is not None else torch.zeros_like(depth)
                    channels.append(rgb * mask_bin)  # [3, H, W]
                elif comp == 'mask':
                    channels.append(mask if mask is not None else torch.zeros_like(depth))  # [1, H, W]
                elif comp == 'depth':
                    channels.append(depth)  # [1, H, W]
                elif comp == 'pps':
                    channels.append(pps_tensor if pps_tensor is not None else torch.zeros_like(depth))  # [1, H, W]
                else:
                    raise ValueError(f"Unknown input_composition component: {comp}")
            image = torch.cat(channels, dim=0)
        elif self.input_channels >= 4:
            image = torch.cat([rgb, depth], dim=0)  # [4, H, W]
        else:
            image = rgb  # [3, H, W]

        return image, label, mask, image_unmasked

    def _get_temporal_item(self, idx: int):
        group = self.clip_groups[idx]
        frame_indices = group['indices']
        if not group.get('is_window', False):
            frame_indices = self._select_clip_indices(frame_indices)
        frames = []
        labels = []
        masks = []
        unmasked_frames = []
        image_names = []
        features = []
        photo_imgs = []
        l_px = []
        size_vals = []
        lesion_ids = []
        for sample_idx in frame_indices:
            sample = self.samples[sample_idx]
            image, label, mask, image_unmasked = self._load_single_sample(sample)
            frames.append(image)
            labels.append(label)
            if self.return_mask and mask is not None:
                masks.append(mask)
            if self.return_unmasked and image_unmasked is not None:
                unmasked_frames.append(image_unmasked)
            image_names.append(sample.get('image_name'))
            size_vals.append(float(sample.get('size_mm', 0.0)))
            lesion_ids.append(str(sample.get('lesion_id', sample.get('video_id', group['group_id']))))
            if self.photometry_features:
                features.append(self._get_sample_photometry_tensor(sample))
            if self.photo_image_enabled:
                photo_imgs.append(self._get_sample_photo_tensor(sample))
            l_px.append(self._get_sample_lpx(sample))

        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        label = group.get('label', None)
        if label is None:
            label = self._aggregate_labels(labels)

        first_name = image_names[0] if image_names else group['group_id']
        size_mm = float(size_vals[0]) if size_vals else 0.0
        output = {
            'image': clip,
            'label': label,
            'image_name': first_name,
            'size_mm': size_mm,
            'video_id': group['group_id'],
            'lesion_id': lesion_ids[0] if lesion_ids else group['group_id'],
            'l_px': torch.tensor(l_px, dtype=torch.float32),
        }
        if self.photometry_features and features:
            output['features'] = torch.stack(features, dim=0)
        if self.photo_image_enabled and photo_imgs:
            output['photo_img'] = torch.stack(photo_imgs, dim=0)
        if self.return_mask and masks:
            output['mask'] = torch.stack(masks, dim=0)
        if self.return_unmasked and unmasked_frames:
            output['image_unmasked'] = torch.stack(unmasked_frames, dim=0)
        return output

    def _find_depth_path(self, sample: Dict) -> Optional[Path]:
        """Find depth map path for a sample (handles all datasets including augmented)"""
        dataset = sample['dataset']
        depth_idx = sample.get('depth_idx', None)

        if dataset == 'polyp_size':
            # Use depth_idx for Polyp_Size
            if depth_idx is not None:
                # Try depth_maps_depthanything first
                depth_path = self.depth_dir / 'polyp_size' / f"{depth_idx:06d}.pt"
                if depth_path.exists():
                    return depth_path

                if self.depth_fallback:
                    # Fallback: try depth_maps_ppsnet (alternative depth source for Polyp_Size)
                    depth_ppsnet_dir = self.depth_dir.parent / 'depth_maps_ppsnet'
                    depth_path_alt = depth_ppsnet_dir / 'polyp_size' / f"{depth_idx:06d}.pt"
                    if depth_path_alt.exists():
                        return depth_path_alt

            # Fallback: try image name
            image_name = sample['image_name']
            name_no_ext = Path(image_name).stem
            depth_path = self.depth_dir / 'polyp_size' / f"{name_no_ext}.pt"
            if depth_path.exists():
                return depth_path

            if self.depth_fallback:
                # Fallback for image name with ppsnet
                depth_ppsnet_dir = self.depth_dir.parent / 'depth_maps_ppsnet'
                depth_path_alt = depth_ppsnet_dir / 'polyp_size' / f"{name_no_ext}.pt"
                if depth_path_alt.exists():
                    return depth_path_alt

        elif dataset == 'real_colon':
            # Use depth_idx for Real-Colon (sequential numbering)
            if depth_idx is not None:
                if self.depth_source == 'metriccol':
                    # MetricCol depth: stored in depth_maps_metriccol/real_colon/
                    metriccol_dir = self.depth_dir.parent / 'depth_maps_metriccol' / 'real_colon'
                    depth_path = metriccol_dir / f"{depth_idx:06d}.pt"
                    if depth_path.exists():
                        return depth_path
                # PPSNet (default or fallback)
                depth_path = self.depth_dir / 'real_colon' / f"{depth_idx:06d}.pt"
                if depth_path.exists():
                    return depth_path

        elif dataset == 'sun':
            # Use depth_idx for SUN (sequential numbering in sun_labels CSV)
            if depth_idx is not None:
                sun_path = self.data_config.get('sun_path')

                if self.depth_source == 'metriccol' and sun_path:
                    # MetricCol depth: stored in sun_path/depth_maps_metriccol/
                    depth_path = Path(sun_path) / 'depth_maps_metriccol' / f"{int(depth_idx):06d}.pt"
                    if depth_path.exists():
                        return depth_path

                # PPSNet (default or fallback)
                # Try relative to depth_maps_dir first (if subfolder 'sun' exists)
                depth_path = self.depth_dir / 'sun' / f"{int(depth_idx):06d}.pt"
                if depth_path.exists():
                    return depth_path
                
                # Fallback: check SUN dataset folder directly for depth_maps_ppsnet
                if sun_path:
                    depth_path_alt = Path(sun_path) / 'depth_maps_ppsnet' / f"{int(depth_idx):06d}.pt"
                    if depth_path_alt.exists():
                        return depth_path_alt


        elif dataset == 'augmented':
            # Augmented samples: depth maps are in augmented_copy_paste/depth_maps/
            # The depth_idx in the merged CSV points to the augmented depth index
            # We need to calculate the augmented sample index
            if depth_idx is not None:
                # Get augmented data directory from config
                augmented_dir = self.data_config.get('augmented_data_dir')
                if augmented_dir:
                    augmented_dir = Path(augmented_dir)
                else:
                    # Default location
                    augmented_dir = self.depth_dir.parent / 'augmented_copy_paste'

                # The merged CSV assigns depth_idx starting from max_original_idx + 1
                # But augmented depth maps are numbered 000000.pt, 000001.pt, etc.
                # We need to compute the offset by reading the original CSV
                real_colon_labels = self.data_config.get('real_colon_labels')
                if 'augmentation' in str(real_colon_labels):
                    # This is the merged CSV, compute max from original
                    # Load original CSV to get max depth_idx
                    original_csv = Path(str(real_colon_labels).replace('with_augmentation', 'binary_5mm'))
                    if original_csv.exists():
                        try:
                            df_orig = pd.read_csv(original_csv)
                            max_original_depth_idx = int(df_orig['depth_idx'].max())
                        except:
                            # Fallback if CSV read fails
                            max_original_depth_idx = 357263  # Known max for Real-Colon
                    else:
                        max_original_depth_idx = 357263  # Known max for Real-Colon

                    # Compute augmented index: aug_idx = depth_idx - max_depth_idx - 1
                    aug_idx = depth_idx - max_original_depth_idx - 1

                    depth_path = augmented_dir / 'depth_maps' / f"{aug_idx:06d}.pt"
                    if depth_path.exists():
                        return depth_path

        return None

    def _find_mask_path(self, sample: Dict) -> Optional[Path]:
        """Find segmentation mask path for a sample (handles both datasets)"""
        dataset = sample['dataset']

        if dataset == 'polyp_size':
            # Polyp_Size uses segmentation masks
            if not self.seg_dir:
                return None

            image_name = sample['image_name']
            name_no_ext = Path(image_name).stem

            # Try various naming patterns
            for pattern in [
                image_name,
                f"{name_no_ext}.png",
                f"{name_no_ext}_mask.png"
            ]:
                mask_path = self.seg_dir / pattern
                if mask_path.exists():
                    return mask_path

            return None

        elif dataset in {'real_colon', 'augmented'}:
            mask_root = self.real_colon_seg_dir if self.real_colon_seg_dir else self.seg_dir
            if not mask_root:
                return None
            image_name = sample.get('image_name', '')
            stem = Path(image_name).stem
            for pattern in (
                image_name,
                f"{stem}.png",
                f"{stem}_binary.png",
                f"{stem}_mask.png",
            ):
                mask_path = mask_root / pattern
                if mask_path.exists():
                    return mask_path
            return None

        elif dataset == 'sun':
            if self.sun_seg_dir:
                image_name = sample.get('image_name', '')
                stem = Path(image_name).stem
                # SUN labels store absolute RGB paths; never prepend those into the mask dir.
                patterns = [
                    f"{stem}_binary.png",
                    f"{stem}_mask.png",
                    f"{stem}.png",
                ]
                if image_name and not Path(image_name).is_absolute():
                    patterns.append(image_name)
                for pattern in patterns:
                    mask_path = self.sun_seg_dir / pattern
                    if mask_path.exists():
                        return mask_path
            # Fallback to SUN GT-style path next to Frame/ folders
            return self._infer_sun_mask_path(Path(sample.get('image_path', '')))

        return None

    def _create_mask_from_bbox(self, bbox: Dict, image_size: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from bounding box"""
        height, width = image_size
        mask = np.zeros((height, width), dtype=np.uint8)

        xmin, ymin, xmax, ymax = self._scale_bbox_xyxy(
            float(bbox['xmin']),
            float(bbox['ymin']),
            float(bbox['xmax']),
            float(bbox['ymax']),
            float(width),
            float(height),
        )
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(width, int(xmax))
        ymax = min(height, int(ymax))

        mask[ymin:ymax, xmin:xmax] = 255

        return mask

    def _create_depth_mask_from_bbox(
        self,
        bbox: Dict,
        depth_shape: Tuple[int, int],
        orig_w: float,
        orig_h: float,
    ) -> np.ndarray:
        """Create bbox mask aligned to depth map coordinates."""
        dh, dw = depth_shape
        resize_target = self.mask_resize_target
        if orig_w <= 0 or orig_h <= 0:
            return self._create_mask_from_bbox(bbox, (dh, dw))

        if dh == dw:
            # Square depth (e.g. PPSNet): uses center-crop alignment
            scale = resize_target / min(orig_h, orig_w)
            resized_w = orig_w * scale
            resized_h = orig_h * scale
            crop_x_offset = (resized_w - resize_target) / 2.0
            crop_y_offset = (resized_h - resize_target) / 2.0
            depth_scale = dw / resize_target

            bbox_x1 = (float(bbox['xmin']) * scale - crop_x_offset) * depth_scale
            bbox_y1 = (float(bbox['ymin']) * scale - crop_y_offset) * depth_scale
            bbox_x2 = (float(bbox['xmax']) * scale - crop_x_offset) * depth_scale
            bbox_y2 = (float(bbox['ymax']) * scale - crop_y_offset) * depth_scale
        else:
            # Non-square depth (e.g. MetricCol): uses direct resize alignment (matching line 1250)
            x_scale = dw / orig_w
            y_scale = dh / orig_h
            bbox_x1 = float(bbox['xmin']) * x_scale
            bbox_y1 = float(bbox['ymin']) * y_scale
            bbox_x2 = float(bbox['xmax']) * x_scale
            bbox_y2 = float(bbox['ymax']) * y_scale

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self._scale_bbox_xyxy(
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, float(dw), float(dh)
        )
        x1 = max(0, int(bbox_x1))
        y1 = max(0, int(bbox_y1))
        x2 = min(dw, int(bbox_x2))
        y2 = min(dh, int(bbox_y2))

        mask = np.zeros((dh, dw), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        return mask

    def _init_copy_paste_augmentation(self, all_samples: List[Dict], config: Dict):
        """Initialize copy-paste augmentation for Scenario 3"""
        try:
            from src.utils.copy_paste import CopyPasteAugmentation
        except ImportError:
            print("Warning: Could not import CopyPasteAugmentation. Copy-paste will be disabled.")
            return None

        # Get config parameters
        aug_config = config.get('augmentation', {}).get('copy_paste', {})
        cp_config = config.get('copy_paste', {})
        
        # Merge configs (copy_paste section takes precedence)
        depth_threshold = aug_config.get(
            'depth_similarity_threshold',
            cp_config.get('depth_similarity_threshold', 0.1)
        )
        min_paste_size = aug_config.get(
            'min_paste_size',
            cp_config.get('min_paste_size', 50)
        )
        max_paste_size = aug_config.get(
            'max_paste_size',
            cp_config.get('max_paste_size', 300)
        )
        blend_kernel_size = aug_config.get(
            'blend_kernel_size',
            cp_config.get('blend_kernel_size', 15)
        )
        erosion_size = aug_config.get(
            'border_erosion',
            aug_config.get(
                'erosion_size',
                cp_config.get('border_erosion', cp_config.get('erosion_size', 5))
            )
        )
        use_color_transfer = aug_config.get(
            'use_color_transfer',
            cp_config.get('use_color_transfer', True)
        )
        apply_prob = aug_config.get('apply_prob', cp_config.get('apply_prob', 0.5))

        # Build source and target pools from all samples
        source_pool = []
        target_pool = []

        for sample in all_samples:
            image_name = sample['image_name']
            dataset = sample['dataset']
            depth_path = self._find_depth_path(sample)

            # Source pool: frames with polyps (label 1 or 2)
            if sample['label'] in [1, 2]:  # Has polyp
                if depth_path and depth_path.exists():
                    # Handle both mask-based (Polyp_Size) and bbox-based (Real-Colon)
                    if dataset == 'polyp_size':
                        mask_path = self._find_mask_path(sample)
                        if mask_path and mask_path.exists():
                            source_pool.append({
                                'image_name': image_name,
                                'rgb_path': str(sample['image_path']),
                                'depth_path': str(depth_path),
                                'mask_path': str(mask_path),
                                'label': sample['label'],
                                'dataset': dataset
                            })
                    elif dataset == 'real_colon':
                        # Store bbox info instead of mask path
                        source_pool.append({
                            'image_name': image_name,
                            'rgb_path': str(sample['image_path']),
                            'depth_path': str(depth_path),
                            'bbox': sample['bbox'],
                            'image_width': sample['image_width'],
                            'image_height': sample['image_height'],
                            'label': sample['label'],
                            'dataset': dataset
                        })

            # Target pool: frames without polyps (label 0)
            elif sample['label'] == 0:
                if depth_path and depth_path.exists():
                    target_pool.append({
                        'image_name': image_name,
                        'rgb_path': str(sample['image_path']),
                        'depth_path': str(depth_path),
                        'label': sample['label'],
                        'dataset': dataset
                    })

        if len(source_pool) == 0 or len(target_pool) == 0:
            print(f"Warning: Copy-paste pools are insufficient. Source: {len(source_pool)}, Target: {len(target_pool)}")
            print("Copy-paste augmentation will be disabled.")
            return None

        # Create augmenter
        augmenter = CopyPasteAugmentation(
            source_pool=source_pool,
            target_pool=target_pool,
            depth_similarity_threshold=depth_threshold,
            min_paste_size=min_paste_size,
            max_paste_size=max_paste_size,
            blend_kernel_size=blend_kernel_size,
            erosion_size=erosion_size,
            use_color_transfer=use_color_transfer,
            p=apply_prob
        )

        print(f"\n✅ Initialized copy-paste augmentation (Scenario 3):")
        print(f"   Source pool (with polyps): {len(source_pool)} samples")
        print(f"   Target pool (no polyps): {len(target_pool)} samples")
        print(f"   Depth similarity threshold: {depth_threshold}")
        print(f"   Apply probability: {apply_prob}")

        return augmenter

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        """Load depth map"""
        depth = torch.load(depth_path)
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if depth.ndim == 3:
            # Handle [1,H,W] or [H,W,1] or multi-channel tensors robustly
            if depth.shape[0] == 1:
                depth = depth.squeeze(0)
            elif depth.shape[2] == 1:
                depth = depth.squeeze(2)
            else:
                # Fallback: take first channel if depth is multi-channel
                if depth.shape[0] <= 4:
                    depth = depth[0]
                elif depth.shape[2] <= 4:
                    depth = depth[..., 0]
                else:
                    raise ValueError(f"Unexpected depth shape: {depth.shape} in {depth_path}")
        return depth.astype(np.float32)

    def _apply_depth_rescaling(self, depth: torch.Tensor, image_name: str) -> torch.Tensor:
        """Apply depth rescaling for Scenario 2"""
        if self.scale_factors and image_name in self.scale_factors:
            scale = self.scale_factors[image_name]
            # Only apply if within confidence threshold
            if 'confidence' not in self.config.get('depth_rescaling', {}) or True:
                depth = depth * scale
        return depth

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth map using dataset-wide statistics"""
        method = self.config.get('augmentation', {}).get('depth_norm_method', 'zscore')

        if method in ('none', 'identity', 'raw'):
            # No normalization — preserve absolute metric depth values
            return depth
        elif method == 'zscore':
            # Use dataset-wide stats if available (preferred)
            depth_stats = self.config.get('depth_stats', {})
            if depth_stats and 'mean' in depth_stats and 'std' in depth_stats:
                # Use precomputed dataset-wide statistics
                mean = depth_stats['mean']
                std = depth_stats['std']
            else:
                # Fallback to per-image normalization (old behavior)
                # This is less stable but works if stats not computed
                mean = depth.mean().item()
                std = depth.std().item()
            depth = (depth - mean) / (std + 1e-8)
        elif method == 'minmax':
            # Map [0, 1] to [-1, 1]
            depth = (depth * 2) - 1

        return depth

    def _normalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Normalize RGB with ImageNet stats"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (rgb - mean) / std


def create_dataloaders(
    config: Dict,
    scenario_id: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None
):
    """
    Create train/val/test dataloaders

    Args:
        config: Configuration dictionary
        scenario_id: Optional override for scenario selection.
            If None, falls back to config['scenario']['id'] and defaults to 1.
        batch_size: Optional override for dataloader batch size.
        num_workers: Optional override for dataloader worker count.
        pin_memory: Optional override for pin_memory flag.

    Returns:
        Dict of dataloaders
    """
    from src.utils.transforms import get_train_transforms, get_val_transforms

    # Determine scenario from override or config
    if scenario_id is None:
        scenario_id = config.get('scenario', {}).get('id', 1)

    # Create transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    # Create datasets
    train_dataset = RGBDPolypDataset(config, split='train', transform=train_transform, scenario_id=scenario_id)
    val_dataset = RGBDPolypDataset(config, split='val', transform=val_transform, scenario_id=scenario_id)
    test_dataset = RGBDPolypDataset(config, split='test', transform=val_transform, scenario_id=scenario_id)

    # Propagate photometry standardization stats from train to val/test
    if train_dataset.photometry_mean is not None:
        val_dataset.set_photometry_stats(train_dataset.photometry_mean, train_dataset.photometry_std)
        test_dataset.set_photometry_stats(train_dataset.photometry_mean, train_dataset.photometry_std)

    batch_size = batch_size or config['data']['batch_size']
    num_workers = num_workers if num_workers is not None else config['data']['num_workers']
    pin_memory = pin_memory if pin_memory is not None else config['data'].get('pin_memory', True)

    # Create samplers for class balancing
    train_sampler = None
    if config['training'].get('use_weighted_sampler', False):
        train_sampler = create_weighted_sampler(train_dataset, config)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def compute_depth_statistics(dataloader, device='cpu', max_samples=200):
    """
    Compute dataset-wide depth statistics for normalization
    
    Uses a small sample (first N batches) to avoid IO overload.
    Much faster than loading all data.
    
    Args:
        dataloader: DataLoader to compute stats from
        device: Device to use for computation
        max_samples: Maximum number of samples to use (default: 200)
    
    Returns:
        (mean, std) tuple
    """
    from tqdm import tqdm
    
    all_depths = []
    sample_count = 0
    
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    
    if max_samples and max_samples < dataset_size:
        # Use first N batches - simple and fast
        num_batches = (max_samples + batch_size - 1) // batch_size
        print(f"Computing depth stats from first {num_batches} batches ({max_samples} samples) to reduce IO load...")
        print(f"  (Total dataset: {dataset_size} samples)")
        
        for idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=num_batches)):
            # Extract depth from RGBD (last channel)
            images = batch['image']
            if images.dim() == 5:
                # [B, T, C, H, W] -> [B*T, 1, H, W]
                depth = images[:, :, 3:4, :, :].reshape(-1, 1, images.size(-2), images.size(-1))
            else:
                # [B, C, H, W]
                depth = images[:, 3:4, :, :]  # [B, 1, H, W]
            
            all_depths.append(depth.cpu())
            sample_count += depth.shape[0]
            
            if sample_count >= max_samples:
                break
            if idx >= num_batches - 1:
                break
    else:
        # Use all samples (not recommended - very slow!)
        print(f"WARNING: Computing stats from ALL {dataset_size} samples - this may be very slow!")
        print("Consider setting depth_stats_max_samples to a smaller value (e.g., 200)")
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = batch['image']
            if images.dim() == 5:
                depth = images[:, :, 3:4, :, :].reshape(-1, 1, images.size(-2), images.size(-1))
            else:
                depth = images[:, 3:4, :, :]  # [B, 1, H, W]
            all_depths.append(depth.cpu())
            sample_count += depth.shape[0]
    
    if not all_depths:
        raise ValueError("No depth samples collected. Check dataloader.")
    
    # Concatenate all depth values
    all_depths = torch.cat(all_depths)
    mean = all_depths.mean().item()
    std = all_depths.std().item()
    
    print(f"✓ Depth statistics: mean={mean:.4f}, std={std:.4f} (from {sample_count} samples)")
    return mean, std


def create_weighted_sampler(dataset: RGBDPolypDataset, config: Dict):
    """Create weighted random sampler for class balancing"""
    threshold = config['training'].get('weighted_sampler_threshold', 0.20)

    # Check if minority class is below threshold
    min_prior = min(dataset.class_priors.values())
    if min_prior >= threshold:
        print(f"Minority class prior ({min_prior:.3f}) >= threshold ({threshold}), not using weighted sampler")
        return None

    # Create sample weights
    if dataset.temporal_enabled and dataset.clip_labels is not None:
        labels = list(dataset.clip_labels)
    else:
        labels = [s['label'] for s in dataset.samples]
    class_weights_list = [dataset.class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=class_weights_list,
        num_samples=len(dataset),
        replacement=True
    )

    print(f"Using WeightedRandomSampler (minority class prior: {min_prior:.3f})")
    return sampler


# Example usage
if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # Load config
    # config_path = Path(__file__).parent.parent.parent / "config" / "scenario1_baseline.yaml"
    # config_path = Path(__file__).parent.parent.parent / "config" / "scenario2_depth_rescaling.yaml"
    config_path = Path(__file__).parent.parent.parent / "config" / "scenario3_copy_paste.yaml"
    print(f"Loading config: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create dataloaders
    dataloaders = create_dataloaders(config)

    print("\nDataloader summary:")
    for name, loader in dataloaders.items():
        if 'dataset' not in name:
            print(f"{name}: {len(loader)} batches")

    # Test loading a batch
    print("\nTesting batch loading...")
    batch = next(iter(dataloaders['train']))
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")  # Should be [B, 4, H, W]
    print(f"Label shape: {batch['label'].shape}")  # Should be [B]
    print(f"Labels in batch: {batch['label'].tolist()}")
