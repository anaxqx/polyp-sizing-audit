"""Train diagnostic polyp-size probes."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.datasets.rgbd_dataset import create_dataloaders, compute_depth_statistics
from src.datasets.depth_only_dataset import create_depth_dataloaders
from src.models.resnet_rgb import create_resnet_rgb_from_config
from src.models.vit_rgbd import create_vit_rgbd_from_config
from src.models.bsenet import create_bsenet_from_config
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.utils.losses import create_loss_function


def _to_jsonable(obj):
    """Convert numpy/torch scalar types to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_jsonable(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj) and obj.numel() == 1:
        return obj.item()
    return obj


def _set_requires_grad(module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _unfreeze_classification_heads(model: torch.nn.Module) -> List[str]:
    """Unfreeze known classification heads and return names that were found."""
    head_attrs = ["head", "fc", "classifier"]
    unfrozen = []
    for attr in head_attrs:
        if hasattr(model, attr):
            _set_requires_grad(getattr(model, attr), True)
            unfrozen.append(attr)
    return unfrozen


def apply_finetune_config(model: torch.nn.Module, config: dict) -> dict:
    ft_cfg = config.get("model", {}).get("finetune", {}) or {}
    mode = str(ft_cfg.get("mode", "full")).lower()

    if mode in {"full", "all"}:
        return {"mode": "full", "trainable": "all"}

    # Freeze everything first
    _set_requires_grad(model, False)

    # Always unfreeze known heads, if present.
    unfrozen_heads = _unfreeze_classification_heads(model)

    details = {"mode": mode, "trainable": "head_only", "heads": unfrozen_heads}

    if mode in {"head_only", "head"}:
        return details

    if mode in {"last_n_blocks", "last_blocks"}:
        n_blocks = int(ft_cfg.get("last_n_blocks", 1))
        if n_blocks <= 0:
            print("Warning: finetune.last_n_blocks <= 0; training head only.")
            return details

        vit_module = getattr(model, "vit", None)
        vit_blocks = getattr(vit_module, "blocks", None)
        if vit_blocks is not None:
            for block in list(vit_blocks)[-n_blocks:]:
                _set_requires_grad(block, True)

            if ft_cfg.get("train_norm", True) and hasattr(vit_module, "norm"):
                _set_requires_grad(vit_module.norm, True)

            if ft_cfg.get("train_patch_embed", False) and hasattr(vit_module, "patch_embed"):
                _set_requires_grad(vit_module.patch_embed, True)

            if ft_cfg.get("train_pos_embed", False) and hasattr(vit_module, "pos_embed"):
                vit_module.pos_embed.requires_grad = True

            if ft_cfg.get("train_cls_token", False) and hasattr(vit_module, "cls_token"):
                vit_module.cls_token.requires_grad = True

            details = {
                "mode": "last_n_blocks",
                "backbone": "vit",
                "last_n_blocks": n_blocks,
                "train_norm": bool(ft_cfg.get("train_norm", True)),
                "train_patch_embed": bool(ft_cfg.get("train_patch_embed", False)),
                "train_pos_embed": bool(ft_cfg.get("train_pos_embed", False)),
                "train_cls_token": bool(ft_cfg.get("train_cls_token", False)),
                "heads": unfrozen_heads,
            }
            return details

        resnet_candidate = model
        if not all(hasattr(resnet_candidate, f"layer{i}") for i in range(1, 5)):
            backbone = getattr(model, "backbone", None)
            if backbone is not None and all(hasattr(backbone, f"layer{i}") for i in range(1, 5)):
                resnet_candidate = backbone

        if all(hasattr(resnet_candidate, f"layer{i}") for i in range(1, 5)):
            layers = [getattr(resnet_candidate, f"layer{i}") for i in range(1, 5)]
            for layer in layers[-n_blocks:]:
                _set_requires_grad(layer, True)
            if ft_cfg.get("train_stem", False):
                for stem_attr in ("conv1", "bn1"):
                    if hasattr(resnet_candidate, stem_attr):
                        _set_requires_grad(getattr(resnet_candidate, stem_attr), True)

            details = {
                "mode": "last_n_blocks",
                "backbone": "resnet",
                "last_n_blocks": n_blocks,
                "train_stem": bool(ft_cfg.get("train_stem", False)),
                "heads": unfrozen_heads,
            }
            return details

        print("Warning: no supported block structure found; training heads only.")
        return details

    raise ValueError(f"Unknown finetune mode: {mode}")


class Trainer:
    """Trainer for polyp size classification"""

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config['logging']['output_dir']) / config['logging']['experiment_name']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        self.scenario_id = config.get('scenario', {}).get('id', 1)
        self.arch = str(config['model'].get('architecture', 'vit_rgbd')).lower()
        supported_arches = {'photometry_mlp', 'bsenet', 'rgb_resnet', 'resnet_rgb', 'vit_rgbd', 'vit_rgb'}
        if self.arch not in supported_arches:
            raise ValueError(f"Unsupported architecture '{self.arch}'. Use one of {sorted(supported_arches)}")

        # Create dataloaders
        print("\nCreating dataloaders...")
        data_cfg = config.get('data', {})
        loader_mode = str(data_cfg.get('loader', '')).lower()
        use_depth_only_loader = loader_mode in {'depth', 'depth_only', 'bsenet'} or self.arch == 'bsenet'
        if (
            not use_depth_only_loader
            and self.arch in {'rgb_resnet', 'resnet_rgb'}
            and 'csv_path' in data_cfg
            and 'depth_dir' in data_cfg
        ):
            # Compatibility mode: allow ResNet training on BSENet-style depth configs.
            use_depth_only_loader = True
            print("Detected BSENet-style depth config with ResNet architecture; using depth-only dataloader compatibility mode.")

        if use_depth_only_loader:
            self.dataloaders = create_depth_dataloaders(config)
            # Normalize key names for the rest of Trainer code.
            datasets = self.dataloaders.get('datasets', {})
            self.dataloaders['train_dataset'] = datasets.get('train')
            self.dataloaders['val_dataset'] = datasets.get('val')
            self.dataloaders['test_dataset'] = datasets.get('test')
        else:
            self.dataloaders = create_dataloaders(config, scenario_id=self.scenario_id)
        self.train_loader = self.dataloaders['train']
        self.val_loader = self.dataloaders['val']

        # Compute depth statistics for normalization (if using zscore)
        depth_norm_method = config.get('augmentation', {}).get('depth_norm_method', 'zscore')
        compute_depth_stats = config.get('training', {}).get('compute_depth_stats', True)
        input_channels = int(config.get('data', {}).get('input_channels', config.get('model', {}).get('input_channels', 4)))
        
        if (not use_depth_only_loader) and input_channels == 4 and depth_norm_method == 'zscore' and not config.get('depth_stats'):
            if compute_depth_stats:
                print("\nComputing depth normalization statistics...")
                # Use a small subset by default to avoid IO overload (100-500 samples)
                max_samples = config.get('depth_stats_max_samples', 200)  # Default: 200 samples
                print(f"Using {max_samples} samples for depth statistics (to reduce IO load)")
                depth_mean, depth_std = compute_depth_statistics(
                    self.train_loader,
                    device=self.device,
                    max_samples=max_samples
                )
                config['depth_stats'] = {'mean': depth_mean, 'std': depth_std}
                print(f"Depth stats: mean={depth_mean:.4f}, std={depth_std:.4f}")
                
                # Update dataset configs to use these stats
                for dataset in [self.dataloaders['train_dataset'], 
                              self.dataloaders['val_dataset'],
                              self.dataloaders['test_dataset']]:
                    dataset.config['depth_stats'] = config['depth_stats']
            else:
                print("\nSkipping depth statistics computation (compute_depth_stats=False)")
                print("Will use per-image normalization as fallback")
                # Set flag to use per-image normalization
                config['depth_stats'] = None
        elif input_channels == 3:
            # RGB-only input: depth stats not applicable
            config['depth_stats'] = None

        # Get class weights
        self.class_weights = None
        if config['training'].get('use_class_weights', True):
            if torch.is_tensor(self.dataloaders.get('class_weights')):
                self.class_weights = self.dataloaders['class_weights'].to(self.device, dtype=torch.float32)
            else:
                train_dataset = self.dataloaders['train_dataset']
                dataset_weights = getattr(train_dataset, 'class_weights', None)
                if isinstance(dataset_weights, dict):
                    # Some splits may be missing a class; default missing weights to 0.0
                    weights = [dataset_weights.get(i, 0.0) for i in range(config['data']['num_classes'])]
                    self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                elif dataset_weights is not None:
                    self.class_weights = torch.as_tensor(dataset_weights, dtype=torch.float32, device=self.device)
            if self.class_weights is not None:
                print(f"\nUsing class weights: {self.class_weights.tolist()}")

        # Create model
        print("\nCreating model...")
        # Auto-detect photometry input dim from dataset
        train_ds = self.dataloaders['train_dataset']
        if getattr(train_ds, 'photometry_cols', None) and config['model'].get('photometry_input_dim', 0) == 0:
            config['model']['photometry_input_dim'] = len(train_ds.photometry_cols)
            print(f"Auto-detected photometry_input_dim = {len(train_ds.photometry_cols)}")
        if self.arch == 'photometry_mlp':
            from src.models.size_models import create_photometry_mlp_from_config
            self.model = create_photometry_mlp_from_config(config)
        elif self.arch == 'bsenet':
            self.model = create_bsenet_from_config(config)
        elif self.arch in {'rgb_resnet', 'resnet_rgb'}:
            self.model = create_resnet_rgb_from_config(config)
        else:
            self.model = create_vit_rgbd_from_config(config)
        self.model = self.model.to(self.device)
        if self.arch != 'photometry_mlp':
            ft_details = apply_finetune_config(self.model, config)
            print(f"Finetune config: {ft_details}")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / max(total_params, 1):.2%})")

        # Loss function
        loss_fn_factory = create_loss_function(config)
        self.criterion = loss_fn_factory(self.class_weights)
        loss_type = config['training'].get('loss_function', 'cross_entropy')
        print(f"Using loss function: {loss_type}")
        self.vrex_lambda = float(config['training'].get('vrex_lambda', 0.0))

        # Optimizer
        optimizer_name = config['training'].get('optimizer', 'adamw').lower()
        lr = config['training']['learning_rate']
        wd = config['training']['weight_decay']

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found; check finetune config.")

        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=wd)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(trainable, lr=lr, weight_decay=wd)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(trainable, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Learning rate scheduler
        scheduler_name = config['training'].get('scheduler', 'cosine').lower()
        max_epochs = config['training']['max_epochs']
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        min_lr = config['training'].get('min_lr', 1e-6)

        if scheduler_name == 'cosine':
            # Cosine annealing with warmup
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=min_lr
            )
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )
        else:
            self.scheduler = None
            self.warmup_scheduler = None

        # AMP scaler
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Optional margin-based pairwise ranking loss (ERk4)
        self.use_margin_ranking = bool(config['training'].get('use_margin_ranking', False))
        self.margin_ranking_weight = float(config['training'].get('margin_ranking_weight', 0.1))
        self.margin_ranking_margin = float(config['training'].get('margin_ranking_margin', 1.0))
        if self.use_margin_ranking:
            print(f"Using margin ranking loss: weight={self.margin_ranking_weight}, margin={self.margin_ranking_margin}mm")

        # Optional mask consistency loss (requires dataset to return image_unmasked)
        self.mask_consistency_weight = float(config['training'].get('mask_consistency_weight', 0.0))
        self.mask_consistency_mode = str(config['training'].get('mask_consistency_mode', 'kl')).lower()
        if self.mask_consistency_weight > 0:
            print(f"Using mask consistency loss: weight={self.mask_consistency_weight}, mode={self.mask_consistency_mode}")

        # Optional mask supervision loss (requires model mask head and dataset masks)
        self.mask_supervision_weight = float(config['training'].get('mask_supervision_weight', 0.0))
        self.mask_supervision_mode = str(config['training'].get('mask_supervision_mode', 'bce')).lower()
        if self.mask_supervision_weight > 0:
            print(f"Using mask supervision loss: weight={self.mask_supervision_weight}, mode={self.mask_supervision_mode}")

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.early_stop_patience = config['training'].get('early_stop_patience', 7)
        self.monitor_metric = config['training'].get('early_stop_metric', 'val/macro_f1')
        self.monitor_mode = config['training'].get('early_stop_mode', 'max')

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_macro_f1': [],
            'train_weighted_accuracy': [],
            'val_loss': [],
            'val_acc': [],
            'val_macro_f1': [],
            'learning_rate': []
        }
        self._pipeline_audit = None

    def _batch_to_images(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get model input tensor from either RGBD loader ('image') or depth-only loader ('depth')."""
        images = batch.get('image')
        if images is None:
            images = batch.get('depth')
        if images is None:
            raise KeyError("Batch is missing both 'image' and 'depth' tensors.")

        if images.dim() == 3:
            images = images.unsqueeze(1)

        expected_channels = int(self.config.get('model', {}).get('input_channels', images.shape[1]))
        if images.dim() == 4 and images.shape[1] != expected_channels:
            if images.shape[1] == 1 and expected_channels > 1:
                images = images.repeat(1, expected_channels, 1, 1)
            elif images.shape[1] > expected_channels:
                images = images[:, :expected_channels]
            else:
                pad = expected_channels - images.shape[1]
                images = torch.cat([images, images[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        return images.to(self.device)

    def audit_split(self, loader, split_name: str, group_by: Optional[str] = None) -> Dict[str, Any]:
        ds = getattr(loader, "dataset", None)
        if ds is None or not hasattr(ds, "samples"):
            print(f"[AUDIT] split={split_name}: dataset has no samples; skipping")
            return {}

        if group_by is None:
            temporal_cfg = self.config.get('data', {}).get('temporal', {}) or {}
            group_by = self.config.get('evaluation', {}).get('aggregate_by')
            if group_by is None and temporal_cfg.get('enabled', False):
                group_by = temporal_cfg.get('group_by', 'video_id')
        if group_by is None:
            group_by = "lesion_id" if any("lesion_id" in s for s in getattr(ds, "samples", [])) else "video_id"

        groups: Dict[str, Dict[str, Any]] = {}
        missing_size = 0
        for s in ds.samples:
            gid = s.get(group_by, None)
            if gid is None:
                gid = s.get("lesion_id", s.get("video_id", "unknown"))
            gid = str(gid)
            g = groups.setdefault(gid, {"frames": 0, "sizes": []})
            g["frames"] += 1
            if "size_mm" in s and s.get("size_mm") is not None:
                try:
                    g["sizes"].append(float(s.get("size_mm", 0.0)))
                except Exception:
                    missing_size += 1
            else:
                missing_size += 1

        lengths = [int(g["frames"]) for g in groups.values()]
        frames_total = int(np.sum(lengths)) if lengths else 0

        polyps_gt5 = 0
        lengths_gt5: List[int] = []
        for g in groups.values():
            if g["sizes"]:
                sz = float(np.median(np.array(g["sizes"], dtype=np.float32)))
                if sz > 5.0:
                    polyps_gt5 += 1
                    lengths_gt5.append(int(g["frames"]))

        def _stats(arr: List[int]) -> Optional[Dict[str, float]]:
            if not arr:
                return None
            a = np.array(arr, dtype=np.float32)
            p10, p90 = np.percentile(a, [10, 90]).tolist()
            return {
                "min": float(np.min(a)),
                "mean": float(np.mean(a)),
                "p10": float(p10),
                "median": float(np.median(a)),
                "p90": float(p90),
                "max": float(np.max(a)),
            }

        out = {
            "split": split_name,
            "group_by": group_by,
            "polyps_total": int(len(groups)),
            "polyps_gt5": int(polyps_gt5),
            "polyps_le5": int(len(groups) - polyps_gt5),
            "frames_total": int(frames_total),
            "frames_per_polyp": _stats(lengths),
            "frames_per_polyp_gt5": _stats(lengths_gt5),
            "missing_size_mm_samples": int(missing_size),
        }

        print(
            f"[AUDIT] split={split_name} group_by={group_by} "
            f"polyps_total={out['polyps_total']} polyps_gt5={out['polyps_gt5']} polyps_le5={out['polyps_le5']}"
        )
        print(
            f"[AUDIT] split={split_name} frames_total={out['frames_total']} "
            f"frames_per_polyp={out['frames_per_polyp']} frames_per_polyp_gt5={out['frames_per_polyp_gt5']}"
        )
        if missing_size:
            print(f"[AUDIT] split={split_name} missing_size_mm_samples={missing_size}")
        return out

    def audit_pipeline(
        self,
        stage: str,
        *,
        polyps_entered: int,
        frames_entered: int,
        frames_kept: int,
        polyps_dropped_by_reason: Optional[Dict[str, int]] = None,
    ) -> None:
        if self._pipeline_audit is None:
            self._pipeline_audit = {}
        entry = self._pipeline_audit.setdefault(
            stage,
            {"polyps_entered": 0, "frames_entered": 0, "frames_kept": 0, "polyps_dropped": {}},
        )
        entry["polyps_entered"] += int(polyps_entered)
        entry["frames_entered"] += int(frames_entered)
        entry["frames_kept"] += int(frames_kept)
        if polyps_dropped_by_reason:
            for reason, count in polyps_dropped_by_reason.items():
                entry["polyps_dropped"][reason] = int(entry["polyps_dropped"].get(reason, 0) + int(count))

    def _get_class_names(self):
        class_names = self.config.get('evaluation', {}).get('class_names')
        if class_names:
            return class_names
        num_classes = self.config['data'].get('num_classes', 3)
        return ['le_5mm', 'gt_5mm'] if num_classes == 2 else [f'class_{i}' for i in range(num_classes)]

    def _format_per_class_metric(self, metrics: dict, class_names, suffix: str, pretty_suffix: str) -> str:
        pairs = [
            f"{name}_{pretty_suffix}: {metrics.get(f'{name}_{suffix}', 0.0):.4f}"
            for name in class_names
            if f"{name}_{suffix}" in metrics
        ]
        return ", ".join(pairs)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch in pbar:
            images = self._batch_to_images(batch)
            labels = batch['label'].to(self.device, dtype=torch.long)  # Ensure labels are long type
            features = batch.get('features')
            if features is not None:
                features = features.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast('cuda', enabled=self.use_amp):
                if self.arch == 'photometry_mlp':
                    logits = self.model(features)
                    mask_logits = None
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, (tuple, list)):
                        logits = outputs[0]
                        mask_logits = outputs[1] if len(outputs) > 1 else None
                    else:
                        logits = outputs
                        mask_logits = None

                loss = self.criterion(logits, labels)

                if self.vrex_lambda > 0 and 'dataset' in batch:
                    _dsets = batch['dataset']
                    _env_losses = []
                    for _env_vals in (('real_colon',), ('sun',)):
                        _mask = torch.tensor(
                            [d in _env_vals for d in _dsets], dtype=torch.bool, device=self.device
                        )
                        if _mask.sum() >= 1:
                            _env_losses.append(self.criterion(logits[_mask], labels[_mask]))
                    if len(_env_losses) >= 2:
                        _env_stack = torch.stack(_env_losses)
                        loss = _env_stack.mean() + self.vrex_lambda * _env_stack.var()

                if self.mask_supervision_weight > 0 and mask_logits is not None and 'mask' in batch:
                    mask_target = batch['mask'].to(self.device)
                    if mask_logits.dim() == 5 and mask_target.dim() == 4:
                        mask_target = mask_target.unsqueeze(2)
                    if mask_logits.shape != mask_target.shape:
                        mask_target = F.interpolate(
                            mask_target.float(),
                            size=mask_logits.shape[-2:],
                            mode='nearest',
                        )
                    if self.mask_supervision_mode == 'mse':
                        mask_loss = F.mse_loss(torch.sigmoid(mask_logits), mask_target.float())
                    else:
                        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target.float())
                    loss = loss + self.mask_supervision_weight * mask_loss

                if self.mask_consistency_weight > 0 and 'image_unmasked' in batch:
                    images_unmasked = batch['image_unmasked'].to(self.device)
                    outputs_unmasked = self.model(images_unmasked)
                    if isinstance(outputs_unmasked, (tuple, list)):
                        outputs_unmasked = outputs_unmasked[0]
                    if self.mask_consistency_mode == 'mse':
                        consistency = F.mse_loss(outputs_unmasked, logits.detach())
                    else:
                        target = torch.softmax(logits.detach(), dim=1)
                        logp = torch.log_softmax(outputs_unmasked, dim=1)
                        consistency = F.kl_div(logp, target, reduction='batchmean')
                    loss = loss + self.mask_consistency_weight * consistency

                # Margin-based pairwise ranking loss (ERk4)
                if self.use_margin_ranking and 'size_mm' in batch:
                    size_mm = batch['size_mm'].to(self.device, dtype=torch.float32)
                    # Use logit difference (>5 mm - <=5 mm) as the score
                    score = logits[:, 1] - logits[:, 0] if logits.size(1) >= 2 else logits.squeeze(1)
                    # Create random pairs within batch
                    idx = torch.randperm(score.size(0), device=score.device)
                    score_j = score[idx]
                    size_j = size_mm[idx]
                    # sign(y_i - y_j): +1 if i larger, -1 if j larger, 0 if equal
                    sign = torch.sign(size_mm - size_j)
                    # Only use pairs where sizes differ (skip ties)
                    valid_pairs = sign.abs() > 0
                    if valid_pairs.any():
                        margin = self.margin_ranking_margin
                        rank_loss = F.relu(-sign[valid_pairs] * (score[:score.size(0)][valid_pairs] - score_j[valid_pairs]) + margin)
                        loss = loss + self.margin_ranking_weight * rank_loss.mean()

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.config['training'].get('gradient_clip_val', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.config['training']['gradient_clip_val'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config['training'].get('gradient_clip_val', 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.config['training']['gradient_clip_val'])
                self.optimizer.step()

            # Statistics
            _, preds = torch.max(logits, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += images.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_corrects / total_samples
            })

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        class_names = self._get_class_names()
        train_metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            y_prob=None,
            class_names=class_names,
        )

        return epoch_loss, epoch_acc, train_metrics

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        val_loss, metrics = self.evaluate_split(self.val_loader, split_name="val", with_loss=True)
        return val_loss, metrics

    @torch.no_grad()
    def evaluate_split(self, loader, split_name: str, with_loss: bool = False):
        """Evaluate a dataloader and return metrics (and loss if requested)."""
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        # Optional aggregation for temporal windows (video/lesion-level metrics)
        temporal_cfg = self.config.get('data', {}).get('temporal', {}) or {}
        aggregate_by = self.config.get('evaluation', {}).get('aggregate_by')
        if aggregate_by is None and temporal_cfg.get('enabled', False):
            aggregate_by = temporal_cfg.get('group_by', 'video_id')
        aggregate_method = self.config.get('evaluation', {}).get('aggregate_method', 'mean_prob')
        use_aggregation = bool(aggregate_by)
        if use_aggregation:
            from collections import defaultdict
            group_probs = defaultdict(list)
            group_logits = defaultdict(list)
            group_probs_all = defaultdict(list)
            group_logits_all = defaultdict(list)
            group_labels = defaultdict(list)

        audit_split_enabled = bool(self.config.get("evaluation", {}).get("audit_split", False) or use_aggregation)
        audit_pipeline_enabled = bool(self.config.get("evaluation", {}).get("audit_pipeline", False) or use_aggregation)
        split_audit = None
        if audit_split_enabled:
            split_audit = self.audit_split(loader, split_name, group_by=aggregate_by)
        if audit_pipeline_enabled:
            self._pipeline_audit = {}

        for batch in tqdm(loader, desc=f"{split_name.capitalize()} Eval"):
            images = self._batch_to_images(batch)
            labels = batch['label'].to(self.device, dtype=torch.long)
            features = batch.get('features')
            if features is not None:
                features = features.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                if self.arch == 'photometry_mlp':
                    logits = self.model(features)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, (tuple, list)):
                        logits = outputs[0]
                    else:
                        logits = outputs
                if with_loss:
                    loss = self.criterion(logits, labels)
                    running_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)
            valid = torch.isfinite(logits).all(dim=1) & torch.isfinite(probs).all(dim=1)
            if not torch.isfinite(probs).all():
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                row_sums = probs.sum(dim=1, keepdim=True)
                uniform = torch.full_like(probs, 1.0 / max(probs.size(1), 1))
                probs = torch.where(row_sums > 0, probs / row_sums, uniform)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if use_aggregation:
                group_ids = batch.get(aggregate_by)
                if group_ids is None:
                    raise KeyError(f"aggregate_by={aggregate_by} not found in batch")
                if isinstance(group_ids, torch.Tensor):
                    group_ids = group_ids.cpu().numpy().tolist()
                probs_np = probs.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                if aggregate_method == 'mean_logit':
                    logits_np = logits.detach().cpu().numpy()
                valid_np = valid.detach().cpu().numpy().astype(bool)
                for i, gid in enumerate(group_ids):
                    gid = str(gid)
                    if aggregate_method == 'mean_logit':
                        group_logits_all[gid].append(logits_np[i])
                        if valid_np[i]:
                            group_logits[gid].append(logits_np[i])
                    else:
                        group_probs_all[gid].append(probs_np[i])
                        if valid_np[i]:
                            group_probs[gid].append(probs_np[i])
                    group_labels[gid].append(int(labels_np[i]))

                if audit_pipeline_enabled:
                    polyps_in_batch = len(set(group_ids))
                    self.audit_pipeline(
                        "input",
                        polyps_entered=polyps_in_batch,
                        frames_entered=len(group_ids),
                        frames_kept=len(group_ids),
                    )
                    self.audit_pipeline(
                        "nan_filter",
                        polyps_entered=polyps_in_batch,
                        frames_entered=len(group_ids),
                        frames_kept=int(valid.sum().item()),
                    )

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        class_names = self.config.get('evaluation', {}).get('class_names')
        if not class_names:
            num_classes = self.config['data'].get('num_classes', 3)
            class_names = ['le_5mm', 'gt_5mm'] if num_classes == 2 else [f'class_{i}' for i in range(num_classes)]
        num_classes = len(class_names)

        if use_aggregation:
            agg_probs = []
            agg_labels = []
            for gid, labels_list in group_labels.items():
                # Majority label for group
                vals, counts = np.unique(labels_list, return_counts=True)
                agg_labels.append(int(vals[counts.argmax()]))
                if aggregate_method == 'mean_logit':
                    logits_list = group_logits[gid] if group_logits[gid] else group_logits_all[gid]
                    if not logits_list:
                        logits_mean = np.zeros((num_classes,), dtype=np.float32)
                    else:
                        logits_mean = np.mean(logits_list, axis=0)
                    logits_mean = logits_mean - np.max(logits_mean)
                    exp = np.exp(logits_mean)
                    prob = exp / np.sum(exp)
                else:
                    probs_list = group_probs[gid] if group_probs[gid] else group_probs_all[gid]
                    if not probs_list:
                        prob = np.full((num_classes,), 1.0 / max(num_classes, 1), dtype=np.float32)
                    else:
                        prob = np.mean(probs_list, axis=0)
                agg_probs.append(prob)
            agg_probs = np.array(agg_probs)
            agg_labels = np.array(agg_labels)
            agg_preds = np.argmax(agg_probs, axis=1)
            metrics = compute_metrics(agg_labels, agg_preds, agg_probs, class_names=class_names)

            if audit_pipeline_enabled:
                if aggregate_method == 'mean_logit':
                    total_frames = sum(len(v) for v in group_logits_all.values())
                    kept_frames = sum(len(v) for v in group_logits.values())
                    polyps_empty = sum(1 for gid in group_labels if len(group_logits[gid]) == 0)
                else:
                    total_frames = sum(len(v) for v in group_probs_all.values())
                    kept_frames = sum(len(v) for v in group_probs.values())
                    polyps_empty = sum(1 for gid in group_labels if len(group_probs[gid]) == 0)
                self.audit_pipeline(
                    "aggregation",
                    polyps_entered=len(group_labels),
                    frames_entered=total_frames,
                    frames_kept=kept_frames,
                    polyps_dropped_by_reason={"empty_valid_fallback": polyps_empty} if polyps_empty else None,
                )
        else:
            metrics = compute_metrics(all_labels, all_preds, all_probs, class_names=class_names)

        if audit_split_enabled and split_audit is not None:
            metrics["audit_split"] = split_audit
        if audit_pipeline_enabled and self._pipeline_audit is not None:
            print(f"[AUDIT_PIPELINE] split={split_name} stages={list(self._pipeline_audit.keys())}")
            for stage in sorted(self._pipeline_audit.keys()):
                entry = self._pipeline_audit[stage]
                print(
                    f"[AUDIT_PIPELINE] split={split_name} stage={stage} "
                    f"polyps_entered={entry['polyps_entered']} "
                    f"polyps_dropped={entry['polyps_dropped']} "
                    f"frames_entered={entry['frames_entered']} frames_kept={entry['frames_kept']}"
                )
            metrics["audit_pipeline"] = self._pipeline_audit

        if with_loss:
            loss_val = running_loss / len(loader.dataset)
            return loss_val, metrics
        return metrics

    def load_checkpoint(self, ckpt_path: Path):
        """Load model weights from a checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {ckpt_path}")

    def generate_full_report(self, checkpoint_path: Path = None):
        """Generate a full metrics report for train/val/test splits."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best.pt'

        if checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: checkpoint not found at {checkpoint_path}, using current model weights.")

        report = {
            'train': self.evaluate_split(self.train_loader, split_name='train'),
            'val': self.evaluate_split(self.val_loader, split_name='val'),
            'test': self.evaluate_split(self.dataloaders['test'], split_name='test'),
        }

        report_path = self.output_dir / 'full_metrics_report.json'
        with open(report_path, 'w') as f:
            json.dump(_to_jsonable(report), f, indent=2)
        print(f"Saved full metrics report to {report_path}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['max_epochs']} epochs...")
        print(f"Probe: {self.config.get('scenario', {}).get('name', self.arch)}")

        for epoch in range(self.config['training']['max_epochs']):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_acc, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()
            val_acc = val_metrics['accuracy']
            val_macro_f1 = val_metrics['macro_f1']

            # Update learning rate
            if epoch < self.config['training'].get('warmup_epochs', 5):
                if self.warmup_scheduler:
                    self.warmup_scheduler.step()
            else:
                if self.scheduler:
                    self.scheduler.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_macro_f1'].append(train_metrics.get('macro_f1', 0.0))
            self.history['train_weighted_accuracy'].append(train_metrics.get('weighted_accuracy', 0.0))
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_macro_f1'].append(val_macro_f1)
            self.history['learning_rate'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(
                f"  Train Macro-F1: {train_metrics.get('macro_f1', 0.0):.4f}, "
                f"Train Weighted-Acc: {train_metrics.get('weighted_accuracy', 0.0):.4f}"
            )
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}")
            
            class_names = self._get_class_names()
            train_f1_str = self._format_per_class_metric(train_metrics, class_names, suffix='f1', pretty_suffix='F1')
            train_acc_str = self._format_per_class_metric(train_metrics, class_names, suffix='acc', pretty_suffix='Acc')
            val_f1_str = self._format_per_class_metric(val_metrics, class_names, suffix='f1', pretty_suffix='F1')
            val_acc_str = self._format_per_class_metric(val_metrics, class_names, suffix='acc', pretty_suffix='Acc')
            if train_f1_str:
                print(f"  Train Per-Class F1: {train_f1_str}")
            if train_acc_str:
                print(f"  Train Per-Class Acc: {train_acc_str}")
            if val_f1_str:
                print(f"  Val Per-Class F1: {val_f1_str}")
            if val_acc_str:
                print(f"  Val Per-Class Acc: {val_acc_str}")
            
            print(f"  Learning Rate: {current_lr:.6f}")

            # Check if best model
            current_metric = val_metrics.get(self.monitor_metric.replace('val/', ''), val_macro_f1)

            is_best = False
            if self.monitor_mode == 'max':
                if current_metric > self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                if current_metric < self.best_metric:
                    is_best = True
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best, val_metrics)

            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best {self.monitor_metric}: {self.best_metric:.4f}")
                break

        # Save training history
        self.save_history()

        print("\n✅ Training completed!")
        print(f"Best {self.monitor_metric}: {self.best_metric:.4f}")
        # Generate full report after training
        self.generate_full_report()

    def save_checkpoint(self, is_best, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'metrics': metrics
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  💾 Saved best checkpoint (Macro-F1: {metrics['macro_f1']:.4f})")

        # Save periodic checkpoints
        if (self.current_epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.current_epoch+1}.pt')

    def save_history(self):
        """Save training history"""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train polyp size classification model")

    parser.add_argument('--config', type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument('--report_only', action='store_true',
                        help="Skip training and only generate full metrics report")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Checkpoint path for report_only (defaults to best.pt)")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Print configuration
    print("="*80)
    print(f"Polyp Size Classification Training")
    print("="*80)
    print(f"Probe: {config.get('scenario', {}).get('name', config.get('model', {}).get('architecture', 'probe'))}")
    print(f"Config: {args.config}")
    print(f"Output dir: {config['logging']['output_dir']}/{config['logging']['experiment_name']}")
    print("="*80)

    # Create trainer
    trainer = Trainer(config, args)

    if args.report_only:
        ckpt = Path(args.checkpoint) if args.checkpoint else trainer.checkpoint_dir / 'best.pt'
        trainer.generate_full_report(checkpoint_path=ckpt)
        return

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
