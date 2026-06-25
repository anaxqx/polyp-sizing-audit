# Monocular Polyp Sizing Audit

[![arXiv](https://img.shields.io/badge/arXiv-2605.20461-b31b1b.svg)](https://arxiv.org/abs/2605.20461)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://anaxqx.github.io/polyp-sizing-audit/)

> **Understanding Model Behavior in Monocular Polyp Sizing**
>
> Xinqi Xiong, Andrea Dunn Beltran, Junmyeong Choi, Sarah K. McGill, Marc Niethammer, Roni Sengupta
>
> MICCAI 2026

Code for diagnostic probes that audit monocular polyp size classification behavior.

The task is binary classification:

- `0`: polyp size `<= 5 mm`
- `1`: polyp size `> 5 mm`

This repository provides a model-behavior audit for monocular polyp sizing task. It contains the public code and lightweight reproducibility assets needed to train and evaluate the core probes:

- ResNet18 RGB probe
- ViT-B RGB probe
- CNN3 depth probe (BseNet)
- MLP on 51 handcrafted features
- MLP no-geometry ablation
- oracle/global scale-factor utilities
- fold-assignment and statistical-test utilities

Raw videos/frames, generated depth maps, generated masks, checkpoints, and third-party model weights are not included. Users should obtain public datasets and third-party assets from their original sources and generate derived artifacts locally.

## Quick Start

```bash
git clone https://github.com/anaxqx/polyp-sizing-audit.git
cd polyp-sizing-audit

conda env create -f environment.yml
conda activate polyp-sizing-audit

pip install -r requirements.txt
```

Then prepare local data following the CSV/path conventions below and run one of the training configs. The training commands require locally prepared public-dataset frames plus derived masks/depth/features.

## Repository Layout

```text
configs/              Example configs for the five public probes
docs/                 Minimal GitHub Pages project page
splits/               Public fold assignments used to regenerate local split CSVs
src/datasets/         Custom dataloaders for RGB/mask/depth/features
src/models/           ResNet, ViT, CNN3 depth (BseNet), and MLP probes
src/utils/            Metrics, losses, and transforms
scripts/              Preprocessing, split, scale-factor, and statistics utilities
train.py              Training entry point
eval.py               Checkpoint evaluation entry point
environment.yml       Conda environment
requirements.txt      pip requirements
LICENSE               MIT License
```

## Environment

```bash
conda env create -f environment.yml
conda activate polyp-sizing-audit
```

## Data Format

Prepare your data locally.

For paper-number reproduction, these metadata files should describe the Real-Colon and SUN-SEG frame set used by the audit: all 232 unique polyps in the two datasets, after dropping frames without an annotated polyp. The paper reports 147 polyps `<=5 mm` and 85 polyps `>5 mm`, evaluated with patient-stratified cross-validation.

### Metadata CSVs

For RGB probes and MLP probes, provide dataset label CSVs such as:

```text
data/metadata/real_colon_labels.csv
data/metadata/sun_labels.csv
```

Required columns:

```text
image_name,video_id,label,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax
```

Recommended columns:

```text
patient_id,polyp_id,frame_id,size_mm,mask_path,depth_idx,image_width,image_height
```

`label` must be binary: `0` for `<=5 mm`, `1` for `>5 mm`.

### Split CSV

Use one split file per fold:

```text
data/splits/fold0.csv
```

Required columns:

```text
dataset,image_name,split
```

`split` is one of `train`, `val`, or `test`. Keep all frames from the same patient/procedure/polyp in the same held-out fold.

The paper folds can be regenerated from the compact public assignment file:

```bash
python scripts/create_fold_splits.py \
  --metadata data/metadata/real_colon_labels.csv \
  --metadata data/metadata/sun_labels.csv \
  --fold-assignments splits/fold_assignments.csv \
  --output-dir data/splits
```

This writes `data/splits/fold0.csv` through `data/splits/fold4.csv`. For fold `k`, the validation fold is `(k + 1) mod 5`, matching the retained training runs.

### Images And Masks

RGB configs expect:

```text
data/raw/real-colon/...
data/raw/sun-seg/...
```

The dataloader can use bounding boxes and/or masks depending on `input_composition`. For masked RGB, either provide `mask_path` in the metadata or organize masks under the dataset-specific segmentation result directory expected by the dataloader.

### Depth Maps

The CNN3 depth probe expects one tensor per frame:

```text
data/depth/real_colon/{depth_idx:06d}.pt
data/depth/sun/{depth_idx:06d}.pt
```

Each `.pt` file should load to a single-frame depth array/tensor. PPSNet relative depth and MetricCol metric depth can both be used as long as the config points to the correct `depth_dir` and `depth_source`.

### Oracle Scale Factors

The paper's oracle factors are diagnostic variables, not deployable estimators. They back-calculate the missing scale from the dataset size label and the raw PPSNet depth-based size proxy:

```text
scale_frame = size_mm / pred_at_1
```

Per-polyp and global factors are fitted as no-intercept least-squares scales. Generate a master oracle CSV and fold-specific `scale_factor` files with:

```bash
python scripts/prepare_oracle_scale_factors.py \
  --metadata data/metadata/combined_labels.csv \
  --depth-dir data/depth \
  --sun-path data/raw/sun-seg \
  --output data/metadata/depth_scale_factors_oracle.csv \
  --fold-split data/splits/fold0.csv \
  --fold 0 \
  --scale-output-dir data/splits/scale_factors
```

Use a generated scale CSV by adding these config fields to `configs/cnn3_depth.yaml`:

```yaml
augmentation:
  depth_norm_method: none
data:
  scale_factors_file: data/splits/scale_factors/depth_scale_factors_oracle_frame_fold0.csv
  scale_factor_col: scale_factor
```

Use `depth_source: metriccol` to evaluate MetricCol depth maps. Use `mask_bbox_scale_factor: 0.8` or `1.2` for the 20% bounding-box perturbation controls. Use `prefer_predicted_mask_bbox: true` plus `real_colon_segmentation_dir` and `sun_segmentation_dir` to substitute PolypPVT predicted masks.

### Feature CSVs

The MLP probes expect precomputed numeric features:

```text
data/features/features_51.csv
data/features/features_no_geometry.csv
```

Required identifier columns:

```text
dataset,video_id,image_path,label
```

All remaining numeric columns are used as features. For the no-geometry ablation, remove:

```text
bbox_w,bbox_h,aspect_ratio,compactness,apparent_area_px,apparent_area_frac
```

Create these CSVs from metadata, images, and masks/bboxes:

```bash
python scripts/prepare_mlp_features.py \
  --metadata data/metadata/real_colon_labels.csv \
  --metadata data/metadata/sun_labels.csv \
  --image-root data/raw \
  --mask-root data/masks \
  --output data/features/features_51.csv \
  --no-geometry-output data/features/features_no_geometry.csv
```

The script uses the 51 photometry/geometry/radial-profile feature schema used by the MLP probe. If `size_mm` is present, labels are derived as `int(size_mm > 5.0)` so the threshold is always 5 mm.

The full feature CSV has this metadata prefix:

```text
dataset,video_id,lesion_id,frame_idx,image_path,size_mm,label
```

followed by 51 numeric feature columns. The no-geometry CSV removes only the six apparent geometry columns listed above.

## Training

```bash
python train.py --config configs/resnet18_rgb.yaml
python train.py --config configs/vit_b_rgb.yaml
python train.py --config configs/cnn3_depth.yaml
python train.py --config configs/mlp_features.yaml
python train.py --config configs/mlp_no_geometry.yaml
```

The configs use AdamW, learning rate `1e-4`, weight decay `1e-4`, 30 epochs, cosine annealing, and class-weighted cross entropy.

## Evaluation

Training writes checkpoints and metrics under `outputs/`.

```bash
python eval.py \
  --config configs/resnet18_rgb.yaml \
  --checkpoint outputs/resnet18_rgb_fold0/checkpoints/best.pt \
  --split test
```

Primary metrics are Macro-F1 and recall for the `>5 mm` class. Report mean and standard deviation over five patient-stratified folds.

For the shortcut-consistent versus shortcut-inconsistent oracle-gain test, prepare a fold-level CSV with columns `fold,group,baseline_macro_f1,oracle_macro_f1` and run:

```bash
python scripts/shortcut_gain_ttest.py --csv data/analysis/shortcut_oracle_gains.csv
```

## Datasets and Third-party Model Weights

The code expects users to obtain datasets and third-party model assets from their original sources:

- Clinical polyp size dataset: [Real-Colon](https://plus.figshare.com/articles/media/REAL-Colon_dataset/22202866) and [SUN-SEG](https://github.com/GewelsJI/VPS) frames/videos
- Depth estimation model: [MetricCol](https://github.com/liuyq055/MetricCol/tree/main) and [PPSNet](https://ppsnet.github.io/)
- Polyp segmentation model: [PolypPVT](https://github.com/dengpingfan/polyp-pvt)

Users need to obtain datasets and third-party weights from their original sources and place derived files at the paths configured locally. This repository does not include trained probe checkpoints.

## Citation

```bibtex
@article{xiong2026understanding,
  title={Understanding Model Behavior in Monocular Polyp Sizing},
  author={Xiong, Xinqi and Beltran, Andrea Dunn and Choi, Junmyeong and McGill, Sarah K and Niethammer, Marc and Sengupta, Roni},
  journal={arXiv preprint arXiv:2605.20461},
  year={2026}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
