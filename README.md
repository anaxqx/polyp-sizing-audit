# Monocular Polyp Sizing Audit

Public code for diagnostic probes that audit monocular polyp size classification behavior.

The task is binary classification:

- `0`: polyp size `<= 5 mm`
- `1`: polyp size `> 5 mm`

This repository is for reproducing a model-behavior audit, not for releasing a deployable clinical sizing model. It contains only the public code needed to train and evaluate the core probes:

- ResNet18 RGB probe
- ViT-B RGB probe
- CNN3 depth probe
- MLP on handcrafted features
- MLP no-geometry ablation

Raw videos/frames, generated depth maps, generated masks, checkpoints, and third-party model weights are not included.

## Quick Start

```bash
git clone https://github.com/anaxqx/polyp-sizing-audit.git
cd polyp-sizing-audit
scripts/create_conda_env.sh
conda activate polyp-sizing-audit
```

Then prepare local data following the CSV/path conventions below and run one of the training configs.

## Repository Layout

```text
configs/              Example configs for the five public probes
src/datasets/         Custom dataloaders for RGB/mask/depth/features
src/models/           ResNet, ViT, CNN3 depth, and MLP probes
src/utils/            Metrics, losses, and transforms
train.py              Training entry point
eval.py               Checkpoint evaluation entry point
environment.yml       Conda environment
scripts/              Environment helper and MLP feature preparation
```

## Environment

```bash
scripts/create_conda_env.sh
conda activate polyp-sizing-audit
```

## Data Format

Prepare your data locally. Paths in the CSVs can be absolute, or relative to the dataset roots in the config.

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

## Datasets and Third-party Model Weights

The code expects users to obtain datasets and third-party model assets from their original sources:

- Clinical polyp size dataset: [Real-Colon](https://plus.figshare.com/articles/media/REAL-Colon_dataset/22202866) and [SUN-SEG](https://github.com/GewelsJI/VPS) frames/videos
- Depth estimation model: [MetricCol](https://github.com/liuyq055/MetricCol/tree/main) and [PPSNet](https://ppsnet.github.io/)
- Polyp segmentation model: [PolypPVT](https://github.com/dengpingfan/polyp-pvt)

Users must obtain datasets and third-party weights from their original sources and place derived files at the paths configured locally.
