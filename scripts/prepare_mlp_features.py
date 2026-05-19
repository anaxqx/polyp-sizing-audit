#!/usr/bin/env python3
"""Prepare the original 51 handcrafted MLP features for the 5 mm task."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm


RADIAL_BINS = 15
SPECULAR_V_THRESH = 0.95
SPECULAR_S_THRESH = 0.10
GEOMETRY_COLUMNS = [
    "bbox_w",
    "bbox_h",
    "aspect_ratio",
    "compactness",
    "apparent_area_px",
    "apparent_area_frac",
]
META_COLUMNS = ["dataset", "video_id", "lesion_id", "frame_idx", "image_path", "size_mm", "label"]
FEATURE_COLUMNS = [
    "specular_ratio",
    "specular_count",
    "bg_gray_median",
    "bg_L_median",
    "apparent_area_px",
    "apparent_area_frac",
    "bbox_w",
    "bbox_h",
    "aspect_ratio",
    "compactness",
    "polyp_bg_ratio",
    "bg_radial_slope",
    "bg_ring_0",
    "bg_ring_1",
    "bg_ring_2",
    "bg_ring_3",
    "bg_ring_4",
    "gray_radial_slope",
    "gray_ec_contrast",
    "gray_profile_0",
    "gray_profile_1",
    "gray_profile_2",
    "gray_profile_3",
    "gray_profile_4",
    "gray_profile_5",
    "gray_profile_6",
    "gray_profile_7",
    "gray_profile_8",
    "gray_profile_9",
    "gray_profile_10",
    "gray_profile_11",
    "gray_profile_12",
    "gray_profile_13",
    "gray_profile_14",
    "lab_radial_slope",
    "lab_ec_contrast",
    "lab_profile_0",
    "lab_profile_1",
    "lab_profile_2",
    "lab_profile_3",
    "lab_profile_4",
    "lab_profile_5",
    "lab_profile_6",
    "lab_profile_7",
    "lab_profile_8",
    "lab_profile_9",
    "lab_profile_10",
    "lab_profile_11",
    "lab_profile_12",
    "lab_profile_13",
    "lab_profile_14",
]
assert len(FEATURE_COLUMNS) == 51

FEATURE_DESCRIPTIONS = {
    "specular_ratio": "Fraction of polyp-mask pixels detected as specular highlights.",
    "specular_count": "Number of specular pixels inside the polyp mask.",
    "bg_gray_median": "Median grayscale intensity in the dilated ring around the polyp mask.",
    "bg_L_median": "Median CIE-LAB L-channel intensity in the dilated ring around the polyp mask.",
    "apparent_area_px": "Polyp apparent area in pixels from the mask or bbox fallback.",
    "apparent_area_frac": "Polyp apparent area divided by full image area.",
    "bbox_w": "Width in pixels of the tight mask bounding box.",
    "bbox_h": "Height in pixels of the tight mask bounding box.",
    "aspect_ratio": "Tight mask bounding-box width divided by height.",
    "compactness": "Mask area divided by the area of a circle with diameter equal to the larger bbox side.",
    "polyp_bg_ratio": "Mean grayscale polyp intensity after normalization by the surrounding ring median.",
    "bg_radial_slope": "Log-intensity slope across five concentric background rings outside the polyp.",
    "bg_ring_0": "Ring-normalized grayscale mean in the first background ring outside the polyp.",
    "bg_ring_1": "Ring-normalized grayscale mean in the second background ring outside the polyp.",
    "bg_ring_2": "Ring-normalized grayscale mean in the third background ring outside the polyp.",
    "bg_ring_3": "Ring-normalized grayscale mean in the fourth background ring outside the polyp.",
    "bg_ring_4": "Ring-normalized grayscale mean in the fifth background ring outside the polyp.",
    "gray_radial_slope": "Log-intensity slope of the grayscale radial profile inside the polyp.",
    "gray_ec_contrast": "Inner-minus-outer grayscale contrast inside the polyp.",
    "lab_radial_slope": "Log-intensity slope of the LAB L-channel radial profile inside the polyp.",
    "lab_ec_contrast": "Inner-minus-outer LAB L-channel contrast inside the polyp.",
}
for idx in range(RADIAL_BINS):
    FEATURE_DESCRIPTIONS[f"gray_profile_{idx}"] = (
        f"Ring-normalized grayscale radial-profile bin {idx} inside the polyp."
    )
    FEATURE_DESCRIPTIONS[f"lab_profile_{idx}"] = (
        f"Ring-normalized LAB L-channel radial-profile bin {idx} inside the polyp."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", action="append", required=True, help="Metadata CSV. Can be repeated.")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Root for relative image paths.")
    parser.add_argument("--mask-root", type=Path, default=None, help="Root for relative mask paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with all 51 feature columns.")
    parser.add_argument("--no-geometry-output", type=Path, default=None, help="Optional no-geometry CSV output.")
    parser.add_argument(
        "--dictionary-output",
        type=Path,
        default=None,
        help="Optional feature dictionary JSON. Defaults to feature_dictionary.json beside --output.",
    )
    return parser.parse_args()


def read_metadata(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = Path(path).stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def resolve_path(value: object, root: Path | None) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    path = Path(str(value))
    if path.is_absolute() or root is None:
        return path
    return root / path


def image_path_from_row(row: pd.Series, image_root: Path) -> Path:
    value = row.get("image_path", row.get("image_name", None))
    path = resolve_path(value, image_root)
    if path is None:
        raise ValueError("metadata row is missing image_path or image_name")
    return path


def bbox_from_row(row: pd.Series, width: int, height: int) -> tuple[int, int, int, int] | None:
    if {"bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"}.issubset(row.index):
        vals = [row["bbox_xmin"], row["bbox_ymin"], row["bbox_xmax"], row["bbox_ymax"]]
    elif "bbox" in row.index and pd.notna(row["bbox"]):
        vals = ast.literal_eval(str(row["bbox"]))
    else:
        return None
    if any(pd.isna(v) for v in vals):
        return None
    x1, y1, x2, y2 = map(float, vals)
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
    return x1, y1, x2, y2


def mask_from_bbox(bbox: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> np.ndarray | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def load_mask(row: pd.Series, image_shape: tuple[int, int], bbox: tuple[int, int, int, int] | None, mask_root: Path | None) -> np.ndarray | None:
    mask_path = resolve_path(row.get("mask_path", None), mask_root)
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != image_shape:
                mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            return mask
    return mask_from_bbox(bbox, image_shape)


def dilated_ring(mask: np.ndarray) -> np.ndarray:
    size = max(3, int(round(min(mask.shape[:2]) * 0.04)))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return cv2.subtract(dilated, mask)


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def radial_features(vals: np.ndarray, norm_dists: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, float, float]:
    profile = np.full(RADIAL_BINS, np.nan)
    for idx in range(RADIAL_BINS):
        in_bin = (norm_dists >= bin_edges[idx]) & (norm_dists < bin_edges[idx + 1])
        if in_bin.sum() > 0:
            profile[idx] = vals[in_bin].mean()

    slope = np.nan
    valid = ~np.isnan(profile)
    if valid.sum() >= 3:
        x_v = np.arange(RADIAL_BINS)[valid]
        y_v = profile[valid]
        slope, _, _, _, _ = scipy_stats.linregress(x_v, np.log(np.maximum(y_v, 1e-8)))
        slope = float(slope)

    ec_contrast = np.nan
    inner = norm_dists < 0.3
    outer = norm_dists >= 0.7
    if inner.sum() > 0 and outer.sum() > 0:
        ec_contrast = float(vals[inner].mean() - vals[outer].mean())
    return profile, slope, ec_contrast


def detect_specular(hsv_float: np.ndarray, mask_px: np.ndarray) -> np.ndarray:
    v_ch = hsv_float[:, :, 2] / 255.0
    s_ch = hsv_float[:, :, 1] / 255.0
    v_max = v_ch[mask_px].max() if mask_px.any() else 1.0
    return mask_px & (v_ch > SPECULAR_V_THRESH * v_max) & (s_ch < SPECULAR_S_THRESH)


def background_radial_features(gray_norm: np.ndarray, mask: np.ndarray, centroid: tuple[int, int] | None) -> dict:
    result = {}
    if centroid is None:
        for idx in range(5):
            result[f"bg_ring_{idx}"] = np.nan
        result["bg_radial_slope"] = np.nan
        return result

    cx, cy = centroid
    h, w = mask.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    dist_map = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    mask_dists = dist_map[mask > 0]
    if len(mask_dists) == 0:
        return background_radial_features(gray_norm, mask, None)

    mask_max_dist = mask_dists.max()
    ring_means = []
    ring_centers = []
    for idx in range(5):
        inner_r = mask_max_dist + idx * 30
        outer_r = mask_max_dist + (idx + 1) * 30
        region = (dist_map >= inner_r) & (dist_map < outer_r) & (mask == 0)
        vals = gray_norm[region]
        mean_val = float(np.mean(vals)) if len(vals) > 10 else np.nan
        result[f"bg_ring_{idx}"] = mean_val
        ring_means.append(mean_val)
        ring_centers.append((inner_r + outer_r) / 2.0)

    valid = [(center, mean) for center, mean in zip(ring_centers, ring_means) if not np.isnan(mean)]
    if len(valid) >= 3:
        xs = np.array([item[0] for item in valid])
        ys = np.log(np.maximum(np.array([item[1] for item in valid]), 1e-8))
        slope, _, _, _, _ = scipy_stats.linregress(xs, ys)
        result["bg_radial_slope"] = float(slope)
    else:
        result["bg_radial_slope"] = np.nan
    return result


def compute_photometry_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    lab_l = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)[:, :, 0]

    mask_px = mask > 0
    ring = dilated_ring(mask)
    ring_px = ring > 0
    mask_area = int(mask_px.sum())
    image_area = h * w

    bg_gray_med = float(np.median(gray[ring_px])) if ring_px.any() else 1.0
    bg_gray_med = max(bg_gray_med, 1.0)
    gray_norm = gray / bg_gray_med

    bg_l_med = float(np.median(lab_l[ring_px])) if ring_px.any() else 1.0
    bg_l_med = max(bg_l_med, 1.0)
    lab_l_norm = lab_l / bg_l_med

    specular_px = detect_specular(hsv, mask_px)
    specular_count = int(specular_px.sum())
    specular_ratio = specular_count / mask_area if mask_area > 0 else 0.0
    valid_mask_px = mask_px & ~specular_px

    ys_all, xs_all = np.where(mask_px)
    bbox_w = int(xs_all.max() - xs_all.min() + 1) if len(xs_all) > 0 else 0
    bbox_h = int(ys_all.max() - ys_all.min() + 1) if len(ys_all) > 0 else 0
    aspect_ratio = bbox_w / max(bbox_h, 1)
    max_dim = max(bbox_w, bbox_h)
    compactness = mask_area / (np.pi * (max_dim / 2) ** 2 + 1e-8)

    result = {
        "specular_ratio": specular_ratio,
        "specular_count": specular_count,
        "bg_gray_median": bg_gray_med,
        "bg_L_median": bg_l_med,
        "apparent_area_px": mask_area,
        "apparent_area_frac": mask_area / image_area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "aspect_ratio": aspect_ratio,
        "compactness": compactness,
        "gray_radial_slope": np.nan,
        "gray_ec_contrast": np.nan,
        "lab_radial_slope": np.nan,
        "lab_ec_contrast": np.nan,
        "gray_radial_profile": [np.nan] * RADIAL_BINS,
        "lab_radial_profile": [np.nan] * RADIAL_BINS,
    }

    centroid = mask_centroid(mask)
    if centroid is not None:
        cx, cy = centroid
        ys, xs = np.where(valid_mask_px)
        if len(xs) > 10:
            dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            max_dist = dists.max()
            if max_dist > 0:
                norm_dists = dists / max_dist
                bin_edges = np.linspace(0, 1, RADIAL_BINS + 1)
                for prefix, arr in [("gray", gray_norm), ("lab", lab_l_norm)]:
                    profile, slope, ec_contrast = radial_features(arr[ys, xs], norm_dists, bin_edges)
                    result[f"{prefix}_radial_profile"] = profile.tolist()
                    result[f"{prefix}_radial_slope"] = slope
                    result[f"{prefix}_ec_contrast"] = ec_contrast

        result.update(background_radial_features(gray_norm, mask, centroid))
        result["polyp_bg_ratio"] = float(np.mean(gray_norm[valid_mask_px])) if valid_mask_px.any() else np.nan
    else:
        result.update(background_radial_features(gray_norm, mask, None))
        result["polyp_bg_ratio"] = np.nan

    return result


def derive_label(row: pd.Series) -> int:
    if "size_mm" in row.index and pd.notna(row["size_mm"]):
        return int(float(row["size_mm"]) > 5.0)
    if "label" not in row.index or pd.isna(row["label"]):
        raise ValueError("metadata row needs size_mm or binary label")
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"label must be binary 0/1 for the 5 mm task, got {label}")
    return label


def extract_row(row: pd.Series, image_root: Path, mask_root: Path | None) -> dict | None:
    image_path = image_path_from_row(row, image_root)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {image_path}")
    bbox = bbox_from_row(row, img.shape[1], img.shape[0])
    mask = load_mask(row, img.shape[:2], bbox, mask_root)
    if mask is None or mask.sum() == 0:
        raise ValueError(f"could not load non-empty mask or bbox for image: {image_path}")

    feats = compute_photometry_features(img, mask)
    out = {
        "dataset": row.get("dataset", "dataset"),
        "video_id": row.get("video_id", ""),
        "lesion_id": row.get("lesion_id", row.get("polyp_id", "")),
        "frame_idx": row.get("frame_idx", row.get("frame_id", "")),
        "image_path": row.get("image_path", row.get("image_name", str(image_path))),
        "size_mm": row.get("size_mm", np.nan),
        "label": derive_label(row),
        "specular_ratio": feats["specular_ratio"],
        "specular_count": feats["specular_count"],
        "bg_gray_median": feats["bg_gray_median"],
        "bg_L_median": feats["bg_L_median"],
        "apparent_area_px": feats["apparent_area_px"],
        "apparent_area_frac": feats["apparent_area_frac"],
        "bbox_w": feats["bbox_w"],
        "bbox_h": feats["bbox_h"],
        "aspect_ratio": feats["aspect_ratio"],
        "compactness": feats["compactness"],
        "polyp_bg_ratio": feats["polyp_bg_ratio"],
        "bg_radial_slope": feats["bg_radial_slope"],
    }
    for idx in range(5):
        out[f"bg_ring_{idx}"] = feats[f"bg_ring_{idx}"]
    for prefix in ["gray", "lab"]:
        out[f"{prefix}_radial_slope"] = feats[f"{prefix}_radial_slope"]
        out[f"{prefix}_ec_contrast"] = feats[f"{prefix}_ec_contrast"]
        profile = feats[f"{prefix}_radial_profile"]
        for idx in range(RADIAL_BINS):
            out[f"{prefix}_profile_{idx}"] = profile[idx] if idx < len(profile) else np.nan
    return out


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.metadata)
    rows = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="features"):
        rows.append(extract_row(row, args.image_root, args.mask_root))

    df = pd.DataFrame(rows)
    df = df[META_COLUMNS + FEATURE_COLUMNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    if args.no_geometry_output is not None:
        args.no_geometry_output.parent.mkdir(parents=True, exist_ok=True)
        keep = [col for col in df.columns if col not in GEOMETRY_COLUMNS]
        df[keep].to_csv(args.no_geometry_output, index=False)

    dictionary_output = args.dictionary_output or args.output.with_name("feature_dictionary.json")
    dictionary_output.parent.mkdir(parents=True, exist_ok=True)
    dictionary_output.write_text(json.dumps(FEATURE_DESCRIPTIONS, indent=2) + "\n")

    print(
        json.dumps(
            {
                "rows": len(df),
                "feature_columns": len(FEATURE_COLUMNS),
                "output": str(args.output),
                "no_geometry_output": str(args.no_geometry_output) if args.no_geometry_output else None,
                "dictionary_output": str(dictionary_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
