#!/usr/bin/env python3
"""Prepare oracle/global scale-factor CSVs for the CNN3 depth probe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_PPS_INTRINSICS = {
    "fx": 227.60416,
    "fy": 237.5,
    "cx": 227.60416,
    "cy": 237.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Combined frame metadata CSV.")
    parser.add_argument("--depth-dir", type=Path, required=True, help="Root containing depth tensors.")
    parser.add_argument("--sun-path", type=Path, default=None, help="Optional SUN root for depth fallback.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/depth_scale_factors_oracle.csv"),
        help="Master oracle-scale CSV.",
    )
    parser.add_argument("--fold-split", type=Path, default=None, help="Optional fold split CSV.")
    parser.add_argument("--fold", type=int, default=0, help="Fold id used in generated scale-file names.")
    parser.add_argument(
        "--scale-output-dir",
        type=Path,
        default=Path("data/splits/scale_factors"),
        help="Output directory for fold-specific scale_factor CSVs when --fold-split is provided.",
    )
    parser.add_argument("--fx", type=float, default=DEFAULT_PPS_INTRINSICS["fx"])
    parser.add_argument("--fy", type=float, default=DEFAULT_PPS_INTRINSICS["fy"])
    parser.add_argument("--cx", type=float, default=DEFAULT_PPS_INTRINSICS["cx"])
    parser.add_argument("--cy", type=float, default=DEFAULT_PPS_INTRINSICS["cy"])
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap; 0 means all rows.")
    return parser.parse_args()


def resolve_depth_path(row: pd.Series, depth_dir: Path, sun_path: Optional[Path]) -> Optional[Path]:
    if "depth_path" in row and pd.notna(row["depth_path"]) and str(row["depth_path"]).strip():
        path = Path(str(row["depth_path"]))
        return path if path.exists() else None

    depth_idx = int(row["depth_idx"])
    dataset = str(row.get("dataset", "")).strip()
    candidates = []
    if dataset == "real_colon":
        candidates.append(depth_dir / "real_colon" / f"{depth_idx:06d}.pt")
    elif dataset == "sun":
        candidates.append(depth_dir / "sun" / f"{depth_idx:06d}.pt")
        if sun_path is not None:
            candidates.append(sun_path / "depth_maps_ppsnet" / f"{depth_idx:06d}.pt")
    candidates.append(depth_dir / f"{depth_idx:06d}.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_depth(path: Path) -> np.ndarray:
    try:
        depth = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        depth = torch.load(path, map_location="cpu")
    if torch.is_tensor(depth):
        depth = depth.cpu().numpy()
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[0]
    return depth.astype(np.float32)


def bbox_mask_for_pps_depth(
    depth_hw: Tuple[int, int],
    image_hw: Tuple[float, float],
    bbox_xyxy: Tuple[float, float, float, float],
) -> np.ndarray:
    dh, dw = depth_hw
    orig_w, orig_h = image_hw
    x1, y1, x2, y2 = bbox_xyxy

    resize_target = 518.0
    scale = resize_target / min(orig_h, orig_w)
    resized_w = orig_w * scale
    resized_h = orig_h * scale
    crop_x_offset = (resized_w - resize_target) / 2.0
    crop_y_offset = (resized_h - resize_target) / 2.0
    depth_scale = float(dw) / resize_target

    bx1 = (x1 * scale - crop_x_offset) * depth_scale
    by1 = (y1 * scale - crop_y_offset) * depth_scale
    bx2 = (x2 * scale - crop_x_offset) * depth_scale
    by2 = (y2 * scale - crop_y_offset) * depth_scale

    ix1 = max(0, int(np.floor(min(bx1, bx2))))
    iy1 = max(0, int(np.floor(min(by1, by2))))
    ix2 = min(dw, int(np.ceil(max(bx1, bx2))))
    iy2 = min(dh, int(np.ceil(max(by1, by2))))

    mask = np.zeros((dh, dw), dtype=np.uint8)
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = 255
    return mask


def calculate_diameter(
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_points: int = 512,
) -> Optional[float]:
    mask_bin = (mask > 127).astype(np.uint8)
    if depth.ndim != 2 or mask_bin.sum() < 10:
        return None

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None

    hull = cv2.convexHull(contour)
    pts = hull[:, 0, :].astype(np.float32)
    if pts.shape[0] < 2:
        return None
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts = pts[idx]

    diff = pts[:, None, :] - pts[None, :, :]
    i, j = np.unravel_index(np.argmax(np.sum(diff * diff, axis=2)), (len(pts), len(pts)))
    ax, ay = pts[i]
    bx, by = pts[j]

    h, w = depth.shape
    xa = int(np.clip(round(ax), 0, w - 1))
    ya = int(np.clip(round(ay), 0, h - 1))
    xb = int(np.clip(round(bx), 0, w - 1))
    yb = int(np.clip(round(by), 0, h - 1))

    da = float(depth[ya, xa])
    db = float(depth[yb, xb])
    if not np.isfinite(da) or not np.isfinite(db) or da <= 0 or db <= 0:
        return None

    xa_3d = (xa - cx) * da / fx
    ya_3d = (ya - cy) * da / fy
    xb_3d = (xb - cx) * db / fx
    yb_3d = (yb - cy) * db / fy
    diam = np.sqrt((xa_3d - xb_3d) ** 2 + (ya_3d - yb_3d) ** 2 + (da - db) ** 2)
    if not np.isfinite(diam) or diam <= 0:
        return None
    return float(diam)


def size_from_row(row: pd.Series) -> float:
    for col in ("size_mm", "gt", "gt_size_mm"):
        if col in row and pd.notna(row[col]):
            return float(row[col])
    raise ValueError("Metadata must contain size_mm, gt, or gt_size_mm for oracle scale factors.")


def fit_scale_ls(df: pd.DataFrame) -> float:
    pred = df["pred_at_1"].astype(float).values
    gt = df["gt_size_mm"].astype(float).values
    den = float(np.sum(pred ** 2))
    if den <= 1e-12:
        return 1.0
    return float(np.sum(pred * gt) / den)


def save_scale_csv(df: pd.DataFrame, path: Path, col: str) -> None:
    out = df[["dataset", "image_name", col, "split"]].rename(columns={col: "scale_factor"})
    out = out[np.isfinite(out["scale_factor"])]
    out.to_csv(path, index=False)
    print(f"{path}: {len(out)} rows")


def build_fold_scale_files(master: pd.DataFrame, fold_split: Path, fold: int, out_dir: Path) -> None:
    split_df = pd.read_csv(fold_split)
    merged = split_df.merge(master, on=["dataset", "image_name"], how="left", validate="one_to_one")
    train_rows = merged[(merged["split"] == "train") & np.isfinite(merged["pred_at_1"])].copy()
    if train_rows.empty:
        raise ValueError(f"No train rows with pred_at_1 after merging {fold_split}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_scale_csv(merged, out_dir / f"depth_scale_factors_oracle_frame_fold{fold}.csv", "oracle_scale_frame")
    save_scale_csv(merged, out_dir / f"depth_scale_factors_oracle_polyp_fold{fold}.csv", "oracle_scale_polyp")

    global_scale = fit_scale_ls(
        train_rows.groupby(["dataset", "video_id"], as_index=False).agg(
            pred_at_1=("pred_at_1", "mean"),
            gt_size_mm=("gt_size_mm", "mean"),
        )
    )
    merged["global_train_scale"] = global_scale
    save_scale_csv(merged, out_dir / f"depth_scale_factors_global_train_fold{fold}.csv", "global_train_scale")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata)
    required = {
        "dataset",
        "image_name",
        "video_id",
        "depth_idx",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "image_width",
        "image_height",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    rows = []
    missing_depth = 0
    invalid_geom = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="oracle scale"):
        depth_path = resolve_depth_path(row, args.depth_dir, args.sun_path)
        if depth_path is None:
            missing_depth += 1
            continue
        depth = load_depth(depth_path)
        if depth.ndim != 2:
            invalid_geom += 1
            continue

        mask = bbox_mask_for_pps_depth(
            (depth.shape[0], depth.shape[1]),
            (float(row["image_width"]), float(row["image_height"])),
            (
                float(row["bbox_xmin"]),
                float(row["bbox_ymin"]),
                float(row["bbox_xmax"]),
                float(row["bbox_ymax"]),
            ),
        )
        pred_at_1 = calculate_diameter(depth, mask, args.fx, args.fy, args.cx, args.cy)
        if pred_at_1 is None or pred_at_1 <= 1e-8:
            invalid_geom += 1
            continue

        gt = size_from_row(row)
        s_frame = gt / pred_at_1
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "image_name": str(row["image_name"]),
                "video_id": str(row["video_id"]),
                "label": int(row["label"]) if "label" in row and pd.notna(row["label"]) else int(gt > 5.0),
                "depth_idx": int(row["depth_idx"]),
                "gt_size_mm": gt,
                "pred_at_1": float(pred_at_1),
                "oracle_scale_frame": float(s_frame),
                "oracle_pred_mm_frame": float(s_frame * pred_at_1),
            }
        )

    master = pd.DataFrame(rows)
    if master.empty:
        raise RuntimeError("No valid oracle-scale rows generated.")

    master["polyp_key"] = master["dataset"].astype(str) + "::" + master["video_id"].astype(str)
    polyp_scales = (
        master.groupby("polyp_key")
        .apply(lambda g: fit_scale_ls(g))
        .to_dict()
    )
    master["oracle_scale_polyp"] = master["polyp_key"].map(polyp_scales).astype(float)
    master["oracle_pred_mm_polyp"] = master["oracle_scale_polyp"] * master["pred_at_1"]
    master["oracle_abs_err_frame"] = np.abs(master["oracle_pred_mm_frame"] - master["gt_size_mm"])
    master["oracle_abs_err_polyp"] = np.abs(master["oracle_pred_mm_polyp"] - master["gt_size_mm"])
    master["scale_factor"] = master["oracle_scale_frame"]
    master = master.drop(columns=["polyp_key"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    print(f"{args.output}: {len(master)} valid rows")
    print(f"missing_depth={missing_depth}, invalid_geometry={invalid_geom}")

    if args.fold_split is not None:
        build_fold_scale_files(master, args.fold_split, args.fold, args.scale_output_dir)


if __name__ == "__main__":
    main()
