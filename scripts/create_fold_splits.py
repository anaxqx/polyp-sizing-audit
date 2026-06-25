#!/usr/bin/env python3
"""Expand public fold assignments into local train/val/test split CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Metadata CSV. Can be repeated for Real-Colon and SUN-SEG.",
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("splits/fold_assignments.csv"),
        help="CSV with dataset, video_id, label, fold columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for generated fold0.csv ... fold4.csv files.",
    )
    return parser.parse_args()


def normalize_dataset(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"sunseg", "sun_seg", "sun"}:
        return "sun"
    if text in {"realcolon", "real_colon", "real_colon_labels"}:
        return "real_colon"
    return text


def normalize_video_id(dataset: str, video_id: object, image_name: object) -> str:
    if pd.notna(video_id) and str(video_id).strip():
        text = str(video_id).strip()
    else:
        text = str(image_name)

    if dataset == "sun":
        match = re.search(r"case(\d+)", text, flags=re.IGNORECASE)
        if match:
            return str(int(match.group(1)))
        if re.fullmatch(r"\d+(\.0)?", text):
            return str(int(float(text)))
        return text

    if dataset == "real_colon":
        return Path(text).name.split("_")[0]

    return text


def read_metadata(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = Path(path).stem
        frames.append(df)
    if not frames:
        raise ValueError("No metadata CSVs were provided.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    meta = read_metadata(args.metadata)
    assignments = pd.read_csv(args.fold_assignments)

    required_meta = {"dataset", "image_name"}
    missing_meta = required_meta - set(meta.columns)
    if missing_meta:
        raise ValueError(f"Metadata missing required columns: {sorted(missing_meta)}")

    required_assign = {"dataset", "video_id", "fold"}
    missing_assign = required_assign - set(assignments.columns)
    if missing_assign:
        raise ValueError(f"Fold assignment CSV missing columns: {sorted(missing_assign)}")

    meta = meta.copy()
    assignments = assignments.copy()
    meta["dataset_norm"] = meta["dataset"].map(normalize_dataset)
    assignments["dataset_norm"] = assignments["dataset"].map(normalize_dataset)

    video_col = "video_id" if "video_id" in meta.columns else None
    meta["video_id_norm"] = [
        normalize_video_id(ds, row.get(video_col) if video_col else None, row["image_name"])
        for ds, (_, row) in zip(meta["dataset_norm"], meta.iterrows())
    ]
    assignments["video_id_norm"] = [
        normalize_video_id(ds, vid, vid)
        for ds, vid in zip(assignments["dataset_norm"], assignments["video_id"])
    ]

    merged = meta.merge(
        assignments[["dataset_norm", "video_id_norm", "fold"]],
        on=["dataset_norm", "video_id_norm"],
        how="left",
        validate="many_to_one",
    )

    missing = merged[merged["fold"].isna()]
    if not missing.empty:
        preview = missing[["dataset", "image_name"]].head(10).to_dict("records")
        raise ValueError(f"{len(missing)} metadata rows did not match fold assignments. Examples: {preview}")

    n_folds = int(assignments["fold"].max()) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        out = merged[["dataset_norm", "image_name", "fold"]].copy()
        out["split"] = "train"
        out.loc[out["fold"].astype(int) == val_fold, "split"] = "val"
        out.loc[out["fold"].astype(int) == test_fold, "split"] = "test"
        out = out.rename(columns={"dataset_norm": "dataset"})
        out = out[["dataset", "image_name", "split"]]
        out_path = args.output_dir / f"fold{test_fold}.csv"
        out.to_csv(out_path, index=False)
        counts = out["split"].value_counts().to_dict()
        print(f"{out_path}: {counts}")


if __name__ == "__main__":
    main()
