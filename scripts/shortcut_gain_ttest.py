#!/usr/bin/env python3
"""Paired t-test for oracle gains in shortcut-consistent vs inconsistent frames."""

from __future__ import annotations

import argparse

import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "CSV with columns fold,group,baseline_macro_f1,oracle_macro_f1. "
            "Group values must include consistent and inconsistent."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    required = {"fold", "group", "baseline_macro_f1", "oracle_macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    work = df.copy()
    work["group"] = work["group"].astype(str).str.lower()
    work["gain"] = work["oracle_macro_f1"].astype(float) - work["baseline_macro_f1"].astype(float)
    pivot = work.pivot(index="fold", columns="group", values="gain")
    if "consistent" not in pivot or "inconsistent" not in pivot:
        raise ValueError("group column must include consistent and inconsistent rows for each fold")
    pivot = pivot[["consistent", "inconsistent"]].dropna()
    if len(pivot) < 2:
        raise ValueError("Need at least two paired folds")

    t_stat, p_value = stats.ttest_rel(pivot["inconsistent"], pivot["consistent"])
    for group in ("consistent", "inconsistent"):
        vals = pivot[group]
        print(f"{group}: gain={vals.mean() * 100:.1f} +/- {vals.std(ddof=1) * 100:.1f} pp")
    print(f"paired t-test: t={t_stat:.4f}, p={p_value:.6g}, n={len(pivot)} folds")


if __name__ == "__main__":
    main()
