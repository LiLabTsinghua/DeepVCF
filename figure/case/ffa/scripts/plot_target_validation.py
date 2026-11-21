#!/usr/bin/env python
"""Plot FFA titer and OD-normalized titer for two validation experiments.

Run from the repository root in deepvcf:
    python figure/case/ffa/scripts/plot_target_validation.py

Inputs: data/processed/cf/case/ffa/experiments/{initial_gene_target_validation,
high_producer_target_validation}.txt.
Outputs: figure/case/ffa/outputs/{initial_gene_target_validation,
high_producer_target_validation}/.
Initial validation compares nine targets with Control; high-producer validation
compares six targets with the pcnBi-acrDi-fadR+ parent. Each strain has three
matched replicates. FFA/OD600 is calculated per replicate before aggregation.
Bars show mean and sample SD with every replicate shown. Tests are unpaired,
equal-variance, one-sided Student t-tests (target greater than baseline),
without multiplicity correction. Stars mark p < 0.05/0.01/0.001.
Exports labeled/no-text SVG/600-dpi PNG and source CSVs.
--data-dir and --output-dir override the default locations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    EDGE_COLOR, save_panel, style_axis,
)

from figure.case.validation_style import (
    BAR_WIDTH, CONTROL_COLOR, POINT_COLOR,
    create_variable_wide_panel, significance_symbol,
)

DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "cf" / "case" / "ffa" / "experiments"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'
EXPERIMENTS = {
    "initial": ("initial_gene_target_validation.txt", 10, "Control"),
    "high_producer": ("high_producer_target_validation.txt", 7, "Parent"),
}
EXPERIMENT_DIRS = {
    "initial": "initial_gene_target_validation",
    "high_producer": "high_producer_target_validation",
}
METRICS = {
    "ffas": ("ffas_mg_per_l", "FFAs (mg L$^{-1}$)"),
    "ffas_per_od": ("ffas_per_od600", "FFAs / OD$_{600}$"),
}
TARGET_COLOR = "#E59E3F"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def load_experiment(path, expected_strains):
    data = pd.read_csv(path, sep="\t")
    required = [
        "strain", "is_baseline", "replicate", "od600",
        "ffas_mg_per_l", "ffas_per_od600",
    ]
    if data.columns.tolist() != required or data[required].isna().any().any():
        raise ValueError(f"Incomplete or unexpected experiment data in {path}")
    if data["strain"].nunique() != expected_strains:
        raise ValueError(f"Expected {expected_strains} strains in {path}")
    if not data["is_baseline"].isin((0, 1)).all() or data["is_baseline"].sum() != 3:
        raise ValueError(f"Expected exactly three baseline replicates in {path}")
    if any(
        sorted(rows["replicate"].tolist()) != [1, 2, 3]
        for _, rows in data.groupby("strain")
    ):
        raise ValueError(f"Each strain needs replicates 1, 2, and 3 in {path}")
    if (data[["od600", "ffas_mg_per_l", "ffas_per_od600"]] < 0).any().any():
        raise ValueError(f"Negative measurement in {path}")
    if data["od600"].le(0).any():
        raise ValueError(f"Nonpositive OD600 in {path}")
    ratio_error = (data["ffas_mg_per_l"] / data["od600"] - data["ffas_per_od600"]).abs()
    if ratio_error.max() > 1e-7:
        raise ValueError(f"FFAs/OD600 is inconsistent with matched replicates in {path}")
    return data


def tick_label(strain, experiment, baseline_label):
    if experiment == "initial":
        return baseline_label if strain == "Control" else strain
    parent = "pcnBi-acrDi-fadR+"
    if strain == parent:
        return baseline_label
    prefix = parent + "-"
    if not strain.startswith(prefix):
        raise ValueError(f"Unexpected high-producer strain name: {strain}")
    return strain[len(prefix):]


def make_source(data, metric, experiment, baseline_label):
    records = []
    for order, strain in enumerate(data["strain"].drop_duplicates(), start=1):
        rows = data.loc[data["strain"].eq(strain)].sort_values("replicate")
        values = rows[metric].to_numpy(dtype=float)
        records.append({
            "plot_order": order,
            "strain": strain,
            "tick_label": tick_label(strain, experiment, baseline_label),
            "is_baseline": int(rows["is_baseline"].iloc[0]),
            "n_replicates": len(values),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "rep1": values[0], "rep2": values[1], "rep3": values[2],
        })
    source = pd.DataFrame(records)
    baseline = data.loc[data["is_baseline"].eq(1), metric].to_numpy(dtype=float)
    baseline_mean = float(baseline.mean())
    source["baseline_mean"] = baseline_mean
    source["mean_difference_vs_baseline"] = source["mean"] - baseline_mean
    source["t_statistic"] = np.nan
    source["degrees_of_freedom"] = np.nan
    source["p_one_sided_greater"] = np.nan
    source["significant_one_sided_0_05"] = False
    target_indices = source.index[source["is_baseline"].eq(0)].tolist()
    for index in target_indices:
        row = source.loc[index]
        target = np.array([row["rep1"], row["rep2"], row["rep3"]], dtype=float)
        test = ttest_ind(target, baseline, equal_var=True, alternative="greater")
        if not np.isfinite(test.pvalue):
            raise ValueError(f"Invalid t-test result for {row['strain']} / {metric}")
        source.loc[index, "t_statistic"] = float(test.statistic)
        source.loc[index, "degrees_of_freedom"] = float(test.df)
        source.loc[index, "p_one_sided_greater"] = float(test.pvalue)
        source.loc[index, "significant_one_sided_0_05"] = test.pvalue < 0.05
    return source


def plot_panel(source, experiment, metric_key, output_dir, show_text=True):
    _, ylabel = METRICS[metric_key]
    fig, ax = create_variable_wide_panel(len(source))
    x = np.arange(len(source), dtype=float)
    means = source["mean"].to_numpy(dtype=float)
    errors = source["sample_std"].to_numpy(dtype=float)
    colors = [CONTROL_COLOR if flag else TARGET_COLOR for flag in source["is_baseline"]]
    ax.bar(
        x, means, yerr=errors, width=BAR_WIDTH, color=colors,
        edgecolor=EDGE_COLOR, linewidth=0.75, alpha=1.0,
        error_kw={"ecolor": POINT_COLOR, "elinewidth": 0.9, "capsize": 2.8},
        zorder=2,
    )
    offsets = np.array([-0.16, 0.0, 0.16])
    for index, row in source.iterrows():
        ax.scatter(
            x[index] + offsets, [row["rep1"], row["rep2"], row["rep3"]],
            s=12, color=POINT_COLOR, linewidths=0, alpha=1.0, zorder=4,
        )
    ax.set_xlim(-0.5, len(source) - 0.5)
    highest = max(
        np.max(means + errors),
        source[["rep1", "rep2", "rep3"]].to_numpy(dtype=float).max(),
    )
    ax.set_ylim(0, highest * 1.17)
    ax.set_xticks(x)
    if show_text:
        ax.set_xticklabels(
            source["tick_label"], rotation=68, ha="right",
            rotation_mode="anchor", fontsize=7.5,
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="y", labelsize=9)
        for index, row in source.iterrows():
            if bool(row["significant_one_sided_0_05"]):
                ax.text(
                    x[index], means[index] + errors[index] + highest * 0.035,
                    significance_symbol(float(row["p_one_sided_greater"])),
                    ha="center", va="bottom", fontsize=10, color=POINT_COLOR,
                )
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    ax.grid(False)
    stem = output_dir / f"{experiment}_{metric_key}"
    if not show_text:
        stem = output_dir / f"{experiment}_{metric_key}_no_text"
    return save_panel(fig, stem)


def main(argv=None):
    args = parse_args(argv)
    for experiment, (filename, n_strains, baseline_label) in EXPERIMENTS.items():
        output_dir = args.output_dir / EXPERIMENT_DIRS[experiment]
        output_dir.mkdir(parents=True, exist_ok=True)
        data = load_experiment(args.data_dir / filename, n_strains)
        for metric_key, (metric, _) in METRICS.items():
            source = make_source(data, metric, experiment, baseline_label)
            source_path = output_dir / f"{experiment}_{metric_key}_source_data.csv"
            source.to_csv(source_path, index=False)
            for show_text in (True, False):
                for output in plot_panel(source, experiment, metric_key, output_dir, show_text):
                    print(f"Saved {output}")
            print(f"Saved {source_path}")


if __name__ == "__main__":
    main()
