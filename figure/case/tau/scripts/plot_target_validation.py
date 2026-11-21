#!/usr/bin/env python
"""Plot tau concentration and OD-normalized concentration for target validation.

Run from the repository root in deepvcf:
    python figure/case/tau/scripts/plot_target_validation.py

Input: data/processed/cf/case/tau/experiments/initial_gene_target_validation.txt.
Output: figure/case/tau/outputs/initial_gene_target_validation/.
The data table supplies the 21 target strains, E7c baseline, plotting order,
protein/locus identifiers, and GEM coverage. Original source_group IDs and
missing values are retained. katE coverage follows the original iML1515 CAT
reaction (b1732 or b3942), whose Catalyzes edge is absent from the KG because
its metabolites were filtered as common metabolites.

Bars show mean and sample SD of matched measurements. Most strains have
three measurements; rpe and sspB have two available tau measurements. Tau/OD
is computed per replicate before independent rounding to three decimals.
E7c remains the provisional baseline in the source experiment; the original
OD wavelength is unspecified. Tests are unpaired, equal-variance, one-sided
Student t-tests (target greater than baseline), without multiplicity correction.
Stars mark p < 0.05/0.01/0.001. The dagger labels on ribE and oxyR report
p >= 0.05. Exports labeled/no-text SVG/600-dpi PNG and source tables.
"""

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
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from figure.case.validation_style import (  # noqa: E402
    BAR_WIDTH, CONTROL_COLOR, POINT_COLOR,
    create_variable_wide_panel, significance_symbol,
)
from figure.figure_style import EDGE_COLOR, save_panel, style_axis  # noqa: E402

DEFAULT_DATA = (
    PROJECT_ROOT / "data/processed/cf/case/tau/experiments"
    / "initial_gene_target_validation.txt"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / 'outputs/initial_gene_target_validation'
)
GEM_COLORS = {
    "Baseline": CONTROL_COLOR,
    "GEM-covered": "#D45B72",
    "GEM-uncovered": "#E59E3F",
}
METRICS = {
    "tau": ("tau_mg_per_l", "Tau (mg L$^{-1}$)"),
    "tau_per_od": ("tau_per_od600", "Tau / OD"),
}
P_VALUE_SYMBOLS = {"tau": {"ribE", "oxyR"}}


def load_experiment(path):
    data = pd.read_csv(path, sep="\t")
    required = [
        "strain", "is_baseline", "replicate", "od600",
        "tau_mg_per_l", "tau_per_od600", "source_group",
        "protein_node", "locus_tag", "gem_group",
    ]
    if data.columns.tolist() != required:
        raise ValueError("Unexpected tau experiment schema")
    if data[["strain", "is_baseline", "replicate", "od600", "source_group"]].isna().any().any():
        raise ValueError("Missing tau record identifiers or OD")
    if not data["is_baseline"].isin((0, 1)).all():
        raise ValueError("Invalid baseline flags")
    if data.duplicated(["source_group", "replicate"]).any():
        raise ValueError("Repeated source-group/replicate identifier")
    if not np.isfinite(data["od600"]).all() or data["od600"].le(0).any():
        raise ValueError("OD must be finite and positive")
    for metric, _ in METRICS.values():
        valid = data[metric].dropna()
        if not np.isfinite(valid).all() or valid.lt(0).any():
            raise ValueError(f"Invalid nonnegative measurement: {metric}")
    if not data["tau_mg_per_l"].isna().equals(data["tau_per_od600"].isna()):
        raise ValueError("Concentration and normalized concentration need matching missing values")
    error = (data["tau_mg_per_l"] / data["od600"] - data["tau_per_od600"]).abs()
    # Allow independent three-decimal rounding of concentration and ratio.
    ratio_tolerance = .0005 * (1 + 1 / data["od600"]) + 1e-7
    if error.gt(ratio_tolerance).any():
        raise ValueError("Tau/OD disagrees with matched replicate measurements")
    for _, rows in data.groupby("source_group", sort=False):
        if rows["strain"].nunique() != 1 or rows["is_baseline"].nunique() != 1:
            raise ValueError("Each source group must have one strain and baseline flag")
        if sorted(rows["replicate"]) != [1, 2, 3]:
            raise ValueError("Each source group must contain replicate slots 1, 2, 3")
        if any(rows[metric].count() < 2 for metric, _ in METRICS.values()):
            raise ValueError("Each metric needs at least two available measurements per group")
    baseline = data.loc[data["is_baseline"].eq(1)]
    if baseline["source_group"].nunique() != 1 or set(baseline["strain"]) != {"E7c"}:
        raise ValueError("Expected E7c as the single provisional baseline group")
    data[['protein_node', 'locus_tag']] = data[['protein_node', 'locus_tag']].fillna('')
    if not data['gem_group'].isin(GEM_COLORS).all():
        raise ValueError('Invalid GEM coverage annotation')
    if not data['is_baseline'].eq(1).equals(data['gem_group'].eq('Baseline')):
        raise ValueError('Baseline flags and coverage annotations disagree')
    if data.drop_duplicates('source_group')['strain'].duplicated().any():
        raise ValueError('Each target must have exactly one experimental group')
    for _, rows in data.groupby('source_group', sort=False):
        if any(rows[column].nunique() != 1 for column in ('protein_node', 'locus_tag', 'gem_group')):
            raise ValueError('Each experimental group must have consistent annotations')
    targets = data.loc[data['is_baseline'].eq(0)]
    if (targets['protein_node'].eq('').any() or targets['locus_tag'].eq('').any()):
        raise ValueError('Target protein and locus annotations must be present')
    return data


def make_source(data, metric):
    groups = data.groupby("source_group", sort=False)
    baseline = data.loc[data["is_baseline"].eq(1), metric].dropna().to_numpy()
    baseline_mean = float(baseline.mean())
    records = []
    for order, (group, rows) in enumerate(groups, 1):
        rows = rows.sort_values("replicate")
        strain = rows["strain"].iloc[0]
        values = rows[metric].dropna().to_numpy(dtype=float)
        record = {
            "plot_order": order, "source_group": group, "strain": strain,
            "tick_label": strain, "is_baseline": int(rows["is_baseline"].iloc[0]),
            "protein_node": rows["protein_node"].iloc[0],
            "locus_tag": rows["locus_tag"].iloc[0],
            "gem_group": rows["gem_group"].iloc[0],
            "n_replicates": len(values), "n_missing": int(rows[metric].isna().sum()),
            "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
            **{f"rep{int(rep)}": value for rep, value in zip(rows["replicate"], rows[metric])},
            "baseline_mean": baseline_mean,
            "mean_difference_vs_baseline": float(values.mean() - baseline_mean),
            "t_statistic": np.nan, "degrees_of_freedom": np.nan,
            "p_one_sided_greater": np.nan, "significant_one_sided_0_05": False,
        }
        if not record["is_baseline"]:
            test = ttest_ind(values, baseline, equal_var=True, alternative="greater")
            if not np.isfinite(test.pvalue):
                raise ValueError(f"Invalid t-test for source group {group}")
            record.update(
                t_statistic=float(test.statistic), degrees_of_freedom=float(test.df),
                p_one_sided_greater=float(test.pvalue),
                significant_one_sided_0_05=bool(test.pvalue < .05),
            )
        records.append(record)
    return pd.DataFrame(records)


def plot_panel(source, metric_key, output_dir, show_text=True):
    fig, ax = create_variable_wide_panel(len(source))
    x = np.arange(len(source), dtype=float)
    means = source["mean"].to_numpy(dtype=float)
    errors = source["sample_std"].to_numpy(dtype=float)
    ax.bar(
        x, means, yerr=errors, width=BAR_WIDTH,
        color=[GEM_COLORS[group] for group in source["gem_group"]],
        edgecolor=EDGE_COLOR, linewidth=.75, alpha=1., zorder=2,
        error_kw={"ecolor": POINT_COLOR, "elinewidth": .9, "capsize": 2.8},
    )
    offsets = np.array([-.16, 0., .16])
    for index, row in source.iterrows():
        values = row[["rep1", "rep2", "rep3"]].to_numpy(dtype=float)
        valid = np.isfinite(values)
        ax.scatter(x[index] + offsets[valid], values[valid], s=12,
                   color=POINT_COLOR, linewidths=0, alpha=1., zorder=4)
    highest = max(float(np.max(means + errors)),
                  float(np.nanmax(source[["rep1", "rep2", "rep3"]].to_numpy(dtype=float))))
    ax.set_xlim(-.5, len(source) - .5)
    ax.set_ylim(0, highest * 1.17)
    if metric_key == "tau_per_od":
        ax.set_yticks(np.arange(0, highest * 1.17 + 1e-9, 50))
    ax.set_xticks(x)
    ax.tick_params(axis="y", labelsize=9)
    if show_text:
        ax.set_xticklabels(source["tick_label"], rotation=68, ha="right",
                           rotation_mode="anchor", fontsize=7.5)
        ax.set_ylabel(METRICS[metric_key][1], fontsize=11)
        ax.legend(
            handles=[Patch(facecolor=color, edgecolor=EDGE_COLOR,
                           label=group if group == "Baseline" else f"{group} gene")
                     for group, color in GEM_COLORS.items()],
            loc="upper left", ncol=3, fontsize=8.5, frameon=False,
        )
        marked = source.loc[
            source["strain"].isin(P_VALUE_SYMBOLS.get(metric_key, set()))
            & ~source["significant_one_sided_0_05"]
        ]
        if not marked.empty:
            detail = "; ".join(
                f"{row.strain}: p={row.p_one_sided_greater:.4f}"
                for row in marked.itertuples()
            )
            ax.text(.01, .89, f"† {detail} (p ≥ 0.05)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=POINT_COLOR)
        for index, row in source.iterrows():
            if row["significant_one_sided_0_05"]:
                ax.text(x[index], means[index] + errors[index] + highest * .035,
                        significance_symbol(row["p_one_sided_greater"]),
                        ha="center", va="bottom", fontsize=10, color=POINT_COLOR)
            elif row["strain"] in P_VALUE_SYMBOLS.get(metric_key, set()):
                ax.text(x[index], means[index] + errors[index] + highest * .035,
                        "†",
                        ha="center", va="bottom", fontsize=10, color=POINT_COLOR)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    ax.grid(False)
    stem = output_dir / (f"initial_{metric_key}" + ("" if show_text else "_no_text"))
    return save_panel(fig, stem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    data = load_experiment(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = data.drop_duplicates("source_group")[[
        "source_group", "strain", "is_baseline", "protein_node", "locus_tag", "gem_group",
    ]]
    coverage.to_csv(args.output_dir / "gem_coverage_source_data.csv", index=False)
    for key, (metric, _) in METRICS.items():
        source = make_source(data, metric)
        source.to_csv(args.output_dir / f"initial_{key}_source_data.csv", index=False)
        for show in (True, False):
            for path in plot_panel(source, key, args.output_dir, show):
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
