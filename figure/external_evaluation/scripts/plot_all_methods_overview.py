#!/usr/bin/env python
"""All-method overview.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_all_methods_overview.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/mechanistic_comparison/all_methods/.
Supplementary comparison including mechanistic methods, ML baselines, and DeepVCF conditions; five seeds, mean and sample SD.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from figure.figure_style import (  # noqa: E402
    BAR_EDGE_WIDTH,
    BAR_WIDTH,
    CATEGORY_TICK_SIZE,
    EDGE_COLOR,
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    create_panel,
    save_panel,
    scatter_size_for_bar,
    style_axis,
)
from figure.external_evaluation.scripts.evaluation_data import (  # noqa: E402
    METHOD_SPECS as BENCHMARK_METHOD_SPECS,
    TEST_KEYS,
    load_source_data as load_benchmark_data,
    summarize,
    ABLATION_COLORS,
    ABLATION_ORDER as ABLATION_METHOD_ORDER,
    ABLATION_TICK_LABELS,
    load_ablation_data,
    EXTERNAL_METHOD_COLORS,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/mechanistic_comparison/all_methods'
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
TEMPLATES = {
    "GEM-covered": "wide_v1",
    "GEM-uncovered": "nine_method_wide_v1",
}
SEED_SCATTER_MAX_SIZE = 16
MECHANISTIC_METHODS = ("FSEOF", "FVSEOF", "iBridge")
METHOD_ORDER = (*MECHANISTIC_METHODS, *ABLATION_METHOD_ORDER)
METHOD_TICK_LABELS = {
    **{method: method for method in MECHANISTIC_METHODS},
    **ABLATION_TICK_LABELS,
}
METHOD_COLORS = {
    **{method: EXTERNAL_METHOD_COLORS[method] for method in MECHANISTIC_METHODS},
    **ABLATION_COLORS,
}
METRICS = {
    "AUPR": ("aupr", "Gene-corruption AUPR", 0.2),
    "AUROC": ("auroc", "Gene-corruption AUROC", 0.2),
    "Hits@30": ("hits30", "Filtered target-gene Hits@30", 0.1),
}
Y_LIMITS = {
    "GEM-covered": {"AUPR": 0.85, "AUROC": 1.0, "Hits@30": 0.55},
    "GEM-uncovered": {"AUPR": 0.70, "AUROC": 1.0, "Hits@30": 0.35},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
    )
    return parser.parse_args()


def validate_and_matrix(data, metric, seeds, method_order, y_max):
    """Check the [0, 1] domain and a complete method-by-seed grid."""
    matrix = (
        data.loc[data["metric"].eq(metric)]
        .pivot(index="method", columns="seed", values="value")
        .loc[list(method_order), list(seeds)]
    )
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"Invalid or missing {metric} seed values.")
    means = values.mean(axis=1)
    stds = values.std(axis=1, ddof=1)
    if np.any(means + stds > y_max):
        raise ValueError(f"{metric} upper mean + SD exceeds its plotted axis.")
    return matrix


def draw_panel(ax, matrix, metric, seeds, method_order, y_max, show_text):
    """Apply the ablation panel's bars, error bars, and seed-point styling."""
    positions = np.arange(len(method_order), dtype=float)
    values = matrix.to_numpy(dtype=float)
    means = values.mean(axis=1)
    stds = values.std(axis=1, ddof=1)
    # A metric cannot fall below zero; retain the full upper SD and bound the
    # lower whisker at the domain edge when mean - SD is negative.
    error_bars = np.vstack((np.minimum(stds, means), stds))
    ax.bar(
        positions, means, yerr=error_bars,
        width=BAR_WIDTH,
        color=[METHOD_COLORS[method] for method in method_order],
        edgecolor=EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        alpha=1.0,
        error_kw={
            "ecolor": EDGE_COLOR,
            "elinewidth": ERRORBAR_LINEWIDTH,
            "capsize": ERRORBAR_CAPSIZE,
            "capthick": ERRORBAR_LINEWIDTH,
        },
        zorder=2,
    )
    ax.set_xlim(-0.55, len(method_order) - 0.45)
    ax.set_ylim(0, y_max)
    scatter_size = scatter_size_for_bar(
        ax, BAR_WIDTH, max_size=SEED_SCATTER_MAX_SIZE,
    )
    offsets = np.linspace(-0.18, 0.18, len(seeds))
    for row, method in enumerate(method_order):
        ax.scatter(
            row + offsets,
            matrix.loc[method].to_numpy(dtype=float),
            s=scatter_size,
            color=EDGE_COLOR,
            linewidths=0,
            alpha=1.0,
            zorder=3,
        )

    style_axis(ax)
    ax.set_xticks(positions)
    if show_text:
        ax.set_xticklabels([METHOD_TICK_LABELS[m] for m in method_order])
        ax.tick_params(axis="x", labelsize=CATEGORY_TICK_SIZE, labelrotation=55)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_ylabel(METRICS[metric][1])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_yticks(np.arange(0, y_max + 1e-9, METRICS[metric][2]))
    if not show_text:
        ax.set_yticklabels([])
    ax.grid(False)


def save_metric_panel(
    output_dir, data, metric, seeds, method_order, y_max, template,
):
    """Export text and no-text versions using the group's panel template."""
    matrix = validate_and_matrix(data, metric, seeds, method_order, y_max)
    stem = output_dir / f"external_all_methods_{METRICS[metric][0]}"
    for show_text, suffix in ((True, ""), (False, "_no_text")):
        fig, ax = create_panel(template)
        draw_panel(ax, matrix, metric, seeds, method_order, y_max, show_text)
        save_panel(fig, Path(f"{stem}{suffix}"))


def run(repo_root, output_dir, seeds):
    """Render three standalone metrics for each external gene group."""
    mechanistic_specs = tuple(
        spec for spec in BENCHMARK_METHOD_SPECS
        if spec.name in MECHANISTIC_METHODS
    )
    mechanistic = load_benchmark_data(
        repo_root, mechanistic_specs, seeds, TEST_KEYS["GEM"],
    )
    mechanistic = mechanistic.loc[
        mechanistic["metric"].isin(METRICS)
    ]
    ablation = load_ablation_data(repo_root, seeds)
    datasets = {}
    for label, key, method_order in (
        ("GEM-covered", "set1", METHOD_ORDER),
        ("GEM-uncovered", "set2", ABLATION_METHOD_ORDER),
    ):
        group_data = ablation.loc[
            ablation["gene_set"].eq(key)
            & ablation["metric"].isin(METRICS),
            ["method", "family", "seed", "metric", "value"],
        ]
        if label == "GEM-covered":
            group_data = pd.concat([mechanistic, group_data], ignore_index=True)
        expected = len(method_order) * len(seeds) * len(METRICS)
        if len(group_data) != expected:
            raise ValueError(
                f"Expected {expected} {label} rows, found {len(group_data)}."
            )
        datasets[label] = (group_data, method_order)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [data.assign(gene_set=label) for label, (data, _) in datasets.items()],
        ignore_index=True,
    ).to_csv(output_dir / "all_methods_source_data.csv", index=False)
    summary = pd.concat(
        [
            summarize(data, order).assign(gene_set=label)
            for label, (data, order) in datasets.items()
        ],
        ignore_index=True,
    )
    summary["errorbar_lower"] = (summary["mean"] - summary["std"]).clip(lower=0)
    summary["errorbar_upper"] = summary["mean"] + summary["std"]
    summary["lower_sd_truncated_at_zero"] = (
        summary["mean"] - summary["std"] < 0
    )
    summary.to_csv(
        output_dir / "all_methods_summary.csv", index=False,
    )
    for label, (data, method_order) in datasets.items():
        group_dir = output_dir / label.lower().replace("-", "_")
        for metric in METRICS:
            save_metric_panel(
                group_dir, data, metric, seeds, method_order,
                Y_LIMITS[label][metric], TEMPLATES[label],
            )
    print(f"Wrote six panels and source tables to {output_dir}")


def main():
    args = parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain at least two distinct values.")
    run(args.repo_root, args.output_dir, seeds)


if __name__ == "__main__":
    main()
