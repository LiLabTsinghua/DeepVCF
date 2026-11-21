#!/usr/bin/env python
"""Embedding ablation.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_embedding_ablation.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/ablation/.
Random versus trained DRN embeddings for LR, RF, MLP, and DeepVCF conditions; five seeds, mean and sample SD.
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/ablation'
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_TEMPLATE = "wide_v1"
SEED_SCATTER_MAX_SIZE = 16

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (
    BAR_EDGE_WIDTH,
    BAR_WIDTH,
    CATEGORY_TICK_SIZE,
    EDGE_COLOR,
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    PANEL_TEMPLATES,
    create_panel,
    save_panel,
    scatter_size_for_bar,
    style_axis,
)


from figure.external_evaluation.scripts.evaluation_data import (
    METRIC_SPECS,
    ABLATION_ORDER as METHOD_ORDER,
    ABLATION_TICK_LABELS as METHOD_TICK_LABELS,
    ABLATION_COLORS,
    load_ablation_data as load_source_data,
    summarize_ablation as summarize,
)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=PROJECT_ROOT,
        help="Repository root (default: inferred from this file).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: outputs/ablation below this script).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Experiment seeds to include (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--template", choices=tuple(PANEL_TEMPLATES), default=DEFAULT_TEMPLATE,
        help="Standalone panel template (default: wide_v1).",
    )
    return parser.parse_args(argv)


def draw_panel(ax, data, metric, seeds, y_max, show_text=True):
    """Draw mean ± sample SD bars with all seed observations."""
    data = data.loc[data["metric"].eq(metric)]
    matrix = (
        data.pivot(index="method", columns="seed", values="value")
        .loc[list(METHOD_ORDER), list(seeds)]
    )
    x = np.arange(len(METHOD_ORDER), dtype=float)
    means = matrix.mean(axis=1).to_numpy(dtype=float)
    stds = matrix.std(axis=1, ddof=1).to_numpy(dtype=float)
    ax.bar(
        x,
        means,
        yerr=stds,
        width=BAR_WIDTH,
        color=[ABLATION_COLORS[method] for method in METHOD_ORDER],
        edgecolor=EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        error_kw={
            "ecolor": EDGE_COLOR,
            "elinewidth": ERRORBAR_LINEWIDTH,
            "capsize": ERRORBAR_CAPSIZE,
            "capthick": ERRORBAR_LINEWIDTH,
        },
        zorder=2,
    )
    scatter_size = scatter_size_for_bar(
        ax, BAR_WIDTH, max_size=SEED_SCATTER_MAX_SIZE,
    )
    offsets = np.linspace(-0.18, 0.18, len(seeds))
    for row, method in enumerate(METHOD_ORDER):
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
    ax.set_xticks(x)
    if show_text:
        ax.set_xticklabels([METHOD_TICK_LABELS[m] for m in METHOD_ORDER])
        ax.tick_params(axis="x", labelsize=CATEGORY_TICK_SIZE, labelrotation=55)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_ylabel(METRIC_SPECS[metric]["label"])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
    ax.set_ylim(0, y_max)
    tick_step = METRIC_SPECS[metric]["tick_step"]
    ax.set_yticks(np.arange(0, y_max + 1e-9, tick_step))
    ax.grid(False)


def plot_panel(
    data, metric, seeds, y_max, output_stem, template, show_text=True,
):
    """Render and save one square standalone panel."""
    fig, ax = create_panel(template)
    draw_panel(ax, data, metric, seeds, y_max, show_text=show_text)
    return save_panel(fig, output_stem)


def run(repo_root, output_dir, seeds, template):
    """Write source data and render both gene-set panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_source_data(repo_root, seeds)
    data.to_csv(output_dir / "external_embedding_ablation_source_data.csv", index=False)
    summarize(data).to_csv(
        output_dir / "external_embedding_ablation_summary.csv", index=False,
    )
    outputs = []
    panel_specs = (
        ("set1", "AUPR", 0.85),
        ("set1", "AUROC", 1.00),
        ("set1", "Hits@30", 0.50),
        ("set2", "AUPR", 0.70),
        ("set2", "AUROC", 1.00),
        ("set2", "Hits@30", 0.35),
    )
    for gene_set, metric, y_max in panel_specs:
        subset = data.loc[data["gene_set"].eq(gene_set)]
        metric_stem = METRIC_SPECS[metric]["stem"]
        stem = output_dir / f"external_embedding_ablation_{gene_set}_{metric_stem}"
        outputs.extend(plot_panel(
            subset, metric, seeds, y_max, stem, template, show_text=True,
        ))
        outputs.extend(plot_panel(
            subset,
            metric,
            seeds,
            y_max,
            Path(f"{stem}_no_text"),
            template,
            show_text=False,
        ))
    print(f"Source rows: {len(data)}")
    print(f"Template: {template}")
    for output in outputs:
        print(f"Saved {output}")


def main(argv=None):
    """Generate the six external embedding-ablation panels."""
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be non-empty and contain no duplicates.")
    run(args.repo_root, args.output_dir, seeds, args.template)


if __name__ == "__main__":
    main()
