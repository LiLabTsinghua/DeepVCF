#!/usr/bin/env python
"""Mechanistic comparison.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_integrated_external_benchmark.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/mechanistic_comparison/.
Six methods for GEM-covered genes and three DeepVCF conditions for GEM-uncovered genes; five seeds. Bars and curves show mean and sample SD.
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MODULE_DIR = PROJECT_ROOT / "figure" / "external_evaluation" / "scripts"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/mechanistic_comparison'
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_TEMPLATE = "square_v1"
DEFAULT_UNCOVERED_TEMPLATE = "narrow_v1"
SEED_SCATTER_MAX_SIZE = 16

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SOURCE_MODULE_DIR))
from figure.figure_style import (
    BAR_EDGE_WIDTH,
    BAR_WIDTH,
    CATEGORY_TICK_SIZE,
    DATA_LINEWIDTH,
    EDGE_COLOR,
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    METHOD_TICK_LABELS,
    PANEL_TEMPLATES,
    create_panel,
    save_panel,
    scatter_size_for_bar,
    style_axis,
)
from figure.external_evaluation.scripts.evaluation_data import (  # noqa: E402
    METHOD_SPECS,
    TEST_KEYS,
    load_source_data,
    summarize,
    EXTERNAL_METHOD_COLORS,
)


SELECTED_METHODS = (
    "FSEOF",
    "FVSEOF",
    "iBridge",
    "DeepVCF",
    "DeepVCF+BE",
    "DeepVCF+BE+MP",
)
NON_GEM_METHODS = (
    "DeepVCF",
    "DeepVCF+BE",
    "DeepVCF+BE+MP",
)
SELECTED_SPECS = tuple(
    spec for spec in METHOD_SPECS if spec.name in SELECTED_METHODS
)
NON_GEM_SPECS = tuple(
    spec for spec in SELECTED_SPECS if spec.name in NON_GEM_METHODS
)
NON_GEM_ORDER = NON_GEM_METHODS
METHOD_ORDER = SELECTED_METHODS
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=None,
        help="Repository root; defaults to the root inferred from this file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory; defaults to this response figure directory.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Experiment seeds to include (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--template",
        choices=tuple(PANEL_TEMPLATES),
        default=DEFAULT_TEMPLATE,
        help="GEM-covered panel template (default: square_v1).",
    )
    parser.add_argument(
        "--uncovered-template",
        choices=tuple(PANEL_TEMPLATES),
        default=DEFAULT_UNCOVERED_TEMPLATE,
        help="GEM-uncovered panel template (default: narrow_v1).",
    )
    return parser.parse_args(argv)


def load_benchmark_data(repo_root, seeds):
    """Load the GEM and non-GEM external benchmark tables."""
    gem_data = load_source_data(
        repo_root, SELECTED_SPECS, seeds, TEST_KEYS["GEM"],
    )
    non_gem_data = load_source_data(
        repo_root, NON_GEM_SPECS, seeds, TEST_KEYS["non-GEM"],
    )
    return gem_data, non_gem_data


def metric_matrix_with_missing(data, metric, method_order, seeds):
    """Return a complete method-by-seed matrix, retaining missing methods."""
    subset = data.loc[data["metric"].eq(metric)]
    matrix = subset.pivot(index="method", columns="seed", values="value")
    return matrix.reindex(index=method_order, columns=list(seeds))


def draw_binary_metric(
    ax, data, metric, method_order, seeds, y_max, bar_width,
    show_text=True,
):
    """Draw mean ± sample SD binary-metric bars with seed observations."""
    matrix = metric_matrix_with_missing(data, metric, method_order, seeds)
    x = np.arange(len(method_order), dtype=float)
    means = matrix.mean(axis=1).to_numpy(dtype=float)
    stds = matrix.std(axis=1, ddof=1).to_numpy(dtype=float)
    valid = np.isfinite(means)
    ax.bar(
        x[valid],
        means[valid],
        yerr=stds[valid],
        width=bar_width,
        color=[
            EXTERNAL_METHOD_COLORS[method]
            for method, is_valid in zip(method_order, valid)
            if is_valid
        ],
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
        ax, bar_width, max_size=SEED_SCATTER_MAX_SIZE,
    )
    for row, method in enumerate(method_order):
        values = matrix.loc[method].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        offsets = np.linspace(-0.18, 0.18, values.size)
        ax.scatter(
            row + offsets,
            values,
            s=scatter_size,
            color=EDGE_COLOR,
            linewidths=0,
            zorder=3,
        )

    style_axis(ax)
    ax.set_xticks(x)
    if show_text:
        ax.set_xticklabels(
            [METHOD_TICK_LABELS[method] for method in method_order],
        )
        ax.tick_params(
            axis="x", labelsize=CATEGORY_TICK_SIZE, labelrotation=55,
        )
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.55, len(method_order) - 0.45)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1e-9, 0.2))
    if show_text:
        ax.set_ylabel(f"Gene-corruption {metric}")
    else:
        ax.set_yticklabels([])
    ax.grid(False)


def draw_hits(ax, data, method_order, seeds, y_max, show_text=True):
    """Draw filtered gene Hits@K means ± sample SD."""
    ks = (10, 20, 30)
    for method in method_order:
        values = np.vstack([
            metric_matrix_with_missing(data, f"Hits@{k}", method_order, seeds)
            .loc[method]
            .to_numpy(dtype=float)
            for k in ks
        ])
        if not np.isfinite(values).any():
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete Hits@K values for {method}.")
        ax.errorbar(
            ks,
            values.mean(axis=1),
            yerr=values.std(axis=1, ddof=1),
            color=EXTERNAL_METHOD_COLORS[method],
            marker="o",
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE,
            label=method,
            zorder=3,
        )

    style_axis(ax)
    ax.set_xticks(ks)
    if show_text:
        ax.set_xlabel("K")
        ax.set_ylabel("Filtered target-gene Hits@K")
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1e-9, 0.1))
    if not show_text:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(False)


def summarize_integrated(gem_data, non_gem_data, method_orders):
    """Combine summary tables while retaining the gene-set identity."""
    frames = []
    for gene_set, data, method_order in (
        ("GEM-COVERED GENE SET", gem_data, method_orders[0]),
        ("GEM-UNCOVERED GENE SET", non_gem_data, method_orders[1]),
    ):
        summary = summarize(data, method_order)
        summary.insert(0, "gene_set", gene_set)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def write_source_data(output_dir, gem_data, non_gem_data):
    """Write combined seed-level and summary source tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.concat(
        [
            gem_data.assign(gene_set="GEM-COVERED GENE SET"),
            non_gem_data.assign(gene_set="GEM-UNCOVERED GENE SET"),
        ],
        ignore_index=True,
    )
    source.to_csv(
        output_dir / "external_benchmark_integrated_source_data.csv",
        index=False,
    )
    summarize_integrated(
        gem_data,
        non_gem_data,
        (METHOD_ORDER, NON_GEM_ORDER),
    ).to_csv(
        output_dir / "external_benchmark_integrated_summary.csv",
        index=False,
    )


def plot_panel(
    data,
    metric,
    seeds,
    y_max,
    output_stem,
    template,
    method_order,
    bar_width,
    show_text=True,
):
    """Create one standalone panel without a title or panel letter."""
    fig, ax = create_panel(template)
    if metric in {"AUPR", "AUROC"}:
        draw_binary_metric(
            ax, data, metric, method_order, seeds, y_max, bar_width,
            show_text=show_text,
        )
    elif metric == "hits":
        draw_hits(
            ax, data, method_order, seeds, y_max, show_text=show_text,
        )
    else:
        raise ValueError(f"Unsupported metric panel: {metric}")
    return save_panel(fig, output_stem)


def matched_bar_width(template, method_count):
    """Preserve the physical bar width of the six-method covered panel."""
    def effective_axes_width_mm(panel_template):
        allocated_width = (
            panel_template.width_mm * panel_template.axes_rect[2]
        )
        allocated_height = (
            panel_template.height_mm * panel_template.axes_rect[3]
        )
        return min(
            allocated_width,
            allocated_height / panel_template.axes_box_aspect,
        )

    reference = PANEL_TEMPLATES[DEFAULT_TEMPLATE]
    reference_axes_mm = effective_axes_width_mm(reference)
    reference_span = len(METHOD_ORDER) + 0.1
    physical_bar_mm = BAR_WIDTH / reference_span * reference_axes_mm

    target = PANEL_TEMPLATES[template]
    target_axes_mm = effective_axes_width_mm(target)
    target_span = method_count + 0.1
    return physical_bar_mm / target_axes_mm * target_span


def plot_all_panels(
    gem_data,
    non_gem_data,
    seeds,
    output_dir,
    covered_template,
    uncovered_template,
):
    """Create six data panels, each with text and no-text exports."""
    panel_specs = (
        ("set1_aupr", gem_data, "AUPR", 0.85, METHOD_ORDER, covered_template),
        ("set1_auroc", gem_data, "AUROC", 1.00, METHOD_ORDER, covered_template),
        ("set1_hits", gem_data, "hits", 0.55, METHOD_ORDER, covered_template),
        ("set2_aupr", non_gem_data, "AUPR", 0.70, NON_GEM_ORDER, uncovered_template),
        ("set2_auroc", non_gem_data, "AUROC", 1.00, NON_GEM_ORDER, uncovered_template),
        ("set2_hits", non_gem_data, "hits", 0.35, NON_GEM_ORDER, covered_template),
    )
    outputs = []
    for panel_name, data, metric, y_max, method_order, panel_template in panel_specs:
        base_stem = output_dir / f"external_benchmark_{panel_name}"
        bar_width = matched_bar_width(panel_template, len(method_order))
        outputs.extend(plot_panel(
            data,
            metric,
            seeds,
            y_max,
            base_stem,
            panel_template,
            method_order,
            bar_width,
            show_text=True,
        ))
        outputs.extend(plot_panel(
            data,
            metric,
            seeds,
            y_max,
            Path(f"{base_stem}_no_text"),
            panel_template,
            method_order,
            bar_width,
            show_text=False,
        ))
    return outputs


def run(repo_root, output_dir, seeds, template, uncovered_template):
    """Load data, write source tables, and render all standalone panels."""
    gem_data, non_gem_data = load_benchmark_data(repo_root, seeds)
    write_source_data(output_dir, gem_data, non_gem_data)
    outputs = plot_all_panels(
        gem_data,
        non_gem_data,
        seeds,
        Path(output_dir),
        template,
        uncovered_template,
    )
    print(f"GEM-covered: {len(gem_data)} source rows")
    print(f"GEM-uncovered: {len(non_gem_data)} source rows")
    print(f"Template: {template}")
    print(f"GEM-uncovered template: {uncovered_template}")
    for output in outputs:
        print(f"Saved {output}")


def main(argv=None):
    """Generate the six standalone external benchmark panels."""
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be non-empty and contain no duplicates.")
    repo_root = args.repo_root or PROJECT_ROOT
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    run(
        repo_root,
        output_dir,
        seeds,
        args.template,
        args.uncovered_template,
    )


if __name__ == "__main__":
    main()
