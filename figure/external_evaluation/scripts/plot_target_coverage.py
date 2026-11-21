#!/usr/bin/env python
"""Hits@30 target overlap.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_target_coverage.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/mechanistic_comparison/target_overlap/.
Directional coverage of unique genes recovered within Hits@30; KO and OE combined, summarized over five seeds.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/mechanistic_comparison/target_overlap'
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_TEMPLATE = "square_v1"
TEST_SET = "gem_gene/head_corrupt"
HIT_K = 30

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    EDGE_COLOR,
    PANEL_TEMPLATES,
    create_panel,
    save_panel,
)


@dataclass(frozen=True)
class MethodSpec:
    """Rank-table location and model selector for one method."""

    name: str
    result_dir: str
    model: str
    tick_label: str


METHOD_SPECS = (
    MethodSpec("FSEOF", "fseof", "fseof", "FSEOF"),
    MethodSpec("FVSEOF", "fvseof", "fvseof", "FVSEOF"),
    MethodSpec("iBridge", "ibridge", "ibridge", "iBridge"),
    MethodSpec("DeepVCF", "deepvcf", "fold_1", "DeepVCF"),
    MethodSpec("DeepVCF+BE", "deepvcf", "ensemble", "+BE"),
    MethodSpec(
        "DeepVCF+BE+MP", "deepvcf_mp", "ensemble",
        "+BE+MP",
    ),
)
METHOD_ORDER = tuple(spec.name for spec in METHOD_SPECS)
TICK_LABELS = tuple(spec.tick_label for spec in METHOD_SPECS)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--template", choices=tuple(PANEL_TEMPLATES), default=DEFAULT_TEMPLATE,
    )
    return parser.parse_args(argv)


def read_target_set(path, model, seed):
    """Read unique target genes recovered within Hits@30."""
    if not path.is_file():
        raise FileNotFoundError(f"Gene-rank table not found: {path}")
    targets = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"model", "seed", "test_set", "gene", "gene_rank"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            if row["model"] != model or row["test_set"] != TEST_SET:
                continue
            if int(row["seed"]) != seed:
                raise ValueError(f"Seed mismatch in {path}")
            if int(row["gene_rank"]) <= HIT_K:
                targets.add(row["gene"])
    return targets


def calculate_coverage(repo_root, seeds):
    """Calculate asymmetric pairwise target coverage for every seed."""
    target_sets = {}
    for seed in seeds:
        for spec in METHOD_SPECS:
            path = (
                repo_root / "results" / "external_evaluation"
                / spec.result_dir / f"seed_{seed}"
                / "gene_corruption_ranks.tsv"
            )
            targets = read_target_set(path, spec.model, seed)
            if not targets:
                raise ValueError(
                    f"No Hits@{HIT_K} targets for {spec.name}, seed {seed}"
                )
            target_sets[(seed, spec.name)] = targets

    records = []
    for seed in seeds:
        for reference in METHOD_ORDER:
            reference_targets = target_sets[(seed, reference)]
            for covering in METHOD_ORDER:
                covering_targets = target_sets[(seed, covering)]
                intersection = len(reference_targets & covering_targets)
                records.append({
                    "seed": seed,
                    "reference_method": reference,
                    "covering_method": covering,
                    "reference_target_count": len(reference_targets),
                    "covering_target_count": len(covering_targets),
                    "intersection_count": intersection,
                    "directional_coverage": intersection / len(reference_targets),
                })
    return pd.DataFrame.from_records(records)


def summarize_coverage(seed_data):
    """Summarize coverage and intersection counts across seeds."""
    return (
        seed_data.groupby(
            ["reference_method", "covering_method"], observed=True,
        )
        .agg(
            coverage_mean=("directional_coverage", "mean"),
            coverage_std=("directional_coverage", "std"),
            intersection_mean=("intersection_count", "mean"),
            intersection_std=("intersection_count", "std"),
            reference_targets_mean=("reference_target_count", "mean"),
            reference_targets_std=("reference_target_count", "std"),
            n_seeds=("seed", "size"),
        )
        .reset_index()
    )


def draw_panel(
    ax, colorbar_ax, summary, show_text=True, annotation="detailed",
    show_annotations=None, show_axis_ticks=False,
):
    """Draw the directional target-coverage heatmap."""
    if show_annotations is None:
        show_annotations = show_text
    indexed = summary.set_index(["reference_method", "covering_method"])
    coverage = np.array([
        [indexed.loc[(row, col), "coverage_mean"] for col in METHOD_ORDER]
        for row in METHOD_ORDER
    ])
    image = ax.imshow(
        coverage * 100.0, cmap="Purples", vmin=0, vmax=100,
        interpolation="nearest", aspect="equal",
    )
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    if show_text:
        ax.set_xticklabels(TICK_LABELS, rotation=42, ha="right", fontsize=8.2)
        ax.set_yticklabels(TICK_LABELS, fontsize=8.2)
        ax.set_xlabel("Covering method")
        ax.set_ylabel("Reference method")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.tick_params(
        length=4.0 if show_axis_ticks else 0,
        width=0.8,
        direction="out",
        color=EDGE_COLOR,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_annotations:
        for row_index, reference in enumerate(METHOD_ORDER):
            for col_index, covering in enumerate(METHOD_ORDER):
                values = indexed.loc[(reference, covering)]
                if annotation == "mean_only":
                    label = f"{100 * values['coverage_mean']:.0f}%"
                elif reference == covering:
                    label = f"n={values['reference_targets_mean']:.1f}"
                else:
                    label = (
                        f"{100 * values['coverage_mean']:.0f}±"
                        f"{100 * values['coverage_std']:.0f}%\n"
                        f"({values['intersection_mean']:.1f})"
                    )
                color = "white" if coverage[row_index, col_index] >= 0.58 else EDGE_COLOR
                ax.text(
                    col_index, row_index, label,
                    ha="center", va="center", color=color, fontsize=5.4,
                )

    colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
    colorbar.outline.set_visible(False)
    colorbar.set_ticks(np.arange(0, 101, 20))
    colorbar.ax.tick_params(length=2.5, width=0.8, labelsize=7.5)
    colorbar.set_label(
        "Directional target coverage (%)" if show_text else "",
        size=8.5,
        labelpad=4,
    )
    if not show_text:
        colorbar.ax.set_yticklabels([])


def plot_panel(
    summary, output_stem, template, show_text=True, annotation="detailed",
    show_annotations=None, show_axis_ticks=False,
):
    """Render and save one target-coverage panel."""
    fig, ax = create_panel(template)
    # A square matrix plus its colorbar needs dedicated space inside the fixed
    # square_v1 canvas; keep all artists within the export-safe margins.
    ax.set_position((0.235, 0.285, 0.59, 0.66))
    ax.set_box_aspect(1.0)
    colorbar_ax = fig.add_axes((0.845, 0.315, 0.022, 0.60))
    draw_panel(
        ax, colorbar_ax, summary, show_text=show_text,
        annotation=annotation, show_annotations=show_annotations,
        show_axis_ticks=show_axis_ticks,
    )
    return save_panel(fig, output_stem)


def run(repo_root, output_dir, seeds, template):
    """Calculate, export, and plot directional target coverage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_data = calculate_coverage(repo_root, seeds)
    summary = summarize_coverage(seed_data)
    seed_data.to_csv(
        output_dir / "hits30_directional_target_coverage_seed_data.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "hits30_directional_target_coverage_summary.csv",
        index=False,
    )

    stem = output_dir / "hits30_directional_target_coverage"
    outputs = []
    outputs.extend(plot_panel(summary, stem, template, show_text=True))
    outputs.extend(plot_panel(
        summary, Path(f"{stem}_mean_only"), template,
        show_text=True, annotation="mean_only",
    ))
    outputs.extend(plot_panel(
        summary, Path(f"{stem}_no_text"), template, show_text=False,
    ))
    outputs.extend(plot_panel(
        summary, Path(f"{stem}_percent_only"), template,
        show_text=False, annotation="mean_only", show_annotations=True,
        show_axis_ticks=True,
    ))
    print(f"Dataset: {TEST_SET}; KO and OE combined")
    print(f"Seeds: {' '.join(map(str, seeds))}")
    print(f"Template: {template}")
    for method in METHOD_ORDER:
        values = seed_data.loc[
            seed_data["reference_method"].eq(method)
            & seed_data["covering_method"].eq(method),
            "reference_target_count",
        ]
        print(f"{method}: {values.mean():.1f} ± {values.std(ddof=1):.1f} unique targets")
    for output in outputs:
        print(f"Saved {output}")


def main(argv=None):
    """Generate the external Hits@30 directional-coverage panel."""
    args = parse_args(argv)
    run(
        args.repo_root.resolve(), args.output_dir.resolve(),
        tuple(args.seeds), args.template,
    )


if __name__ == "__main__":
    main()
