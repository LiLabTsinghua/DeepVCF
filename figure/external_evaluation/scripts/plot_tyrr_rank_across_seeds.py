#!/usr/bin/env python
"""tyrR ranking.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_tyrr_rank_across_seeds.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/tyrr/.
tyrR knockout rankings for 3dhsk across five seeds, comparing FoldUnit 1 and the two ensemble conditions.
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
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/external_evaluation"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/tyrr'
SEEDS = (1, 2, 3, 4, 5)
TEST_SET = "non_gem_gene/head_corrupt"
TARGET_GENE = "ECO_Protein:P07604"
TARGET_RELATION = "knock out"
TARGET_METABOLITE = "Metabolite:3dhsk_c"

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    DATA_LINEWIDTH,
    EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    METHOD_COLORS,
    PNG_DPI,
    create_panel,
)


METHODS = (
    ("DeepVCF", "deepvcf", "fold_1", "o"),
    ("DeepVCF+BE", "deepvcf", "ensemble", "o"),
    (
        "DeepVCF+BE+MP",
        "deepvcf_mp",
        "ensemble",
        "o",
    ),
)
METHOD_LABELS = {
    "DeepVCF": "DeepVCF",
    "DeepVCF+BE": "+BE",
    "DeepVCF+BE+MP": "+BE+MP",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def load_source_data(results_root: Path) -> pd.DataFrame:
    """Load the selected intervention rank for every method and seed."""
    records = []
    for method, result_dir, model_row, _ in METHODS:
        for seed in SEEDS:
            path = (
                results_root / result_dir / f"seed_{seed}"
                / "gene_corruption_ranks.tsv"
            )
            frame = pd.read_csv(path, sep="\t")
            selected = frame.loc[
                frame["test_set"].eq(TEST_SET)
                & frame["model"].eq(model_row)
                & frame["gene"].eq(TARGET_GENE)
                & frame["relation"].eq(TARGET_RELATION)
                & frame["metabolite"].eq(TARGET_METABOLITE)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one tyrR KO-to-3dhsk row in {path}; "
                    f"found {len(selected)}."
                )
            records.append({
                "method": method,
                "seed": seed,
                "gene_rank": int(selected.iloc[0]["gene_rank"]),
                "gene": TARGET_GENE,
                "relation": TARGET_RELATION,
                "metabolite": TARGET_METABOLITE,
            })
    return pd.DataFrame.from_records(records)


def draw_panel(data: pd.DataFrame, *, show_text: bool):
    """Draw three model trajectories across seeds."""
    fig, ax = create_panel("square_v1")
    for method, _, _, marker in METHODS:
        subset = data.loc[data["method"].eq(method)].sort_values("seed")
        ax.plot(
            subset["seed"],
            subset["gene_rank"],
            color=METHOD_COLORS[method],
            linewidth=DATA_LINEWIDTH,
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.set_xlim(0.75, 5.25)
    ax.set_xticks(SEEDS)
    ax.set_ylim(0.65, 4.35)
    ax.set_yticks((1, 2, 3, 4))
    if show_text:
        ax.set_xlabel("Seed")
        ax.set_ylabel("Gene rank")
        ax.legend(
            loc="upper right",
            fontsize=8.5,
            handlelength=1.5,
            handletextpad=0.45,
            labelspacing=0.45,
        )
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(False)
    return fig


def save_panel(fig, stem: Path):
    """Export editable SVG and 600-dpi PNG files."""
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    plt.close(fig)


def run(results_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_source_data(results_root)
    data.to_csv(output_dir / "tyrr_ko_3dhsk_rank_source_data.csv", index=False)
    stem = output_dir / "tyrr_ko_3dhsk_rank_across_seeds"
    save_panel(draw_panel(data, show_text=True), stem)
    save_panel(draw_panel(data, show_text=False), Path(f"{stem}_no_text"))
    print(data[["method", "seed", "gene_rank"]].to_string(index=False))
    print(f"Saved {stem}.svg/.png and no-text variants")


def main(argv=None):
    args = parse_args(argv)
    run(args.results_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
