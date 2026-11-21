#!/usr/bin/env python
"""Positive sample composition by GEM coverage.

Run from the repository root in deepvcf:
    python figure/cross_species_transfer/scripts/plot_positive_sample_counts.py

Outputs: figure/cross_species_transfer/outputs/sample_composition/.
Inputs: data/processed/cf/datasets/<species>_dataset.txt and data/processed/kg/<species>/kg_enhanced.txt. Counts rows for SCE, CGL, BSU, PPU. GEM-covered genes are heads of Catalyzes relations in the corresponding enhanced KG.
Exports labeled/no-text SVG/600-dpi PNG and source tables.
This script reads existing datasets/results and performs no model training.
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
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


# =============================================================================
# Cross-Species Positive-Sample Composition
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    BAR_EDGE_WIDTH,
    EDGE_COLOR,
    create_panel,
    save_panel,
    style_axis,
)

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "cf" / "datasets"
DEFAULT_KG_ROOT = PROJECT_ROOT / "data" / "processed" / "kg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/sample_composition'

SPECIES_ORDER = ("SCE", "CGL", "BSU", "PPU")
GROUP_ORDER = ("GEM-covered", "GEM-uncovered")
GROUP_COLORS = {
    "GEM-covered": "#D45B72",
    "GEM-uncovered": "#E59E3F",
}


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence. If ``None``, use ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def load_gem_genes(kg_path):
    """Load genes represented in a species GEM from catalysis relations.

    Args:
        kg_path: Three-column enhanced-KG file.

    Returns:
        set[str]: Protein nodes occurring as heads of ``Catalyzes`` relations.

    Raises:
        FileNotFoundError: If the KG file does not exist.
        ValueError: If the KG contains no catalysis relations.
    """
    kg_path = Path(kg_path)
    if not kg_path.is_file():
        raise FileNotFoundError(f"KG file not found: {kg_path}")
    frame = pd.read_csv(
        kg_path,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
    )
    gem_genes = set(frame.loc[frame["relation"].eq("Catalyzes"), "head"])
    if not gem_genes:
        raise ValueError(f"No Catalyzes genes found in {kg_path}")
    return gem_genes


def load_source_data(dataset_root, kg_root):
    """Count positive samples by species and GEM coverage.

    Args:
        dataset_root: Directory containing species positive datasets.
        kg_root: Directory containing species enhanced KGs.

    Returns:
        pandas.DataFrame: Long-form sample-count table.

    Raises:
        FileNotFoundError: If a required dataset or KG is missing.
        ValueError: If a dataset is malformed or contains uncovered genes.
    """
    records = []
    for species in SPECIES_ORDER:
        dataset_path = Path(dataset_root) / f"{species}_dataset.txt"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        dataset = pd.read_csv(
            dataset_path,
            sep="\t",
            header=None,
            names=["gene", "relation", "metabolite"],
            dtype=str,
        )
        gem_genes = load_gem_genes(
            Path(kg_root) / species / "kg_enhanced.txt"
        )
        is_gem_covered = dataset["gene"].isin(gem_genes)
        counts = {
            "GEM-covered": int(is_gem_covered.sum()),
            "GEM-uncovered": int((~is_gem_covered).sum()),
        }
        for group in GROUP_ORDER:
            records.append({
                "species": species,
                "gene_group": group,
                "positive_samples": counts[group],
            })
    return pd.DataFrame.from_records(records)


def plot_counts(source_data, output_dir, show_text=True):
    """Draw and export the grouped positive-sample count chart.

    Args:
        source_data: Long-form table returned by ``load_source_data``.
        output_dir: Directory receiving SVG, PNG, and source-data CSV files.

    Returns:
        tuple[pathlib.Path, pathlib.Path]: SVG and PNG output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = create_panel("square_v1")
    x = np.arange(len(SPECIES_ORDER), dtype=float)
    width = 0.34

    for offset, group in zip((-0.5, 0.5), GROUP_ORDER):
        group_data = (
            source_data.loc[source_data["gene_group"].eq(group)]
            .set_index("species")
            .reindex(SPECIES_ORDER)
        )
        values = group_data["positive_samples"].to_numpy(dtype=int)
        bars = ax.bar(
            x + offset * width,
            values,
            width=width,
            color=GROUP_COLORS[group],
            edgecolor=EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=group,
            alpha=1.0,
            zorder=2,
        )
        if show_text:
            ax.bar_label(
                bars,
                labels=[str(value) for value in values],
                padding=3,
                fontsize=11,
            )

    ax.set_xticks(x, SPECIES_ORDER)
    ax.set_xlim(-0.55, len(SPECIES_ORDER) - 0.45)
    ax.set_ylim(0, 380)
    ax.set_yticks([0, 100, 200, 300])
    ax.tick_params(axis="x", length=0, pad=5)
    if show_text:
        ax.set_ylabel("Positive samples")
        ax.legend(loc="upper right", fontsize=10.5)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    ax.grid(False)

    output_stem = output_dir / (
        "positive_sample_counts"
        if show_text
        else "positive_sample_counts_no_text"
    )
    return save_panel(fig, output_stem)


def main(argv=None):
    """Build source data and export the positive-sample composition figure."""
    args = parse_args(argv)
    source_data = load_source_data(args.dataset_root, args.kg_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(
        args.output_dir / "positive_sample_counts_source_data.csv",
        index=False,
    )
    outputs = []
    for show_text in (True, False):
        outputs.extend(plot_counts(source_data, args.output_dir, show_text))
    print(source_data.to_string(index=False))
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
