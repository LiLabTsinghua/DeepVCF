#!/usr/bin/env python3
"""Cross-species KG protein embeddings.

Run from the repository root in deepvcf:
    python figure/embedding_visualization/scripts/plot_cross_kg_embeddings.py
    python figure/embedding_visualization/scripts/plot_cross_kg_embeddings.py --replot-only

Outputs: figure/embedding_visualization/outputs/cross_kg/.
1 x 4: ECO+SCE, ECO+CGL, ECO+BSU, ECO+PPU. Inputs: results/drn/cross_species/ECO_<target>/enhanced/seed_<seed>/drn_trained_model.pt and data/processed/kg/<species>/kg_enhanced.txt. Categories distinguish ECO/target species and GEM coverage.
Defaults: checkpoint seed 1, zero-based fold 0, projection seed 42.
L2-normalized embeddings are projected independently with PCA, t-SNE, and
UMAP. --methods selects a subset. --replot-only redraws the exported CSV
coordinates without loading checkpoints or fitting new projections; seed,
fold, and projection-seed options apply only when computing new coordinates.
Exports category counts, exact coordinates, and labeled/no-text SVG/PNG.
Run projection computations outside the sandbox in the GPU environment;
these sklearn/UMAP implementations themselves use CPU. Nonlinear projection
orientation and scale are arbitrary; compare organization within each panel.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/drn/cross_species"
DEFAULT_KG_ROOT = PROJECT_ROOT / "data/processed/kg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/cross_kg'

TARGET_SPECIES = ("SCE", "CGL", "BSU", "PPU")
PAIR_LABELS = {species: f"ECO+{species}" for species in TARGET_SPECIES}
CATEGORY_ORDER = (
    "eco_gem_covered",
    "eco_gem_uncovered",
    "target_gem_covered",
    "target_gem_uncovered",
)
CATEGORY_LABELS = {
    "eco_gem_covered": "ECO GEM-covered",
    "eco_gem_uncovered": "ECO GEM-uncovered",
    "target_gem_covered": "Target GEM-covered",
    "target_gem_uncovered": "Target GEM-uncovered",
}
CATEGORY_COLORS = {
    "eco_gem_covered": "#D45B72",
    "eco_gem_uncovered": "#E59E3F",
    "target_gem_covered": "#3A6EA5",
    "target_gem_uncovered": "#6F9E6E",
}
SCATTER_AREA = 1.15

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    MM_PER_INCH,
    WIDE_V1,
    apply_publication_style,
    save_panel,
)


from figure.figure_io import write_csv
from figure.embedding_visualization.scripts.embedding_common import (
    PROJECTION_METHODS, PROJECTION_LABELS,
    l2_normalize, project_embeddings, load_coordinate_groups,
    load_kg_categories,
)


@dataclass(frozen=True)
class EmbeddingData:
    """Protein embeddings and their species/GEM categories."""

    species: str
    entities: np.ndarray
    categories: np.ndarray
    embeddings: np.ndarray


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_KG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=PROJECTION_METHODS,
        default=list(PROJECTION_METHODS),
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Reuse exported projection coordinates and only redraw figures.",
    )
    return parser.parse_args(argv)


def load_gene_categories(kg_path: Path, species: str):
    """Split one species' proteins by presence of a Catalyzes edge."""
    categories = load_kg_categories(kg_path, species)
    covered, uncovered = categories['gem_covered_gene'], categories['gem_uncovered_gene']
    if not covered or not uncovered:
        raise ValueError(f'Empty GEM category for {species} in {kg_path}')
    return covered, uncovered


def load_pair_embeddings(results_root, kg_root, target_species, seed, fold):
    """Load one ECO-target checkpoint and select its four protein groups."""
    checkpoint_path = (
        results_root / f"ECO_{target_species}" / "enhanced"
        / f"seed_{seed}" / "drn_trained_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    eco_covered, eco_uncovered = load_gene_categories(
        kg_root / "ECO/kg_enhanced.txt", "ECO",
    )
    target_covered, target_uncovered = load_gene_categories(
        kg_root / target_species / "kg_enhanced.txt", target_species,
    )
    category_sets = {
        "eco_gem_covered": eco_covered,
        "eco_gem_uncovered": eco_uncovered,
        "target_gem_covered": target_covered,
        "target_gem_uncovered": target_uncovered,
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    node_to_id = checkpoint["kg_metadata"]["node_to_id"]
    states = [value for key, value in checkpoint["folds_state_dict"].items()
                   if key.endswith(".drn.node_emb")]
    if not 0 <= fold < len(states):
        raise ValueError(f"Invalid fold {fold} for ECO_{target_species}")
    embeddings = l2_normalize(states[fold].detach().cpu().numpy())
    if embeddings.shape[0] != len(node_to_id):
        raise ValueError(f"Embedding/mapping mismatch for ECO_{target_species}")

    entities = []
    categories = []
    indices = []
    for category in CATEGORY_ORDER:
        present = sorted(category_sets[category].intersection(node_to_id))
        if len(present) != len(category_sets[category]):
            raise ValueError(
                f"ECO_{target_species} is missing "
                f"{len(category_sets[category]) - len(present)} {category} genes"
            )
        for entity in present:
            entities.append(entity)
            categories.append(category)
            indices.append(int(node_to_id[entity]))
    return EmbeddingData(
        species=target_species,
        entities=np.asarray(entities, dtype=str),
        categories=np.asarray(categories, dtype=str),
        embeddings=embeddings[np.asarray(indices, dtype=int)],
    )


def export_coordinates(method, projected_by_species, output_dir):
    rows = []
    for species in TARGET_SPECIES:
        data, coordinates = projected_by_species[species]
        for entity, category, coordinate in zip(
            data.entities, data.categories, coordinates,
        ):
            rows.append({
                "projection": PROJECTION_LABELS[method],
                "cross_kg": PAIR_LABELS[species],
                "entity": entity,
                "category": CATEGORY_LABELS[category],
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
            })
    write_csv(output_dir / f"cross_kg_embedding_{method}_coordinates.csv", rows)


def load_exported_coordinates(method, output_dir):
    """Load the exact coordinates exported for the cross-KG comparison."""
    groups = load_coordinate_groups(
        output_dir / f'cross_kg_embedding_{method}_coordinates.csv',
        'cross_kg', PAIR_LABELS, CATEGORY_LABELS, PROJECTION_LABELS[method],
    )
    return {species: (EmbeddingData(
        species=species, entities=entities, categories=categories,
        embeddings=np.empty((len(entities), 0), dtype=float),
    ), coordinates) for species, (entities, categories, coordinates) in groups.items()}


def plot_projection(method, projected_by_species, output_dir, show_text):
    """Draw four cross-KG projections in one wide 1 x 4 figure."""
    apply_publication_style()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(WIDE_V1.width_mm / MM_PER_INCH, WIDE_V1.height_mm / MM_PER_INCH),
        facecolor="white",
        squeeze=False,
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.16, top=0.72, wspace=0.06)
    for column, species in enumerate(TARGET_SPECIES):
        ax = axes[0, column]
        data, coordinates = projected_by_species[species]
        for category in CATEGORY_ORDER[::-1]:
            mask = data.categories == category
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=SCATTER_AREA,
                marker="o",
                color=CATEGORY_COLORS[category],
                edgecolors="none",
                linewidths=0,
                alpha=1.0,
                rasterized=False,
            )
        ax.margins(0.035)
        ax.set_box_aspect(1)
        ax.set_axis_off()
        if show_text:
            ax.set_title(PAIR_LABELS[species], fontsize=9, pad=3)

    stem = output_dir / f"cross_kg_embedding_{method}"
    if show_text:
        handles = [
            Line2D(
                [], [], linestyle="none", marker="o", markersize=4.0,
                markerfacecolor=CATEGORY_COLORS[category],
                markeredgecolor="none", label=CATEGORY_LABELS[category],
            )
            for category in CATEGORY_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.505, 0.955),
            ncol=2,
            fontsize=7.2,
            handletextpad=0.35,
            columnspacing=1.0,
            borderaxespad=0,
        )
    else:
        stem = Path(f"{stem}_no_text")
    return save_panel(fig, stem)


def run(
    results_root, kg_root, output_dir, seed, fold, projection_seed, methods,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        species: load_pair_embeddings(
            results_root, kg_root, species, seed, fold,
        )
        for species in TARGET_SPECIES
    }
    count_rows = []
    for species, data in selected.items():
        for category in CATEGORY_ORDER:
            count_rows.append({
                "cross_kg": PAIR_LABELS[species],
                "category": CATEGORY_LABELS[category],
                "count": int(np.count_nonzero(data.categories == category)),
                "seed": seed,
                "fold": fold,
            })
    write_csv(output_dir / "cross_kg_embedding_category_counts.csv", count_rows)

    for method in methods:
        projected = {}
        for species, data in selected.items():
            print(f"Projecting {PAIR_LABELS[species]} with {PROJECTION_LABELS[method]}...")
            projected[species] = (
                data,
                project_embeddings(data.embeddings, method, projection_seed),
            )
        export_coordinates(method, projected, output_dir)
        plot_projection(method, projected, output_dir, show_text=True)
        plot_projection(method, projected, output_dir, show_text=False)


def main(argv=None):
    args = parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("--methods must not contain duplicates.")
    if args.replot_only:
        output_dir = args.output_dir.resolve()
        for method in args.methods:
            projected = load_exported_coordinates(method, output_dir)
            plot_projection(method, projected, output_dir, show_text=True)
            plot_projection(method, projected, output_dir, show_text=False)
        print(f"Redrawn from saved coordinates in {output_dir}")
        return
    run(
        args.results_root.resolve(),
        args.kg_root.resolve(),
        args.output_dir.resolve(),
        args.seed,
        args.fold,
        args.projection_seed,
        args.methods,
    )
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
