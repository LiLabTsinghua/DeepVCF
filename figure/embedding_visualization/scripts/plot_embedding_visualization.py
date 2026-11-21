#!/usr/bin/env python3
"""ECO DRN embedding comparison.

Run from the repository root in deepvcf:
    python figure/embedding_visualization/scripts/plot_embedding_visualization.py
    python figure/embedding_visualization/scripts/plot_embedding_visualization.py --replot-only

Outputs: figure/embedding_visualization/outputs/eco/.
1 x 3: Random, Basic, Enhanced. Inputs: results/drn/ECO/<variant>/seed_<seed>/drn_trained_model.pt and data/processed/kg/ECO/kg_enhanced.txt. Categories: metabolites, GEM-covered genes, GEM-uncovered genes; Basic lacks the last group.
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

os.environ.setdefault('MPLCONFIGDIR', '/tmp/deepvcf-matplotlib')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "drn" / "ECO"
DEFAULT_KG_PATH = PROJECT_ROOT / "data" / "processed" / "kg" / "ECO" / "kg_enhanced.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/eco'

VARIANTS = ("random", "basic", "enhanced")
VARIANT_LABELS = {
    "random": "Random",
    "basic": "Basic",
    "enhanced": "Enhanced",
}
CATEGORY_ORDER = ("metabolite", "gem_covered_gene", "gem_uncovered_gene")
CATEGORY_LABELS = {
    "metabolite": "Metabolites",
    "gem_covered_gene": "GEM-covered genes",
    "gem_uncovered_gene": "GEM-uncovered genes",
}
CATEGORY_COLORS = {
    "metabolite": "#4FA3A5",
    "gem_covered_gene": "#D45B72",
    "gem_uncovered_gene": "#E59E3F",
}
CATEGORY_MARKERS = {
    "metabolite": "o",
    "gem_covered_gene": "o",
    "gem_uncovered_gene": "o",
}
SCATTER_AREA = 2.0

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import create_small_multiples, save_panel  # noqa: E402


from figure.figure_io import write_csv
from figure.embedding_visualization.scripts.embedding_common import (
    PROJECTION_METHODS, PROJECTION_LABELS,
    l2_normalize, project_embeddings, load_coordinate_groups,
    load_kg_categories,
)


@dataclass(frozen=True)
class EmbeddingData:
    """Selected entity embeddings and their biological categories."""

    variant: str
    entities: np.ndarray
    categories: np.ndarray
    embeddings: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--kg-path", type=Path, default=DEFAULT_KG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument('--methods', nargs='+', choices=PROJECTION_METHODS,
                        default=list(PROJECTION_METHODS))
    parser.add_argument('--replot-only', action='store_true',
                        help='Redraw from saved coordinates without loading checkpoints.')
    return parser.parse_args(argv)


def load_category_sets(kg_path: Path) -> dict[str, set[str]]:
    """Define the ECO metabolite and GEM coverage categories."""
    categories = load_kg_categories(kg_path, 'ECO')
    if any(not values for values in categories.values()):
        raise ValueError('One or more requested entity categories are empty.')
    return categories


def load_selected_embeddings(
    checkpoint_path: Path,
    variant: str,
    fold: int,
    category_sets: dict[str, set[str]],
) -> EmbeddingData:
    """Load one fold and retain only the three requested entity categories."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DRN checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    node_to_id = checkpoint["kg_metadata"]["node_to_id"]
    fold_states = [value for key, value in checkpoint["folds_state_dict"].items()
                   if key.endswith(".drn.node_emb")]
    if not 0 <= fold < len(fold_states):
        raise ValueError(
            f"Fold {fold} is invalid for {variant}; expected 0-{len(fold_states)-1}."
        )
    embeddings = l2_normalize(
        fold_states[fold].detach().cpu().numpy()
    )
    if embeddings.shape[0] != len(node_to_id):
        raise ValueError(f"Embedding/mapping length mismatch for {variant}.")

    entities: list[str] = []
    categories: list[str] = []
    indices: list[int] = []
    for category in CATEGORY_ORDER:
        present = sorted(category_sets[category].intersection(node_to_id))
        for entity in present:
            entities.append(entity)
            categories.append(category)
            indices.append(int(node_to_id[entity]))

    if not indices:
        raise ValueError(f"No requested entities are present for {variant}.")
    selected = embeddings[np.asarray(indices, dtype=int)]
    return EmbeddingData(
        variant=variant,
        entities=np.asarray(entities, dtype=str),
        categories=np.asarray(categories, dtype=str),
        embeddings=selected,
    )


def export_coordinates(
    method: str,
    projected_by_variant: dict[str, tuple[EmbeddingData, np.ndarray]],
    output_dir: Path,
) -> None:
    """Export the exact coordinates drawn in one projection figure."""
    rows: list[dict] = []
    for variant in VARIANTS:
        data, coordinates = projected_by_variant[variant]
        for entity, category, (x_value, y_value) in zip(
            data.entities, data.categories, coordinates,
        ):
            rows.append({
                "projection": PROJECTION_LABELS[method],
                "variant": VARIANT_LABELS[variant],
                "entity": entity,
                "category": CATEGORY_LABELS[category],
                "x": float(x_value),
                "y": float(y_value),
            })
    write_csv(output_dir / f"embedding_{method}_coordinates.csv", rows)


def load_exported_coordinates(method, output_dir):
    """Load the exact coordinates exported for this comparison."""
    groups = load_coordinate_groups(
        output_dir / f"embedding_{method}_coordinates.csv", 'variant', VARIANT_LABELS,
        CATEGORY_LABELS, PROJECTION_LABELS[method],
    )
    return {variant: (EmbeddingData(
        variant=variant, entities=entities, categories=categories,
        embeddings=np.empty((len(entities), 0), dtype=float),
    ), coordinates) for variant, (entities, categories, coordinates) in groups.items()}


def plot_projection(
    method: str,
    projected_by_variant: dict[str, tuple[EmbeddingData, np.ndarray]],
    output_dir: Path,
    show_text: bool,
) -> tuple[Path, Path]:
    """Plot Random, Basic, and Enhanced as a 1 x 3 small-multiples figure."""
    fig, axes = create_small_multiples(
        "small_multiples_v1",
        nrows=1,
        ncols=3,
        sharex=False,
        sharey=False,
        panel_width_mm=42.0,
        panel_height_mm=42.0,
    )
    # These embedding panels have no coordinate axes. Preserve the 42 mm data
    # region while removing the template's otherwise unused lower whitespace.
    if show_text:
        canvas_height_mm = 58.0
        bottom_margin_mm = 4.0
        top_margin_mm = 12.0
    else:
        canvas_height_mm = 48.0
        bottom_margin_mm = 3.0
        top_margin_mm = 3.0
    fig.set_figheight(canvas_height_mm / 25.4)
    fig.subplots_adjust(
        bottom=bottom_margin_mm / canvas_height_mm,
        top=1.0 - top_margin_mm / canvas_height_mm,
    )
    for column, variant in enumerate(VARIANTS):
        ax = axes[0, column]
        data, coordinates = projected_by_variant[variant]
        for category in CATEGORY_ORDER[::-1]:
            mask = data.categories == category
            if not np.any(mask):
                continue
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=SCATTER_AREA,
                marker=CATEGORY_MARKERS[category],
                color=CATEGORY_COLORS[category],
                edgecolors="none",
                linewidths=0,
                alpha=1.0,
                rasterized=False,
            )
        ax.margins(0.035)
        ax.set_axis_off()
        if show_text:
            ax.set_title(VARIANT_LABELS[variant], fontsize=10, pad=4)

    output_stem = output_dir / f"embedding_{method}_comparison"
    if show_text:
        handles = [
            Line2D(
                [], [], linestyle="none",
                marker=CATEGORY_MARKERS[category], markersize=4.2,
                markerfacecolor=CATEGORY_COLORS[category],
                markeredgecolor="none",
                label=CATEGORY_LABELS[category],
            )
            for category in CATEGORY_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.985),
            ncol=3,
            fontsize=8.0,
            handletextpad=0.35,
            columnspacing=0.9,
            borderaxespad=0,
        )
    else:
        output_stem = Path(f"{output_stem}_no_text")
    return save_panel(fig, output_stem)


def run_analysis(
    results_root: Path,
    kg_path: Path,
    output_dir: Path,
    seed: int = 1,
    fold: int = 0,
    projection_seed: int = 42,
    methods=PROJECTION_METHODS,
) -> dict[str, dict[str, tuple[EmbeddingData, np.ndarray]]]:
    """Load one reproducible checkpoint and generate all three figures."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    category_sets = load_category_sets(kg_path.resolve())
    selected = {
        variant: load_selected_embeddings(
            results_root.resolve() / variant / f"seed_{seed}" / "drn_trained_model.pt",
            variant,
            fold,
            category_sets,
        )
        for variant in VARIANTS
    }

    count_rows = []
    for variant, data in selected.items():
        for category in CATEGORY_ORDER:
            count_rows.append({
                "variant": VARIANT_LABELS[variant],
                "category": CATEGORY_LABELS[category],
                "count": int(np.count_nonzero(data.categories == category)),
                "seed": seed,
                "fold": fold,
            })
    write_csv(output_dir / "embedding_category_counts.csv", count_rows)

    all_projections = {}
    for method in methods:
        projected_by_variant = {
            variant: (
                data,
                project_embeddings(data.embeddings, method, projection_seed),
            )
            for variant, data in selected.items()
        }
        export_coordinates(method, projected_by_variant, output_dir)
        for show_text in (True, False):
            plot_projection(method, projected_by_variant, output_dir, show_text)
        all_projections[method] = projected_by_variant
    return all_projections


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        raise ValueError('--methods must not contain duplicates.')
    if args.replot_only:
        output_dir = args.output_dir.resolve()
        for method in args.methods:
            projected = load_exported_coordinates(method, output_dir)
            for show_text in (True, False):
                plot_projection(method, projected, output_dir, show_text)
        print(f'Redrawn from saved coordinates in {output_dir}')
        return
    projections = run_analysis(
        args.results_root,
        args.kg_path,
        args.output_dir,
        seed=args.seed,
        fold=args.fold,
        projection_seed=args.projection_seed,
        methods=args.methods,
    )
    for method, by_variant in projections.items():
        counts = ", ".join(
            f"{VARIANT_LABELS[v]}={len(data.entities)}"
            for v, (data, _coordinates) in by_variant.items()
        )
        print(f"{PROJECTION_LABELS[method]}: {counts}")
    print(f"Figures and source data saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
