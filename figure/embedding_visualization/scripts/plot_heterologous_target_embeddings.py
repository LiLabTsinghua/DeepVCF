#!/usr/bin/env python3
"""Endogenous and heterologous target embeddings.

Run from the repository root in deepvcf:
    python figure/embedding_visualization/scripts/plot_heterologous_target_embeddings.py
    python figure/embedding_visualization/scripts/plot_heterologous_target_embeddings.py --replot-only

Outputs: figure/embedding_visualization/outputs/heterologous_targets/.
1 x 2: Basic-extended, Enhanced-extended. Inputs: results/heterologous_target_prediction/<variant>/seed_<seed>/drn_trained_model.pt and the Enhanced ECO KG for endogenous coverage. Categories: endogenous GEM-covered, endogenous GEM-uncovered, and heterologous proteins.
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
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "heterologous_target_prediction"
DEFAULT_REFERENCE_KG = (
    PROJECT_ROOT / "data" / "processed" / "kg" / "ECO" / "kg_enhanced.txt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/heterologous_targets'

VARIANTS = ("basic_extended", "enhanced_extended")
VARIANT_LABELS = {
    "basic_extended": "Basic-extended",
    "enhanced_extended": "Enhanced-extended",
}
CATEGORY_ORDER = (
    "gem_covered_endogenous",
    "gem_uncovered_endogenous",
    "heterologous",
)
CATEGORY_LABELS = {
    "gem_covered_endogenous": "Endogenous GEM-covered genes",
    "gem_uncovered_endogenous": "Endogenous GEM-uncovered genes",
    "heterologous": "Heterologous targets",
}
LEGEND_LABELS = {
    "gem_covered_endogenous": "GEM-covered",
    "gem_uncovered_endogenous": "GEM-uncovered",
    "heterologous": "Heterologous targets",
}
CATEGORY_COLORS = {
    "gem_covered_endogenous": "#D45B72",
    "gem_uncovered_endogenous": "#E59E3F",
    "heterologous": "#773ADA",
}
SCATTER_AREA = 2.0
HETEROLOGOUS_PREFIX = "ECO_Protein_heterologous:"

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import create_small_multiples, save_panel  # noqa: E402


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


from figure.figure_io import write_csv
from figure.embedding_visualization.scripts.embedding_common import (
    PROJECTION_METHODS, PROJECTION_LABELS,
    l2_normalize, project_embeddings, load_coordinate_groups,
    load_kg_categories,
)


@dataclass(frozen=True)
class EmbeddingData:
    """Selected embeddings and their target categories."""

    variant: str
    entities: np.ndarray
    categories: np.ndarray
    embeddings: np.ndarray


def parse_args(argv=None):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reference-kg", type=Path, default=DEFAULT_REFERENCE_KG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument('--methods', nargs='+', choices=PROJECTION_METHODS,
                        default=list(PROJECTION_METHODS))
    parser.add_argument('--replot-only', action='store_true',
                        help='Redraw from saved coordinates without loading checkpoints.')
    return parser.parse_args(argv)


def load_endogenous_categories(kg_path):
    """Define endogenous protein categories from the Enhanced ECO KG."""
    categories = load_kg_categories(kg_path, 'ECO')
    covered, uncovered = categories['gem_covered_gene'], categories['gem_uncovered_gene']
    if not covered or not uncovered:
        raise ValueError('Endogenous GEM coverage categories cannot be empty.')
    return covered, uncovered


def load_embeddings(checkpoint_path, variant, fold, covered, uncovered):
    """Load one fold and classify all endogenous and heterologous proteins."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    node_to_id = checkpoint["kg_metadata"]["node_to_id"]
    states = [value for key, value in checkpoint["folds_state_dict"].items()
                   if key.endswith(".drn.node_emb")]
    if not 0 <= fold < len(states):
        raise ValueError(f"Invalid fold {fold} for {variant}")
    embeddings = l2_normalize(states[fold].detach().cpu().numpy())

    entities = []
    categories = []
    indices = []
    for entity, index in sorted(node_to_id.items()):
        if entity.startswith(HETEROLOGOUS_PREFIX):
            category = "heterologous"
        elif entity in covered:
            category = "gem_covered_endogenous"
        elif entity in uncovered:
            category = "gem_uncovered_endogenous"
        else:
            continue
        entities.append(entity)
        categories.append(category)
        indices.append(int(index))
    if not entities:
        raise ValueError(f"No target proteins found in {checkpoint_path}")
    return EmbeddingData(
        variant=variant,
        entities=np.asarray(entities, dtype=str),
        categories=np.asarray(categories, dtype=str),
        embeddings=embeddings[np.asarray(indices, dtype=int)],
    )


def export_coordinates(method, projected_by_variant, output_dir):
    """Export every point drawn in one projection figure."""
    rows = []
    for variant in VARIANTS:
        data, coordinates = projected_by_variant[variant]
        for entity, category, coordinate in zip(
            data.entities, data.categories, coordinates,
        ):
            rows.append({
                "projection": PROJECTION_LABELS[method],
                "variant": VARIANT_LABELS[variant],
                "entity": entity,
                "category": CATEGORY_LABELS[category],
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
            })
    write_csv(output_dir / f"heterologous_target_{method}_coordinates.csv", rows)


def load_exported_coordinates(method, output_dir):
    """Load the exact coordinates exported for this comparison."""
    groups = load_coordinate_groups(
        output_dir / f"heterologous_target_{method}_coordinates.csv", 'variant', VARIANT_LABELS,
        CATEGORY_LABELS, PROJECTION_LABELS[method],
    )
    return {variant: (EmbeddingData(
        variant=variant, entities=entities, categories=categories,
        embeddings=np.empty((len(entities), 0), dtype=float),
    ), coordinates) for variant, (entities, categories, coordinates) in groups.items()}


def plot_projection(method, projected_by_variant, output_dir, show_text):
    """Draw Basic-extended and Enhanced-extended as a 1 x 2 figure."""
    fig, axes = create_small_multiples(
        "small_multiples_v1", nrows=1, ncols=2,
        sharex=False, sharey=False,
        panel_width_mm=42.0, panel_height_mm=42.0,
    )
    if show_text:
        canvas_height_mm, bottom_mm, top_mm = 58.0, 4.0, 12.0
    else:
        canvas_height_mm, bottom_mm, top_mm = 48.0, 3.0, 3.0
    fig.set_figheight(canvas_height_mm / 25.4)
    fig.subplots_adjust(
        bottom=bottom_mm / canvas_height_mm,
        top=1.0 - top_mm / canvas_height_mm,
    )

    for column, variant in enumerate(VARIANTS):
        ax = axes[0, column]
        data, coordinates = projected_by_variant[variant]
        for category in CATEGORY_ORDER[::-1]:
            mask = data.categories == category
            if np.any(mask):
                ax.scatter(
                    coordinates[mask, 0], coordinates[mask, 1],
                    s=SCATTER_AREA, marker="o",
                    color=CATEGORY_COLORS[category],
                    edgecolors="none", linewidths=0,
                    alpha=1.0, rasterized=False,
                )
        ax.margins(0.035)
        ax.set_axis_off()
        if show_text:
            ax.set_title(VARIANT_LABELS[variant], fontsize=10, pad=4)

    stem = output_dir / f"heterologous_target_embedding_{method}"
    if show_text:
        handles = [
            Line2D(
                [], [], linestyle="none", marker="o", markersize=4.2,
                markerfacecolor=CATEGORY_COLORS[category],
                markeredgecolor="none", label=LEGEND_LABELS[category],
            )
            for category in CATEGORY_ORDER
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.985),
            ncol=3, fontsize=7.2, handletextpad=0.3,
            columnspacing=0.7, borderaxespad=0,
        )
    else:
        stem = Path(f"{stem}_no_text")
    return save_panel(fig, stem)


def run(results_root, reference_kg, output_dir, seed, fold, projection_seed,
        methods=PROJECTION_METHODS):
    """Load target embeddings and generate all three projection figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    covered, uncovered = load_endogenous_categories(reference_kg)
    selected = {
        variant: load_embeddings(
            results_root / variant / f"seed_{seed}" / "drn_trained_model.pt",
            variant, fold, covered, uncovered,
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
    write_csv(output_dir / "heterologous_target_category_counts.csv", count_rows)

    for method in methods:
        projected = {
            variant: (
                data,
                project_embeddings(data.embeddings, method, projection_seed),
            )
            for variant, data in selected.items()
        }
        export_coordinates(method, projected, output_dir)
        plot_projection(method, projected, output_dir, show_text=True)
        plot_projection(method, projected, output_dir, show_text=False)
        counts = ", ".join(
            f"{VARIANT_LABELS[v]}={len(data.entities)}"
            for v, (data, _) in projected.items()
        )
        print(f"{PROJECTION_LABELS[method]}: {counts}")


def main(argv=None):
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
    run(
        args.results_root.resolve(), args.reference_kg.resolve(),
        args.output_dir.resolve(), args.seed, args.fold,
        args.projection_seed, args.methods,
    )
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
