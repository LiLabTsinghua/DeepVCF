#!/usr/bin/env python
"""Hits@30 target distance.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_hit30_graph_distance.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/target_distance/.
Basic-KG shortest-path distances for recovered GEM-covered targets, combined and per intervention; five seeds.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASIC_KG = PROJECT_ROOT / "data" / "processed" / "kg" / "ECO" / "kg_basic.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/target_distance'
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_TEMPLATE = "wide_v1"
HIT_K = 30
TEST_SET = "gem_gene/head_corrupt"

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    DATA_LINEWIDTH,
    EDGE_COLOR,
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    METHOD_COLORS,
    PANEL_TEMPLATES,
    create_panel,
    save_panel,
    style_axis,
)


@dataclass(frozen=True)
class MethodSpec:
    """Rank-table location and model selector for one method."""

    name: str
    result_dir: str
    model: str


METHOD_SPECS = (
    MethodSpec("FSEOF", "fseof", "fseof"),
    MethodSpec("FVSEOF", "fvseof", "fvseof"),
    MethodSpec("iBridge", "ibridge", "ibridge"),
    MethodSpec("DeepVCF", "deepvcf", "fold_1"),
    MethodSpec("DeepVCF+BE", "deepvcf", "ensemble"),
    MethodSpec(
        "DeepVCF+BE+MP", "deepvcf_mp", "ensemble",
    ),
)
METHOD_ORDER = tuple(spec.name for spec in METHOD_SPECS)
DISTANCE_COLORS = {
    "FSEOF": "#F3D9A6",
    "FVSEOF": "#E6AD87",
    "iBridge": "#C98A3D",
    "DeepVCF": METHOD_COLORS["DeepVCF"],
    "DeepVCF+BE": METHOD_COLORS["DeepVCF+BE"],
    "DeepVCF+BE+MP": METHOD_COLORS["DeepVCF+BE+MP"],
}
SOURCE_FIELDS = (
    "method",
    "seed",
    "model",
    "positive_index",
    "gene",
    "relation",
    "metabolite",
    "gene_rank",
    "graph_distance",
    "reachable",
)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=PROJECT_ROOT,
        help="Repository root (default: inferred from this file).",
    )
    parser.add_argument(
        "--basic-kg", type=Path, default=DEFAULT_BASIC_KG,
        help="Basic ECO knowledge graph used for shortest paths.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: outputs/target_distance).",
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


def load_basic_graph(path):
    """Load an undirected adjacency map from the three-column basic KG."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Basic knowledge graph not found: {path}")
    adjacency = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"Expected three tab-separated fields at {path}:{line_number}."
                )
            head, _, tail = fields
            adjacency[head].add(tail)
            adjacency[tail].add(head)
    if not adjacency:
        raise ValueError(f"Basic knowledge graph is empty: {path}")
    return dict(adjacency)


def shortest_paths_from(source, adjacency):
    """Return unweighted undirected distances from one source node."""
    if source not in adjacency:
        return {}
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def read_hit_rows(path, spec, seed, adjacency, distance_cache):
    """Read Hits@30 records and attach basic-KG shortest distances."""
    if not path.is_file():
        raise FileNotFoundError(f"Gene-rank table not found: {path}")
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "model", "seed", "test_set", "positive_index", "gene",
            "relation", "metabolite", "gene_rank",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            if row["model"] != spec.model or row["test_set"] != TEST_SET:
                continue
            if int(row["seed"]) != seed:
                raise ValueError(f"Seed mismatch in {path}.")
            gene_rank = int(row["gene_rank"])
            if gene_rank > HIT_K:
                continue
            metabolite = row["metabolite"]
            if metabolite not in distance_cache:
                distance_cache[metabolite] = shortest_paths_from(
                    metabolite, adjacency,
                )
            distance = distance_cache[metabolite].get(row["gene"])
            records.append({
                "method": spec.name,
                "seed": seed,
                "model": spec.model,
                "positive_index": int(row["positive_index"]),
                "gene": row["gene"],
                "relation": row["relation"],
                "metabolite": metabolite,
                "gene_rank": gene_rank,
                "graph_distance": distance,
                "reachable": distance is not None,
            })
    if not records:
        raise ValueError(f"No Hits@{HIT_K} records for {spec.name}, seed {seed}.")
    return records


def load_distance_data(repo_root, basic_kg, seeds):
    """Load all hit-level distances from the selected rank tables."""
    adjacency = load_basic_graph(basic_kg)
    distance_cache = {}
    records = []
    for spec in METHOD_SPECS:
        for seed in seeds:
            path = (
                repo_root / "results" / "external_evaluation"
                / spec.result_dir / f"seed_{seed}"
                / "gene_corruption_ranks.tsv"
            )
            records.extend(read_hit_rows(
                path, spec, seed, adjacency, distance_cache,
            ))
    source = pd.DataFrame.from_records(records, columns=SOURCE_FIELDS)
    return source


def count_distances(source, seeds, relation=None, distance_levels=None):
    """Count hit records at every finite distance for each method and seed."""
    if relation is not None:
        source = source.loc[source["relation"].eq(relation)].copy()
    reachable = source.loc[source["reachable"]].copy()
    reachable["graph_distance"] = pd.to_numeric(
        reachable["graph_distance"], errors="raise", downcast="integer",
    )
    if distance_levels is None:
        distance_levels = tuple(sorted(reachable["graph_distance"].unique()))
    else:
        distance_levels = tuple(distance_levels)
    if not distance_levels or distance_levels[0] < 1:
        raise ValueError("Finite graph distances must be positive integers.")
    counts = (
        reachable.groupby(
            ["method", "seed", "graph_distance"], observed=True,
        )
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [METHOD_ORDER, seeds, distance_levels],
                names=["method", "seed", "graph_distance"],
            ),
            fill_value=0,
        )
        .rename("count")
        .reset_index()
    )
    expected = len(METHOD_ORDER) * len(seeds) * len(distance_levels)
    if len(counts) != expected:
        raise ValueError(f"Expected {expected} distance-count rows, found {len(counts)}.")
    observed_totals = counts.groupby(["method", "seed"])["count"].sum()
    expected_totals = reachable.groupby(["method", "seed"]).size()
    if not observed_totals.equals(expected_totals.reindex(observed_totals.index)):
        raise ValueError("Distance counts do not sum to the reachable hit totals.")
    unreachable = (
        source.loc[~source["reachable"]]
        .groupby(["method", "seed"], observed=True)
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [METHOD_ORDER, seeds], names=["method", "seed"],
            ),
            fill_value=0,
        )
        .rename("count")
        .reset_index()
    )
    return counts, unreachable


def summarize_counts(counts):
    """Calculate mean count and sample SD across experiment seeds."""
    return (
        counts.groupby(["method", "graph_distance"], observed=True)["count"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )


def draw_panel(ax, count_summary, ylabel, ymax, show_text=True):
    """Draw per-distance hit counts as means ± seed SD."""
    distances = np.sort(count_summary["graph_distance"].unique())
    for method in METHOD_ORDER:
        subset = (
            count_summary.loc[count_summary["method"].eq(method)]
            .set_index("graph_distance")
            .loc[distances]
        )
        ax.errorbar(
            distances,
            subset["mean"].to_numpy(dtype=float),
            yerr=subset["std"].to_numpy(dtype=float),
            color=DISTANCE_COLORS[method],
            marker="o",
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE,
            label=method,
            alpha=1.0,
            zorder=3,
        )

    style_axis(ax)
    ax.set_xticks(distances)
    if show_text:
        ax.set_xlabel("Shortest-path distance to metabolite")
        ax.set_ylabel(ylabel)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlim(distances.min() - 0.5, distances.max() + 0.5)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1, 10))
    ax.grid(False)


def plot_panel(
    count_summary, output_stem, template, ylabel, ymax, show_text=True,
):
    """Render and save one standalone distance panel."""
    fig, ax = create_panel(template)
    draw_panel(
        ax, count_summary, ylabel=ylabel, ymax=ymax, show_text=show_text,
    )
    return save_panel(fig, output_stem)


def run(repo_root, basic_kg, output_dir, seeds, template):
    """Calculate graph distances, export tables, and render the panel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_distance_data(repo_root, basic_kg, seeds)
    counts, unreachable = count_distances(source, seeds)
    count_summary = summarize_counts(counts)
    source.to_csv(
        output_dir / "hits30_target_metabolite_graph_distance_source_data.csv",
        index=False,
    )
    counts.to_csv(
        output_dir / "hits30_target_metabolite_graph_distance_seed_counts.csv",
        index=False,
    )
    count_summary.to_csv(
        output_dir / "hits30_target_metabolite_graph_distance_count_summary.csv",
        index=False,
    )
    unreachable.to_csv(
        output_dir / "hits30_target_metabolite_graph_distance_unreachable_counts.csv",
        index=False,
    )
    stem = output_dir / "hits30_target_metabolite_graph_distance"
    outputs = []
    outputs.extend(plot_panel(
        count_summary, stem, template,
        ylabel="Number of targets in top 30", ymax=60, show_text=True,
    ))
    outputs.extend(plot_panel(
        count_summary,
        Path(f"{stem}_no_text"),
        template,
        ylabel="Number of targets in top 30",
        ymax=60,
        show_text=False,
    ))

    distance_levels = tuple(sorted(counts["graph_distance"].unique()))
    relation_specs = (
        ("knock out", "knockout", "Number of KO targets in top 30"),
        ("overexpress", "overexpression", "Number of OE targets in top 30"),
    )
    relation_results = []
    for relation, suffix, ylabel in relation_specs:
        relation_counts, relation_unreachable = count_distances(
            source, seeds, relation=relation, distance_levels=distance_levels,
        )
        relation_summary = summarize_counts(relation_counts)
        relation_results.append((relation, suffix, ylabel, relation_counts,
                                 relation_unreachable, relation_summary))

    highest_bound = max(
        float((summary["mean"] + summary["std"].fillna(0)).max())
        for *_, summary in relation_results
    )
    relation_ymax = max(10, int(np.ceil(highest_bound / 10.0) * 10))

    for relation, suffix, ylabel, relation_counts, relation_unreachable, relation_summary in relation_results:
        relation_counts.to_csv(
            output_dir / f"hits30_target_metabolite_graph_distance_{suffix}_seed_counts.csv",
            index=False,
        )
        relation_summary.to_csv(
            output_dir / f"hits30_target_metabolite_graph_distance_{suffix}_count_summary.csv",
            index=False,
        )
        relation_unreachable.to_csv(
            output_dir / f"hits30_target_metabolite_graph_distance_{suffix}_unreachable_counts.csv",
            index=False,
        )
        relation_stem = output_dir / f"hits30_target_metabolite_graph_distance_{suffix}"
        outputs.extend(plot_panel(
            relation_summary, relation_stem, template,
            ylabel=ylabel, ymax=relation_ymax, show_text=True,
        ))
        outputs.extend(plot_panel(
            relation_summary, Path(f"{relation_stem}_no_text"), template,
            ylabel=ylabel, ymax=relation_ymax, show_text=False,
        ))
        relation_hits = int(source["relation"].eq(relation).sum())
        print(f"{relation}: {relation_hits} Hits@{HIT_K} records")
    print(f"Hits@{HIT_K} records: {len(source)}")
    print(f"Unreachable records: {(~source['reachable']).sum()}")
    print(f"Template: {template}")
    for output in outputs:
        print(f"Saved {output}")


def main(argv=None):
    """Generate the basic-KG Hits@30 target-distance panel."""
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be non-empty and contain no duplicates.")
    run(args.repo_root, args.basic_kg, args.output_dir, seeds, args.template)


if __name__ == "__main__":
    main()
