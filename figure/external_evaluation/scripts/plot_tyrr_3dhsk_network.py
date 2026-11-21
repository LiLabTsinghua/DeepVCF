#!/usr/bin/env python
"""tyrR network.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_tyrr_3dhsk_network.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/tyrr/.
Two-hop and three-hop KG paths for the tyrR case, exported separately and as a combined panel. Inputs: data/processed/kg/ECO/kg_enhanced.txt and data/raw/ECO/eco_gene_info.txt.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KG = PROJECT_ROOT / "data/processed/kg/ECO/kg_enhanced.txt"
DEFAULT_GENE_INFO = PROJECT_ROOT / "data/raw/ECO/eco_gene_info.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/tyrr'
TEMPLATE = "narrow_v1"
HOP_CUTOFF = 3

SOURCE = "ECO_Protein:P07604"
SOURCE_LABEL = "tyrR"
TARGETS = {
    "ECO_Protein:P00887": "aroH",
    "ECO_Protein:P00888": "aroF",
    "ECO_Protein:P23173": "tnaB",
    "ECO_Protein:P0A853": "tnaA",
    "ECO_Protein:P0AAD2": "mtr",
    "ECO_Protein:P0A9T0": "serA",
}

SOURCE_COLOR = "#E59E3F"
TARGET_COLOR = "#D45B72"
INTERMEDIATE_COLOR = "#9FA3AE"
INTERMEDIATE_MARKER = "o"
EDGE_COLOR = "#000000"

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import PNG_DPI, create_panel  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--gene-info", type=Path, default=DEFAULT_GENE_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def load_gene_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=("uniprot", "gene_name", "locus", "description"),
        dtype=str,
    )
    return dict(zip(frame["uniprot"], frame["gene_name"]))


def load_kg(path: Path) -> nx.Graph:
    """Load an undirected KG while retaining all relations per node pair."""
    graph = nx.Graph()
    relations = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed KG row at {path}:{line_number}")
            head, relation, tail = fields
            graph.add_edge(head, tail)
            relations[frozenset((head, tail))].add(relation)
    nx.set_edge_attributes(
        graph,
        {
            tuple(edge): {"relations": tuple(sorted(values))}
            for edge, values in relations.items()
        },
    )
    return graph


def enumerate_paths(graph: nx.Graph):
    """Enumerate all simple source-to-target paths of at most three hops."""
    paths = []
    for target in TARGETS:
        target_paths = nx.all_simple_paths(
            graph, SOURCE, target, cutoff=HOP_CUTOFF,
        )
        paths.extend((target, tuple(path)) for path in target_paths)
    if not paths:
        raise ValueError("No tyrR-to-target paths found.")
    return paths


def summarize_paths(graph: nx.Graph, paths):
    """Count node/edge support and construct a path-edge-only graph."""
    node_counts = Counter()
    edge_counts = Counter()
    normalized_positions = defaultdict(list)
    target_support = Counter()
    path_graph = nx.Graph()

    for target, path in paths:
        target_support[target] += 1
        hop_count = len(path) - 1
        for index, node in enumerate(path):
            node_counts[node] += 1
            normalized_positions[node].append(index / hop_count)
        for left, right in zip(path, path[1:]):
            edge = frozenset((left, right))
            edge_counts[edge] += 1
            path_graph.add_edge(left, right)

    for left, right in path_graph.edges():
        source_edge = graph.get_edge_data(left, right, default={})
        path_graph[left][right]["relations"] = source_edge.get(
            "relations", tuple()
        )
    return (
        path_graph,
        node_counts,
        edge_counts,
        normalized_positions,
        target_support,
    )


def node_type(node: str) -> str:
    if node.startswith("ECO_Protein:"):
        return "protein"
    if node.startswith("GO:"):
        return "GO"
    if node.startswith("Reaction:"):
        return "reaction"
    if node.startswith("KO:"):
        return "KO"
    return "other"


def node_label(node: str, gene_names: dict[str, str]) -> str:
    if node.startswith("ECO_Protein:"):
        accession = node.removeprefix("ECO_Protein:")
        return gene_names.get(accession, accession)
    return node.split(":", 1)[-1]


def build_layout(
    graph: nx.Graph,
    paths,
    normalized_positions,
    target_support,
):
    """Create the reference centre-and-ring anchored spring layout."""
    present_targets = sorted(
        (target for target in TARGETS if target in graph),
        key=lambda target: (-target_support[target], TARGETS[target]),
    )
    anchors = {SOURCE: (0.0, 0.0)}
    for index, target in enumerate(present_targets):
        angle = np.pi / 2 - 2 * np.pi * index / len(present_targets)
        anchors[target] = (float(np.cos(angle)), float(np.sin(angle)))

    rng = np.random.default_rng(0)
    initial_positions = {
        node: anchors.get(
            node,
            (float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.6, 0.6))),
        )
        for node in graph.nodes()
    }
    return nx.spring_layout(
        graph,
        pos=initial_positions,
        fixed=list(anchors),
        seed=0,
        k=0.35,
        iterations=120,
        weight=None,
    )


def export_source_data(
    output_dir: Path,
    output_prefix: str,
    graph: nx.Graph,
    paths,
    positions,
    node_counts,
    edge_counts,
    gene_names,
):
    node_rows = []
    for node in sorted(graph.nodes()):
        role = (
            "source" if node == SOURCE
            else "target" if node in TARGETS
            else "intermediate"
        )
        node_rows.append({
            "node": node,
            "label": node_label(node, gene_names),
            "role": role,
            "node_type": node_type(node),
            "path_count": node_counts[node],
            "x": positions[node][0],
            "y": positions[node][1],
        })
    pd.DataFrame(node_rows).to_csv(
        output_dir / f"{output_prefix}_nodes.csv", index=False,
    )

    edge_rows = []
    for left, right in sorted(graph.edges()):
        edge = frozenset((left, right))
        edge_rows.append({
            "node_1": left,
            "node_2": right,
            "relations": ";".join(graph[left][right].get("relations", ())),
            "path_count": edge_counts[edge],
        })
    pd.DataFrame(edge_rows).to_csv(
        output_dir / f"{output_prefix}_edges.csv", index=False,
    )

    path_rows = [
        {
            "target": target,
            "target_label": TARGETS[target],
            "hops": len(path) - 1,
            "path": " -> ".join(path),
        }
        for target, path in paths
    ]
    pd.DataFrame(path_rows).to_csv(
        output_dir / f"{output_prefix}_paths.csv", index=False,
    )


def draw_panel(
    graph,
    positions,
    node_counts,
    edge_counts,
    gene_names,
    *,
    show_text: bool,
    max_edge_reference: int,
    max_node_reference: int,
    ax=None,
):
    """Draw a centre-and-ring path-support network on narrow_v1."""
    if ax is None:
        fig, ax = create_panel(TEMPLATE)
    else:
        fig = ax.figure
    ax.set_axis_off()
    ax.set_aspect("equal")
    edges = list(graph.edges())
    widths = [
        0.08 + 0.48 * np.sqrt(
            edge_counts[frozenset(edge)] / max_edge_reference
        )
        for edge in edges
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=edges,
        width=widths,
        edge_color=EDGE_COLOR,
        alpha=1.0,
        ax=ax,
    )

    intermediate_nodes = [
        node for node in graph
        if node != SOURCE and node not in TARGETS
    ]
    sizes = [
        12 + 62 * np.sqrt(node_counts[node] / max_node_reference)
        for node in intermediate_nodes
    ]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=intermediate_nodes,
        node_size=sizes,
        node_color=INTERMEDIATE_COLOR,
        node_shape=INTERMEDIATE_MARKER,
        edgecolors="white",
        linewidths=0.45,
        ax=ax,
    )

    target_nodes = [target for target in TARGETS if target in graph]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=target_nodes,
        node_size=[
            42 + 100 * np.sqrt(node_counts[node] / max_node_reference)
            for node in target_nodes
        ],
        node_color=TARGET_COLOR,
        edgecolors="white",
        linewidths=0.65,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[SOURCE],
        node_size=[155],
        node_color=SOURCE_COLOR,
        edgecolors="white",
        linewidths=0.75,
        ax=ax,
    )

    if show_text:
        highlighted = [(SOURCE, SOURCE_LABEL), *[
            (target, label) for target, label in TARGETS.items()
            if target in graph
        ]]
        center = np.mean(
            np.asarray([positions[node] for node, _ in highlighted]), axis=0,
        )
        for node, label in highlighted:
            x, y = positions[node]
            direction = np.asarray((x, y)) - center
            norm = max(float(np.linalg.norm(direction)), 1e-6)
            direction /= norm
            text_x = x + 0.035 * direction[0]
            text_y = y + 0.045 * direction[1]
            ax.text(
                text_x,
                text_y,
                label,
                ha="left" if direction[0] >= 0 else "right",
                va="bottom" if direction[1] >= 0 else "top",
                fontsize=8.5 if node != SOURCE else 9,
                fontstyle="italic",
                color="#111111",
            )

    coordinates = np.asarray(list(positions.values()), dtype=float)
    x_extent = max(1.40, float(np.max(np.abs(coordinates[:, 0]))) + 0.40)
    y_extent = max(1.25, float(np.max(np.abs(coordinates[:, 1]))) + 0.25)
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(-y_extent, y_extent)
    return fig


def save_panel(fig, stem: Path):
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, facecolor="white")
    plt.close(fig)


def run(kg_path: Path, gene_info_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_names = load_gene_names(gene_info_path)
    full_graph = load_kg(kg_path)
    paths = enumerate_paths(full_graph)
    _, all_node_counts, all_edge_counts, _, _ = summarize_paths(
        full_graph, paths,
    )
    max_node_reference = max(all_node_counts.values())
    max_edge_reference = max(all_edge_counts.values())
    panel_payloads = {}

    for hop_count in (2, 3):
        selected_paths = [
            (target, path) for target, path in paths
            if len(path) - 1 == hop_count
        ]
        (
            path_graph,
            node_counts,
            edge_counts,
            normalized_positions,
            target_support,
        ) = summarize_paths(full_graph, selected_paths)
        positions = build_layout(
            path_graph,
            selected_paths,
            normalized_positions,
            target_support,
        )
        panel_payloads[hop_count] = (
            path_graph, positions, node_counts, edge_counts,
        )
        output_prefix = f"tyrr_3dhsk_{hop_count}hop_network"
        export_source_data(
            output_dir,
            output_prefix,
            path_graph,
            selected_paths,
            positions,
            node_counts,
            edge_counts,
            gene_names,
        )

        stem = output_dir / output_prefix
        for show_text, output_stem in (
            (True, stem),
            (False, Path(f"{stem}_no_text")),
        ):
            save_panel(
                draw_panel(
                    path_graph,
                    positions,
                    node_counts,
                    edge_counts,
                    gene_names,
                    show_text=show_text,
                    max_edge_reference=max_edge_reference,
                    max_node_reference=max_node_reference,
                ),
                output_stem,
            )
        print(
            f"{hop_count}-hop: {len(selected_paths)} paths, "
            f"{path_graph.number_of_nodes()} nodes, "
            f"{path_graph.number_of_edges()} edges"
        )
        for target in TARGETS:
            print(f"  {TARGETS[target]}: {target_support[target]} paths")
        print(f"Saved {stem}.svg/.png and no-text variants")

    combined_stem = output_dir / "tyrr_3dhsk_2hop_3hop_network"
    for show_text, output_stem in (
        (True, combined_stem),
        (False, Path(f"{combined_stem}_no_text")),
    ):
        fig = plt.figure(figsize=(120 / 25.4, 78 / 25.4), facecolor="white")
        axes = (
            fig.add_axes((0.00, 0.00, 0.50, 1.00)),
            fig.add_axes((0.50, 0.00, 0.50, 1.00)),
        )
        for ax, hop_count in zip(axes, (2, 3)):
            path_graph, positions, node_counts, edge_counts = panel_payloads[
                hop_count
            ]
            draw_panel(
                path_graph,
                positions,
                node_counts,
                edge_counts,
                gene_names,
                show_text=show_text,
                max_edge_reference=max_edge_reference,
                max_node_reference=max_node_reference,
                ax=ax,
            )
            if show_text:
                ax.text(
                    0.04, 0.96, f"{hop_count}-hop",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, color="#111111",
                )
        save_panel(fig, output_stem)
    print(f"Saved {combined_stem}.svg/.png and no-text variants")


def main(argv=None):
    args = parse_args(argv)
    run(args.kg.resolve(), args.gene_info.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
