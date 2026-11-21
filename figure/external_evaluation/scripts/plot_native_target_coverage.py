#!/usr/bin/env python
"""Native-target coverage.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_native_target_coverage.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/native_candidates/.
External positive gene-intervention-metabolite triples covered by each baseline scored-candidate set, before rank filtering.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from figure.external_evaluation.scripts.native_candidate_data import (
    COLORS, METHODS, OUTPUT, mech, native_data,
)
from figure.figure_style import (
    BAR_DOT_BAR_WIDTH_MM, BAR_EDGE_WIDTH, CATEGORY_TICK_SIZE,
    EDGE_COLOR, create_bar_dot_panel, save_panel, style_axis,
)


def build_coverage():
    _, _, candidates, _ = native_data()
    frames = []
    for group in ("gem_gene",):
        path = ROOT / "data/processed/cf/evaluation/external_test" / group / "head_corrupt_test.txt"
        data = mech.read_cf(path)
        frames.append(data.loc[data.label.eq(1)].assign(gene_set=group))
    source = pd.concat(frames, ignore_index=True).drop_duplicates(["gene", "relation", "metabolite"])
    for method in METHODS:
        scored = {(gene, direction, metabolite)
                  for (metabolite, direction), genes in candidates[method].items()
                  for gene in genes}
        source[f"covered_by_{method}"] = [
            (row.gene, row.relation, row.metabolite) in scored
            for row in source.itertuples(index=False)
        ]
    rows = [{"method": "GEM-covered", "target_count": len(source),
             "fraction_of_external_positives": 1.0,
             "unique_gene_count": source.gene.nunique()}]
    for method in METHODS:
        covered = source.loc[source[f"covered_by_{method}"]]
        rows.append({"method": method, "target_count": len(covered),
                     "fraction_of_external_positives": len(covered) / len(source),
                     "unique_gene_count": covered.gene.nunique()})
    return source, pd.DataFrame(rows)


def plot_panel(summary, show_text):
    fig, ax, groups = create_bar_dot_panel([1, 3])
    positions = np.array([*groups[0], *groups[1]])
    for row, position in zip(summary.itertuples(index=False), positions):
        ax.bar(position, row.target_count, width=BAR_DOT_BAR_WIDTH_MM,
               color="#B0B0B0" if row.method == "GEM-covered" else COLORS[row.method],
               edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, alpha=1)
        if show_text:
            ax.text(position, row.target_count + 9, str(row.target_count),
                    ha="center", va="bottom", fontsize=13)
    ax.set_ylim(0, 400)
    ax.set_yticks(np.arange(0, 401, 100))
    ax.set_xticks(positions)
    if show_text:
        ax.set_xticklabels(summary.method, fontsize=CATEGORY_TICK_SIZE,
                           rotation=35, ha="right")
        ax.set_ylabel("External positive targets", fontsize=13)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    return save_panel(fig, OUTPUT / f"native_target_coverage{'' if show_text else '_no_text'}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source, summary = build_coverage()
    assert summary.target_count.between(0, len(source)).all()
    source.to_csv(OUTPUT / "external_positive_target_coverage_source.csv", index=False)
    summary.to_csv(OUTPUT / "external_positive_target_coverage_summary.csv", index=False)
    for show in (True, False):
        for path in plot_panel(summary, show):
            print(f"Saved {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
