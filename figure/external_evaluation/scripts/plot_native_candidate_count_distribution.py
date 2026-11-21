#!/usr/bin/env python
"""Native-candidate count distribution.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_native_candidate_count_distribution.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/native_candidates/.
Scored genes per fixed intervention-metabolite query, shown as an ECDF over nonempty queries; source tables retain zero queries.
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
    CATEGORY_TICK_SIZE, DATA_LINEWIDTH, create_panel, save_panel, style_axis,
)


def build_counts():
    _, _, candidates, _ = native_data()
    test = mech.read_cf(mech.SPECIES_CONFIG["ECO"]["test_path"])
    metabolites = sorted(test.metabolite.unique())
    directions = ("overexpress", "knock out")
    return pd.DataFrame([
        {"method": method, "metabolite": metabolite, "direction": direction,
         "scored_target_count": len(candidates[method].get((metabolite, direction), []))}
        for method in METHODS for metabolite in metabolites for direction in directions
    ])


def build_ecdf(data):
    records = []
    for method in METHODS:
        values = data.loc[data.method.eq(method) & data.scored_target_count.gt(0),
                          "scored_target_count"].to_numpy()
        counts, frequencies = np.unique(values, return_counts=True)
        cumulative = np.cumsum(frequencies)
        for count, frequency, total in zip(counts, frequencies, cumulative):
            records.append({"method": method, "scored_target_count": int(count),
                            "query_frequency": int(frequency),
                            "cumulative_query_count": int(total),
                            "n_scored_queries": len(values),
                            "cumulative_fraction": total / len(values)})
    return pd.DataFrame(records)


def plot_distribution(data, show_text):
    fig, ax = create_panel("square_v1")
    ecdf = build_ecdf(data)
    largest = int(ecdf.scored_target_count.max())
    upper = int(np.ceil(largest / 100) * 100)
    for method in METHODS:
        group = ecdf.loc[ecdf.method.eq(method)].sort_values("scored_target_count")
        x = np.r_[0, group.scored_target_count.to_numpy(), upper]
        y = np.r_[0, group.cumulative_fraction.to_numpy(), 1]
        ax.step(x, y, where="post", color=COLORS[method], linewidth=DATA_LINEWIDTH,
                label=method, alpha=1, clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, upper)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks(np.arange(0, 1.01, .2))
    if show_text:
        ax.set_ylabel("Cumulative fraction", fontsize=13)
        ax.set_xlabel("Scored target genes", fontsize=13)
        ax.legend(loc="lower right", fontsize=CATEGORY_TICK_SIZE,
                  frameon=False, handlelength=1.5)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    return save_panel(fig, OUTPUT / f"native_candidate_count_distribution{'' if show_text else '_no_text'}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = build_counts()
    assert data.scored_target_count.ge(0).all()
    data.to_csv(OUTPUT / "direction_metabolite_candidate_counts.csv", index=False)
    build_ecdf(data).to_csv(OUTPUT / "direction_metabolite_candidate_count_ecdf.csv", index=False)
    summaries = []
    for method in METHODS:
        all_values = data.loc[data.method.eq(method), "scored_target_count"]
        values = all_values[all_values.gt(0)]
        summaries.append({"method": method, "total_query_count": len(all_values),
                          "scored_query_count": len(values),
                          "zero_query_count": int(all_values.eq(0).sum()),
                          "mean": values.mean(), "std": values.std(ddof=1),
                          "min": values.min(), "q1": values.quantile(.25),
                          "median": values.median(), "q3": values.quantile(.75),
                          "max": values.max()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "direction_metabolite_candidate_count_summary.csv", index=False)
    for show in (True, False):
        for path in plot_distribution(data, show):
            print(f"Saved {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
