#!/usr/bin/env python
"""Native-target counts.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_native_target_counts.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/native_candidates/.
Distinct gene-intervention targets per metabolite and percentage matching external positives. Counts include zero-prediction metabolites; percentages exclude them. Bars show metabolite-level mean and sample SD.
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

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from figure.external_evaluation.scripts.native_candidate_data import (
    COLORS, METHODS, OUTPUT, mech, native_data,
)
from figure.figure_style import (
    MM_PER_INCH,
    BAR_DOT_BAR_WIDTH_MM,
    BAR_DOT_ERROR_CAP_WIDTH_MM,
    create_bar_dot_panel,
    save_panel,
    style_axis,
)


def build_data():
    positives, _, candidates, _ = native_data()
    test = mech.read_cf(mech.SPECIES_CONFIG["ECO"]["test_path"])
    metabolites = sorted(test.metabolite.unique())
    records, target_records = [], []
    for method in METHODS:
        for metabolite in metabolites:
            predicted = {(gene, relation) for (met, relation), genes in candidates[method].items()
                         if met == metabolite for gene in genes}
            validated = {(row.gene, row.relation) for row in positives.loc[
                positives.metabolite.eq(metabolite)].itertuples(index=False)}
            correct = predicted & validated
            predicted_genes = {gene for gene, _ in predicted}
            correct_genes = {gene for gene, _ in correct}
            records.append({
                "method": method, "metabolite": metabolite,
                "predicted_target_count": len(predicted),
                "correct_target_count": len(correct),
                "external_positive_target_count": len(validated),
                "predicted_unique_gene_count": len(predicted_genes),
                "correct_unique_gene_count": len(correct_genes),
                "validated_fraction_of_predictions": len(correct) / len(predicted) if predicted else np.nan,
                "external_positive_recall": len(correct) / len(validated) if validated else np.nan,
            })
            for gene, relation in sorted(predicted):
                target_records.append({"method": method, "metabolite": metabolite,
                                       "gene": gene, "relation": relation,
                                       "matches_external_positive": int((gene, relation) in validated)})
    return pd.DataFrame(records), pd.DataFrame(target_records)


def plot_panel(data, metric, show_text):
    fig, ax, centers = create_bar_dot_panel([3])
    positions = centers[0]
    largest = 0.0
    for method, position in zip(METHODS, positions):
        group = data.loc[data.method.eq(method)].sort_values("metabolite")
        values = group[metric].dropna().to_numpy(dtype=float)
        mean, std = float(values.mean()), float(values.std(ddof=1))
        largest = max(largest, mean + std)
        ax.bar(position, mean, width=BAR_DOT_BAR_WIDTH_MM,
               color=COLORS[method], edgecolor="#111111", linewidth=.8, zorder=2)
        ax.errorbar(position, mean, yerr=np.array([[min(mean, std)], [std]]),
                    fmt="none", ecolor="#111111", elinewidth=1,
                    capsize=BAR_DOT_ERROR_CAP_WIDTH_MM * 72 / MM_PER_INCH / 2,
                    capthick=1, zorder=3)
    ax.set_ylim(0, min(100, largest * 1.10) if metric == "correct_target_percent" else largest * 1.10)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="upper"))
    ax.set_xticks(positions)
    if show_text:
        ax.set_xticklabels(METHODS, rotation=35, ha="right", fontsize=9.5)
        label = "Predicted targets\nper metabolite" if metric == "predicted_target_count" else "Correct targets (%)"
        ax.set_ylabel(label, fontsize=13, linespacing=1.05)
        width_mm, height_mm = fig.get_size_inches() * MM_PER_INCH
        ax.yaxis.set_label_coords(5.2 / width_mm, 48 / height_mm,
                                 transform=fig.transFigure)
        ax.yaxis.label.set_verticalalignment("center")
        counts = data.groupby("method")[metric].count()
        assert counts.nunique() == 1
        fig.text(.5, .04, f"n={counts.iloc[0]} metabolites", ha="center", va="center", fontsize=13)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    kind = "predicted" if metric == "predicted_target_count" else "correct"
    stem = OUTPUT / f"native_target_counts_{kind}{'' if show_text else '_no_text'}"
    return save_panel(fig, stem)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, targets = build_data()
    data["correct_target_percent"] = data.validated_fraction_of_predictions * 100
    assert data.correct_target_percent.dropna().between(0, 100).all()
    assert data.correct_target_count.le(data.predicted_target_count).all()
    assert data.correct_target_count.le(data.external_positive_target_count).all()
    data.to_csv(OUTPUT / "metabolite_target_counts.csv", index=False)
    targets.to_csv(OUTPUT / "native_predicted_targets.csv", index=False)
    summary = data.groupby("method", sort=False).agg(
        n_metabolites=("metabolite", "size"),
        n_metabolites_with_predictions=("predicted_target_count", lambda x: int(x.gt(0).sum())),
        mean_predicted_targets=("predicted_target_count", "mean"),
        median_predicted_targets=("predicted_target_count", "median"),
        total_predicted_targets=("predicted_target_count", "sum"),
        std_predicted_targets=("predicted_target_count", "std"),
        mean_correct_targets=("correct_target_count", "mean"),
        std_correct_targets=("correct_target_count", "std"),
        total_correct_targets=("correct_target_count", "sum"),
        n_metabolites_with_defined_percent=("correct_target_percent", "count"),
        mean_correct_target_percent=("correct_target_percent", "mean"),
        std_correct_target_percent=("correct_target_percent", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT / "metabolite_target_count_summary.csv", index=False)
    for metric in ("predicted_target_count", "correct_target_percent"):
        for show in (True, False):
            for path in plot_panel(data, metric, show):
                print(f"Saved {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
