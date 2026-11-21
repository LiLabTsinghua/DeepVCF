#!/usr/bin/env python
"""Zero-to-50-shot transfer trajectories.

Run from the repository root in deepvcf:
    python figure/cross_species_transfer/scripts/plot_few_shot.py

Outputs: figure/cross_species_transfer/outputs/few_shot/.
Inputs: results/cross_species_transfer/{zero_shot,few_shot}/. SCE and CGL, each with GEM-covered and GEM-uncovered groups, at 0/10/20/30/40/50 support examples. Uses five-fold ensemble metrics from five seeds. Separate AUPR, AUROC, and Hits@30 panels show every seed value. Linear regression is fitted to the six shot-level means; the band is the 95% confidence interval for the fitted mean. Fit-statistics CSVs report slope, intercept, p value, and R-squared.
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
from scipy.stats import linregress, t


# Mandatory editable-text SVG settings before any figure is created.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


# =============================================================================
# DeepVCF Cross-Species Few-Shot Figures
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    create_panel,
    save_panel,
    style_axis,
)
from figure.cross_species_transfer.scripts.transfer_common import (
    DEFAULT_SEEDS, read_json,
    validate_metric as _validate_metric, validate_seeds,
)

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "cross_species_transfer"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/few_shot'
SHOT_ORDER = (0, 10, 20, 30, 40, 50)
PAIR_ORDER = ("ECO_SCE", "ECO_CGL")
PAIR_TO_SPECIES = {
    "ECO_SCE": "SCE",
    "ECO_CGL": "CGL",
}
SPECIES_LABELS = {
    "SCE": "S. cerevisiae",
    "CGL": "C. glutamicum",
}
GENE_GROUPS = ("gem_gene", "non_gem_gene")
GENE_GROUP_LABELS = {
    "gem_gene": "GEM genes",
    "non_gem_gene": "Non-GEM genes",
}
HIT_METRICS = ("gene_hit@10", "gene_hit@20", "gene_hit@30")
HIT_LABELS = {
    "gene_hit@10": "Hits@10",
    "gene_hit@20": "Hits@20",
    "gene_hit@30": "Hits@30",
}
DEEPVCF_BE_COLOR = "#7A36E0"


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence used by tests. If ``None``, arguments
            are read from ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Seeds to include (default: 1 2 3 4 5).",
    )
    return parser.parse_args(argv)


def _metrics_path(results_root, pair, shot, seed):
    """Return the zero-shot or few-shot metrics path for one run."""
    if shot == 0:
        return (
            results_root / "zero_shot" / pair / f"seed_{seed}"
            / "metrics.json"
        )
    return (
        results_root / "few_shot" / pair / f"shot_{shot}" / f"seed_{seed}"
        / "metrics.json"
    )


def load_source_data(results_root, seeds=DEFAULT_SEEDS):
    """Load zero-shot and few-shot ensemble metrics into one long table.

    Args:
        results_root (str or Path): Root containing transfer outputs.
        seeds (sequence[int]): Expected experiment seeds.

    Returns:
        pandas.DataFrame: One row per species, shot, gene group, and seed.

    Raises:
        FileNotFoundError: If a required metrics file is absent.
        ValueError: If metadata, test-set structure, or metric values are
            inconsistent.
    """
    results_root = Path(results_root)
    records = []
    for pair in PAIR_ORDER:
        target_species = PAIR_TO_SPECIES[pair]
        for shot in SHOT_ORDER:
            for seed in seeds:
                path = _metrics_path(results_root, pair, shot, seed)
                payload = read_json(path)
                if payload.get("pair") != pair:
                    raise ValueError(f"Pair mismatch in {path}: {payload.get('pair')!r}")
                if int(payload.get("seed", -1)) != seed:
                    raise ValueError(f"Seed mismatch in {path}: {payload.get('seed')!r}")
                if payload.get("target_species") != target_species:
                    raise ValueError(
                        f"Target-species mismatch in {path}: "
                        f"{payload.get('target_species')!r}"
                    )
                if shot and int(payload.get("shot", -1)) != shot:
                    raise ValueError(f"Shot mismatch in {path}: {payload.get('shot')!r}")
                test_sets = payload.get("test_sets")
                if not isinstance(test_sets, dict):
                    raise ValueError(f"Missing test_sets in {path}")
                for gene_group in GENE_GROUPS:
                    test_key = f"{gene_group}/head_corrupt"
                    if test_key not in test_sets:
                        raise ValueError(f"Missing {test_key} in {path}")
                    ensemble = test_sets[test_key].get("ensemble")
                    if not isinstance(ensemble, dict):
                        raise ValueError(f"Missing ensemble metrics in {path}/{test_key}")
                    binary = ensemble.get("binary")
                    ranking = ensemble.get("target_gene_ranking")
                    if not isinstance(binary, dict) or not isinstance(ranking, dict):
                        raise ValueError(f"Incomplete gene metrics in {path}/{test_key}")
                    binary_key = "gem" if gene_group == "gem_gene" else "non_gem"
                    group_binary = binary.get(binary_key)
                    if not isinstance(group_binary, dict):
                        raise ValueError(f"Missing {binary_key} metrics in {path}/{test_key}")
                    row = {
                        "pair": pair,
                        "target_species": target_species,
                        "target_species_label": SPECIES_LABELS[target_species],
                        "gene_group": gene_group,
                        "gene_group_label": GENE_GROUP_LABELS[gene_group],
                        "shot": shot,
                        "shot_label": "Zero" if shot == 0 else str(shot),
                        "seed": seed,
                        "AUPR": _validate_metric(
                            group_binary["aupr"], f"{path}/{test_key}/aupr"
                        ),
                        "AUROC": _validate_metric(
                            group_binary["auroc"], f"{path}/{test_key}/auroc"
                        ),
                    }
                    for metric in HIT_METRICS:
                        row[HIT_LABELS[metric]] = _validate_metric(
                            ranking[metric], f"{path}/{test_key}/{metric}"
                        )
                    records.append(row)
    data = pd.DataFrame.from_records(records)
    if data.empty:
        raise ValueError(f"No transfer metrics found below {results_root}")
    return data


def summarize(data):
    """Calculate mean, sample standard deviation, and seed count."""
    value_columns = ["AUPR", "AUROC", *HIT_LABELS.values()]
    summary = (
        data.groupby(
            ["pair", "target_species", "target_species_label", "gene_group",
             "gene_group_label", "shot", "shot_label"],
            observed=True,
        )[value_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def _group_rows(data, species, gene_group, shot):
    """Return rows for one species, group, and shot in seed order."""
    return data.loc[
        (data["target_species"] == species)
        & (data["gene_group"] == gene_group)
        & (data["shot"] == shot)
    ].sort_values("seed")


def write_source_data(data, summary, output_dir):
    """Write combined and gene-group-specific source tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        output_dir / "few_shot_gene_corruption_source_data.csv", index=False,
    )
    summary.to_csv(
        output_dir / "few_shot_gene_corruption_summary.csv", index=False,
    )
    for gene_group in GENE_GROUPS:
        suffix = "gem_gene" if gene_group == "gem_gene" else "non_gem_gene"
        group_data = data.loc[data["gene_group"] == gene_group]
        group_summary = summary.loc[summary["gene_group"] == gene_group]
        group_data.to_csv(
            output_dir / f"few_shot_gene_corruption_{suffix}_source_data.csv",
            index=False,
        )
        group_summary.to_csv(
            output_dir / f"few_shot_gene_corruption_{suffix}_summary.csv",
            index=False,
        )


def plot_metric_vs_shot(
    data, output_dir, gene_group, species, metric, stem_metric,
    ylabel, show_text=True,
):
    """Plot one species' bounded-metric trajectory in a square_v1 panel."""
    fig, ax = create_panel("square_v1")
    means = []
    all_values = []
    for shot in SHOT_ORDER:
        rows = _group_rows(data, species, gene_group, shot)
        values = rows[metric].to_numpy(dtype=float)
        all_values.extend(values.tolist())
        means.append(float(values.mean()))
        jitter = np.linspace(-1.15, 1.15, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(
            np.full(values.size, shot, dtype=float) + jitter, values,
            s=20, color="#161616", edgecolors="#161616", linewidths=0,
            alpha=1.0, zorder=4,
        )
    x = np.asarray(SHOT_ORDER, dtype=float)
    y = np.asarray(means, dtype=float)
    fit = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 200)
    fitted = fit.intercept + fit.slope * xx
    residuals = y - (fit.intercept + fit.slope * x)
    residual_se = np.sqrt(np.sum(residuals ** 2) / (len(x) - 2))
    mean_se = residual_se * np.sqrt(
        1 / len(x) + (xx - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    )
    ci = t.ppf(0.975, df=len(x) - 2) * mean_se
    ax.fill_between(
        xx, fitted - ci, fitted + ci,
        color=DEEPVCF_BE_COLOR, alpha=0.22, linewidth=0, zorder=1,
    )
    ax.plot(xx, fitted, color=DEEPVCF_BE_COLOR,
            linewidth=2.0, linestyle="-", zorder=2)
    style_axis(ax)
    ax.set_xlim(-3, 53)
    visible_low = min(min(all_values), float(np.min(fitted - ci)))
    visible_high = max(max(all_values), float(np.max(fitted + ci)))
    padding = max(0.025, (visible_high - visible_low) * 0.16)
    y_min = max(0.0, np.floor((visible_low - padding) * 10) / 10)
    y_max = min(1.0, np.ceil((visible_high + padding) * 10) / 10)
    # Keep boundary ticks and seed markers clear of the axes frame without
    # changing the clean 0.1-spaced tick sequence.
    display_pad = 0.025
    ax.set_ylim(max(0.0, y_min - display_pad), min(1.0, y_max + display_pad))
    ax.set_xticks(SHOT_ORDER)
    ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.1))
    if show_text:
        ax.set_xlabel("Target-species training examples")
        ax.set_ylabel(ylabel)
        ax.text(
            0.97, 0.06, f"slope={fit.slope:.4f}\np={fit.pvalue:.3g}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#161616",
        )
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(False)
    stats = pd.DataFrame([{
        "gene_group": gene_group, "species": species,
        "slope_per_shot": fit.slope, "intercept": fit.intercept,
        "p_value": fit.pvalue, "r_squared": fit.rvalue ** 2,
        "n_shot_points": len(x),
    }])
    stats_path = output_dir / f"few_shot_{stem_metric}_vs_shot_{gene_group}_{species.lower()}_fit_stats.csv"
    stats.to_csv(stats_path, index=False)
    stem = output_dir / f"few_shot_{stem_metric}_vs_shot_{gene_group}_{species.lower()}"
    if not show_text:
        stem = output_dir / f"few_shot_{stem_metric}_vs_shot_{gene_group}_{species.lower()}_no_text"
    return save_panel(fig, stem), stats_path


def main(argv=None):
    """Load transfer results, export source data, and render figures."""
    args = parse_args(argv)
    validate_seeds(args.seeds)
    data = load_source_data(args.results_root, tuple(args.seeds))
    summary = summarize(data)
    write_source_data(data, summary, args.output_dir)
    for gene_group in GENE_GROUPS:
        for species in ("SCE", "CGL"):
            for show_text in (True, False):
                for metric, stem_metric, ylabel in (
                    ('AUPR', 'aupr', 'Gene-corruption AUPR'),
                    ('AUROC', 'auroc', 'Gene-corruption AUROC'),
                    ('Hits@30', 'hits30', 'Filtered target-gene Hits@30'),
                ):
                    outputs, stats_path = plot_metric_vs_shot(
                        data, args.output_dir, gene_group, species,
                        metric, stem_metric, ylabel, show_text,
                    )
                    print(f"Saved {stats_path}")
                    for output in outputs:
                        print(f"Saved {output}")
    print(f"Few-shot figures and source data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
