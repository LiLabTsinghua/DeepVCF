#!/usr/bin/env python
"""Zero-shot transfer benchmark.

Run from the repository root in deepvcf:
    python figure/cross_species_transfer/scripts/plot_zero_shot.py

Outputs: figure/cross_species_transfer/outputs/zero_shot/.
Inputs: results/cross_species_transfer/zero_shot/<pair>/seed_<seed>/metrics.json and per-method mechanistic metrics beside them. GEM-covered: SCE, CGL, BSU, PPU, comparing FSEOF/FVSEOF/iBridge with DeepVCF and +BE. GEM-uncovered: SCE, CGL, BSU; PPU is omitted because only two positives are available. DeepVCF uses fixed FoldUnit 2; +BE uses the five-fold ensemble. Five seeds, mean and sample SD, with individual seed points.
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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# =============================================================================
# Cross-Species Zero-Shot Benchmark
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    BAR_DOT_BAR_WIDTH_MM,
    BAR_DOT_ERROR_CAP_WIDTH_MM,
    BAR_DOT_ERROR_LINEWIDTH,
    BAR_DOT_POINT_DIAMETER_MM,
    BAR_EDGE_WIDTH,
    CATEGORY_TICK_SIZE,
    MM_PER_INCH,
    METHOD_COLORS as SHARED_METHOD_COLORS,
    create_bar_dot_panel,
    save_panel,
)

DEFAULT_DEEPVCF_ROOT = PROJECT_ROOT / "results" / "cross_species_transfer"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "results" / "cross_species_transfer" / "zero_shot"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/zero_shot'

SPECIES_ORDER = ("SCE", "CGL", "BSU", "PPU")
NON_GEM_SPECIES_ORDER = ("SCE", "CGL", "BSU")
SPECIES_TO_PAIR = {
    "SCE": "ECO_SCE",
    "CGL": "ECO_CGL",
    "BSU": "ECO_BSU",
    "PPU": "ECO_PPU",
}
METHOD_ORDER = ("FSEOF", "FVSEOF", "iBridge", "DeepVCF", "DeepVCF+BE")
DEEPVCF_METHODS = ("DeepVCF", "DeepVCF+BE")
SINGLE_MODEL_FOLD = 2
METHOD_KEYS = {
    "FSEOF": "fseof",
    "FVSEOF": "fvseof",
    "iBridge": "ibridge",
}
METHOD_COLORS = {
    "FSEOF": "#F3D9A6",
    "FVSEOF": "#E6AD87",
    "iBridge": "#C98A3D",
    "DeepVCF": SHARED_METHOD_COLORS["DeepVCF"],
    "DeepVCF+BE": SHARED_METHOD_COLORS["DeepVCF+BE"],
}
EDGE_COLOR = "#161616"


from figure.cross_species_transfer.scripts.transfer_common import (
    DEFAULT_SEEDS, HIT_METRIC_KEYS as METRIC_KEYS, read_json,
    validate_metric as validate_value, validate_seeds,
)

def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence. If ``None``, use ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepvcf-root", type=Path, default=DEFAULT_DEEPVCF_ROOT,
    )
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Seeds to include (default: 1 2 3 4 5).",
    )
    return parser.parse_args(argv)


def extract_values(binary, ranking, context):
    """Extract binary and gene-ranking metrics from evaluation blocks."""
    values = {
        "AUPR": validate_value(binary["aupr"], f"{context}/aupr"),
        "AUROC": validate_value(binary["auroc"], f"{context}/auroc"),
    }
    for label, key in METRIC_KEYS.items():
        values[label] = validate_value(ranking[key], f"{context}/{key}")
    return values


def load_deepvcf_values(root, species, seed, gene_group="gem_gene", method="DeepVCF+BE"):
    """Load metrics from fixed FoldUnit 2 or the five-FoldUnit ensemble.

    Args:
        root: Cross-species result root.
        species: Target species abbreviation.
        seed: Experiment seed.
        gene_group: ``gem_gene`` or ``non_gem_gene`` test group.
        method: ``DeepVCF`` (FoldUnit 2) or ``DeepVCF+BE`` (ensemble).

    Returns:
        dict[str, float]: AUPR and filtered Hits@K values.

    Raises:
        ValueError: If metadata or ensemble metrics are inconsistent.
    """
    pair = SPECIES_TO_PAIR[species]
    path = Path(root) / "zero_shot" / pair / f"seed_{seed}" / "metrics.json"
    payload = read_json(path)
    if payload.get("pair") != pair or payload.get("target_species") != species:
        raise ValueError(f"Species metadata mismatch in {path}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Seed metadata mismatch in {path}")
    filter_path = payload.get("target_filter_path")
    if Path(filter_path or "").name != f"{species}_dataset.txt":
        raise ValueError(f"Unexpected target ranking filter in {path}: {filter_path}")
    if gene_group not in {"gem_gene", "non_gem_gene"}:
        raise ValueError(f"Unsupported gene group: {gene_group}")
    test_set = payload["test_sets"][f"{gene_group}/head_corrupt"]
    if method == "DeepVCF":
        folds = [fold for fold in test_set["folds"] if int(fold["fold"]) == SINGLE_MODEL_FOLD]
        if len(folds) != 1:
            raise ValueError(f"Expected exactly one FoldUnit {SINGLE_MODEL_FOLD} in {path}")
        result = folds[0]
    elif method == "DeepVCF+BE":
        result = test_set["ensemble"]
    else:
        raise ValueError(f"Unsupported DeepVCF method: {method}")
    binary_key = "gem" if gene_group == "gem_gene" else "non_gem"
    binary = result["binary"].get(binary_key) or result["binary"]["overall"]
    return extract_values(binary, result["target_gene_ranking"], path)


def load_baseline_values(root, species, method, seed):
    """Load one mechanistic baseline result when available.

    Args:
        root: Mechanistic-baseline result root.
        species: Target species abbreviation.
        method: Display method name.
        seed: Experiment seed.

    Returns:
        dict[str, float] or None: Metrics, or ``None`` when unavailable.

    Raises:
        ValueError: If present result metadata are inconsistent.
    """
    method_key = METHOD_KEYS[method]
    path = (Path(root) / SPECIES_TO_PAIR[species] / f"seed_{seed}"
            / method_key / "metrics.json")
    if not path.is_file():
        return None
    payload = read_json(path)
    if payload.get("species") != species or payload.get("method") != method_key:
        raise ValueError(f"Method metadata mismatch in {path}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Seed metadata mismatch in {path}")
    metrics = payload["metrics"]
    return extract_values(
        metrics["binary"], metrics["target_gene_ranking"], path,
    )


def load_source_data(deepvcf_root, baseline_root, seeds=DEFAULT_SEEDS):
    """Load method-level zero-shot GEM-gene benchmark data.

    Args:
        deepvcf_root: Cross-species DeepVCF result root.
        baseline_root: Mechanistic-baseline result root.
        seeds: Experiment seeds.

    Returns:
        pandas.DataFrame: One row per available species, method, seed, metric.

    Raises:
        ValueError: If an available method has incomplete seeds or metrics.
    """
    records = []
    for species in SPECIES_ORDER:
        for method in METHOD_ORDER:
            available_seeds = 0
            for seed in seeds:
                values = (
                    load_deepvcf_values(deepvcf_root, species, seed, method=method)
                    if method in DEEPVCF_METHODS
                    else load_baseline_values(
                        baseline_root, species, method, seed,
                    )
                )
                if values is None:
                    continue
                available_seeds += 1
                for metric, value in values.items():
                    records.append({
                        "species": species,
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                    })
            if available_seeds not in (0, len(seeds)):
                raise ValueError(
                    f"Incomplete seeds for {species}/{method}: "
                    f"{available_seeds}/{len(seeds)}"
                )
    data = pd.DataFrame.from_records(records)
    if data.empty:
        raise ValueError("No benchmark results were loaded.")
    return data


def summarize(data):
    """Calculate mean, sample SD, and seed count for each comparison."""
    return (
        data.groupby(["species", "method", "metric"], observed=True)["value"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )


def load_non_gem_source_data(deepvcf_root, seeds=DEFAULT_SEEDS):
    """Load GEM-uncovered metrics from both DeepVCF conditions.

    Args:
        deepvcf_root: Cross-species DeepVCF result root.
        seeds: Experiment seeds.

    Returns:
        pandas.DataFrame: One row per species, seed, and metric.
    """
    records = []
    for species in NON_GEM_SPECIES_ORDER:
        for method in DEEPVCF_METHODS:
            for seed in seeds:
                values = load_deepvcf_values(
                    deepvcf_root, species, seed, gene_group="non_gem_gene", method=method,
                )
                for metric, value in values.items():
                    records.append({
                        "species": species,
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                    })
    return pd.DataFrame.from_records(records)


def _metric_values(data, species, method, metric):
    """Return seed-ordered values for one species, method, and metric."""
    values = data.loc[
        data["species"].eq(species)
        & data["method"].eq(method)
        & data["metric"].eq(metric)
    ].sort_values("seed")["value"].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"Missing values for {species}/{method}/{metric}.")
    return values


def _metric_label(metric):
    return "Gene-corruption Hits@30" if metric == "Hits@30" else f"Gene-corruption {metric}"


def _metric_stem(metric):
    return metric.lower().replace("@", "")


def _metric_scale(data, metric):
    """Use 0.2-spaced ticks inside the valid probability domain."""
    subset = data.loc[data["metric"].eq(metric)]
    if subset.empty:
        raise ValueError(f"No {metric} values to plot")
    grouped = subset.groupby(["species", "method"])["value"].agg(["mean", "std"])
    low = float((grouped["mean"] - grouped["std"].fillna(0)).min())
    high = float((grouped["mean"] + grouped["std"].fillna(0)).max())
    if low < 0 or high > 1:
        raise ValueError(f"{metric} mean ± SD leaves the [0, 1] domain")
    upper = min(1.0, max(0.2, np.ceil((max(high, subset["value"].max()) + 0.02) / 0.2) * 0.2))
    return float(upper), np.arange(0, upper + 1e-9, 0.2)


def plot_metric(data, output_dir, metric, *, non_gem=False, show_text=True):
    """Draw one standalone grouped bar-and-seed-dot panel."""
    species_order = NON_GEM_SPECIES_ORDER if non_gem else SPECIES_ORDER
    methods = DEEPVCF_METHODS if non_gem else METHOD_ORDER
    fig, ax, centers = create_bar_dot_panel([len(methods)] * len(species_order))
    upper, ticks = _metric_scale(data, metric)
    ax.set_ylim(0, upper)
    ax.set_yticks(ticks)
    dot_area = (BAR_DOT_POINT_DIAMETER_MM * 72 / MM_PER_INCH) ** 2
    capsize = BAR_DOT_ERROR_CAP_WIDTH_MM * 72 / MM_PER_INCH / 2
    for species, positions in zip(species_order, centers):
        for method, position in zip(methods, positions):
            values = _metric_values(data, species, method, metric)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ax.bar(position, mean, width=BAR_DOT_BAR_WIDTH_MM,
                   color=METHOD_COLORS[method], edgecolor=EDGE_COLOR,
                   linewidth=BAR_EDGE_WIDTH, alpha=1.0, zorder=2)
            ax.errorbar(position, mean, yerr=std, fmt="none", ecolor=EDGE_COLOR,
                        elinewidth=BAR_DOT_ERROR_LINEWIDTH, capsize=capsize,
                        capthick=BAR_DOT_ERROR_LINEWIDTH, alpha=1.0, zorder=3)
            offsets = np.linspace(-2.5, 2.5, len(values))
            ax.scatter(position + offsets, values, s=dot_area, color=EDGE_COLOR,
                       linewidths=0, alpha=1.0, zorder=4)
    ax.set_xticks([np.mean(positions) for positions in centers])
    if show_text:
        ax.set_xticklabels(species_order, fontsize=CATEGORY_TICK_SIZE)
        ax.set_ylabel(_metric_label(metric))
        if methods:
            handles = [Patch(facecolor=METHOD_COLORS[method], edgecolor=EDGE_COLOR,
                             linewidth=BAR_EDGE_WIDTH, label=method)
                       for method in methods]
            fig.legend(handles=handles, loc="lower center",
                       bbox_to_anchor=(0.5, 0.06), ncol=len(methods),
                       fontsize=CATEGORY_TICK_SIZE, handlelength=1.0,
                       columnspacing=1.1)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    stem = output_dir / f"zero_shot_{'non_gem' if non_gem else 'gem'}_{_metric_stem(metric)}"
    if not show_text:
        stem = output_dir / f"{stem.name}_no_text"
    return save_panel(fig, stem)


def main(argv=None):
    """Load benchmark results and export figures with source data."""
    args = parse_args(argv)
    validate_seeds(args.seeds)
    data = load_source_data(
        args.deepvcf_root, args.baseline_root, tuple(args.seeds),
    )
    summary = summarize(data)
    non_gem_data = load_non_gem_source_data(
        args.deepvcf_root, tuple(args.seeds),
    )
    non_gem_summary = summarize(non_gem_data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        args.output_dir / "zero_shot_gem_benchmark_source_data.csv", index=False,
    )
    summary.to_csv(
        args.output_dir / "zero_shot_gem_benchmark_summary.csv", index=False,
    )
    non_gem_data.to_csv(
        args.output_dir / "zero_shot_non_gem_source_data.csv", index=False,
    )
    non_gem_summary.to_csv(
        args.output_dir / "zero_shot_non_gem_summary.csv", index=False,
    )
    outputs = []
    for metric in ("AUROC", "AUPR", "Hits@30"):
        for show_text in (True, False):
            outputs.extend(plot_metric(
                data, args.output_dir, metric, show_text=show_text,
            ))
            outputs.extend(plot_metric(
                non_gem_data, args.output_dir, metric,
                non_gem=True, show_text=show_text,
            ))
    print(summary.to_string(index=False))
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
