#!/usr/bin/env python
"""Training-strategy comparison.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_training_strategy_ablation.py

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/training_strategy/.
Task only, MP only, MP followed by task fine-tuning, and joint training; five seeds, mean and sample SD. MP-only input: results/external_evaluation/mechanistic_pretraining_only/ensemble_metrics_by_seed.tsv.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / 'outputs/training_strategy'
)
DEFAULT_PRETRAIN_ONLY = (
    PROJECT_ROOT / "results" / "external_evaluation"
    / "mechanistic_pretraining_only" / "ensemble_metrics_by_seed.tsv"
)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_TEMPLATE = "square_v1"
SEED_SCATTER_MAX_SIZE = 22

TEST_KEYS = {
    "set1": "gem_gene/head_corrupt",
    "set2": "non_gem_gene/head_corrupt",
}
METRIC_SPECS = {
    "AUPR": {
        "field": "aupr",
        "group": "overall",
        "label": "Gene-corruption AUPR",
        "stem": "aupr",
        "tick_step": 0.2,
    },
    "AUROC": {
        "field": "auroc",
        "group": "overall",
        "label": "Gene-corruption AUROC",
        "stem": "auroc",
        "tick_step": 0.2,
    },
    "Hits@30": {
        "field": "gene_hit@30",
        "group": "target_gene",
        "label": "Filtered target-gene Hits@30",
        "stem": "hits30",
        "tick_step": 0.1,
    },
}
METHOD_SPECS = (
    ("Task only", "deepvcf"),
    ("MP only", None),
    ("MP → Task FT", "deepvcf_mp"),
    ("Joint MP + Task", "deepvcf_joint_training"),
)
METHOD_ORDER = tuple(name for name, _ in METHOD_SPECS)
METHOD_COLORS = {
    "Task only": "#773ADA",
    "MP only": "#E5A04B",
    "MP → Task FT": "#451594",
    "Joint MP + Task": "#3F9C8C",
}

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    BAR_EDGE_WIDTH,
    BAR_WIDTH,
    CATEGORY_TICK_SIZE,
    EDGE_COLOR,
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    PANEL_TEMPLATES,
    create_panel,
    save_panel,
    scatter_size_for_bar,
    style_axis,
)

from figure.external_evaluation.scripts.evaluation_data import read_json


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pretrain-only", type=Path, default=DEFAULT_PRETRAIN_ONLY,
        help="Seed-level TSV produced by checkpoint re-evaluation.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--template", choices=tuple(PANEL_TEMPLATES), default=DEFAULT_TEMPLATE,
    )
    return parser.parse_args(argv)


def metric_from_ensemble(ensemble, metric):
    """Extract one metric from a DeepVCF ensemble result block."""
    spec = METRIC_SPECS[metric]
    if spec["group"] == "overall":
        return float(ensemble["binary"]["overall"][spec["field"]])
    return float(ensemble["target_gene_ranking"][spec["field"]])


def load_trained_models(repo_root, seeds):
    """Load ensemble results for downstream-trained strategies."""
    rows = []
    for method, result_dir in METHOD_SPECS:
        if result_dir is None:
            continue
        for seed in seeds:
            path = (
                repo_root / "results" / "external_evaluation"
                / result_dir / f"seed_{seed}" / "metrics.json"
            )
            payload = read_json(path)
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch in {path}")
            for gene_set, test_key in TEST_KEYS.items():
                ensemble = payload["test_sets"][test_key]["ensemble"]
                for metric in METRIC_SPECS:
                    rows.append({
                        "gene_set": gene_set,
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "value": metric_from_ensemble(ensemble, metric),
                    })
    return rows


def load_pretrain_only(path, seeds):
    """Load MP-only ensemble metrics exported by the re-evaluation notebook."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing MP-only result table: {path}")
    table = pd.read_csv(path, sep="\t")
    rows = []
    for gene_set, test_key in TEST_KEYS.items():
        for metric, spec in METRIC_SPECS.items():
            selected = table.loc[
                table["mode"].eq("pretrain_only")
                & table["test_set"].eq(test_key)
                & table["group"].eq(spec["group"])
                & table["metric"].eq(spec["field"])
                & table["seed"].isin(seeds),
                ["seed", "value"],
            ].copy()
            if selected["seed"].duplicated().any():
                raise ValueError(
                    f"Duplicate MP-only rows for {test_key}/{spec['field']}"
                )
            selected = selected.set_index("seed").reindex(seeds)
            if selected["value"].isna().any():
                missing = selected.index[selected["value"].isna()].tolist()
                raise ValueError(
                    f"Missing MP-only seeds {missing} for {test_key}/{spec['field']}"
                )
            for seed, value in selected["value"].items():
                rows.append({
                    "gene_set": gene_set,
                    "method": "MP only",
                    "seed": int(seed),
                    "metric": metric,
                    "value": float(value),
                })
    return rows


def load_source_data(repo_root, pretrain_only_path, seeds):
    """Combine and validate all four training strategies."""
    rows = load_trained_models(repo_root, seeds)
    rows.extend(load_pretrain_only(pretrain_only_path, seeds))
    data = pd.DataFrame.from_records(rows)
    expected = len(METHOD_ORDER) * len(seeds) * len(TEST_KEYS) * len(METRIC_SPECS)
    if len(data) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(data)}")
    if not np.isfinite(data["value"]).all() or not data["value"].between(0, 1).all():
        raise ValueError("All AUPR, AUROC, and Hits@30 values must lie in [0, 1].")
    return data


def summarize(data):
    """Calculate mean, sample SD, and number of seeds."""
    data = data.copy()
    data["method"] = pd.Categorical(
        data["method"], categories=METHOD_ORDER, ordered=True,
    )
    return (
        data.groupby(["gene_set", "method", "metric"], observed=True)["value"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )


def panel_y_max(matrix, tick_step):
    """Choose a valid rounded upper bound without clipping mean + SD."""
    upper = float((matrix.mean(axis=1) + matrix.std(axis=1, ddof=1)).max())
    return min(1.0, max(2 * tick_step, math.ceil((upper + 0.02) / tick_step) * tick_step))


def draw_panel(ax, data, metric, seeds, show_text=True):
    """Draw mean +/- sample SD with all five seed observations."""
    matrix = (
        data.loc[data["metric"].eq(metric)]
        .pivot(index="method", columns="seed", values="value")
        .loc[list(METHOD_ORDER), list(seeds)]
    )
    x = np.arange(len(METHOD_ORDER), dtype=float)
    means = matrix.mean(axis=1).to_numpy(dtype=float)
    stds = matrix.std(axis=1, ddof=1).to_numpy(dtype=float)
    ax.bar(
        x, means, yerr=stds, width=BAR_WIDTH,
        color=[METHOD_COLORS[name] for name in METHOD_ORDER],
        edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH,
        error_kw={
            "ecolor": EDGE_COLOR,
            "elinewidth": ERRORBAR_LINEWIDTH,
            "capsize": ERRORBAR_CAPSIZE,
            "capthick": ERRORBAR_LINEWIDTH,
        },
        zorder=2,
    )
    point_size = scatter_size_for_bar(
        ax, BAR_WIDTH, max_size=SEED_SCATTER_MAX_SIZE,
    )
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    for index, method in enumerate(METHOD_ORDER):
        ax.scatter(
            index + offsets,
            matrix.loc[method].to_numpy(dtype=float),
            s=point_size,
            color=EDGE_COLOR,
            linewidths=0,
            alpha=1.0,
            zorder=3,
        )

    style_axis(ax)
    ax.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
    ax.set_xticks(x)
    tick_step = METRIC_SPECS[metric]["tick_step"]
    y_max = panel_y_max(matrix, tick_step)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1e-9, tick_step))
    if show_text:
        ax.set_xticklabels(METHOD_ORDER)
        ax.tick_params(axis="x", labelsize=CATEGORY_TICK_SIZE, labelrotation=42)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.set_ylabel(METRIC_SPECS[metric]["label"])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(False)


def render_panel(data, metric, seeds, output_stem, template, show_text):
    """Render and save one standalone panel."""
    fig, ax = create_panel(template)
    draw_panel(ax, data, metric, seeds, show_text=show_text)
    return save_panel(fig, output_stem)


def run(repo_root, output_dir, pretrain_only_path, seeds, template):
    """Write source tables and six publication panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_source_data(repo_root, pretrain_only_path, seeds)
    source_path = output_dir / "external_training_strategy_source_data.csv"
    summary_path = output_dir / "external_training_strategy_summary.csv"
    data.to_csv(source_path, index=False)
    summarize(data).to_csv(summary_path, index=False)

    outputs = [source_path, summary_path]
    for gene_set in TEST_KEYS:
        subset = data.loc[data["gene_set"].eq(gene_set)]
        for metric, spec in METRIC_SPECS.items():
            stem = output_dir / f"external_training_strategy_{gene_set}_{spec['stem']}"
            outputs.extend(render_panel(
                subset, metric, seeds, stem, template, show_text=True,
            ))
            outputs.extend(render_panel(
                subset, metric, seeds, Path(f"{stem}_no_text"), template,
                show_text=False,
            ))
    print(f"Source rows: {len(data)}")
    print(f"Template: {template}")
    for output in outputs:
        print(f"Saved {output}")


def main(argv=None):
    """Generate training-strategy comparison panels."""
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be non-empty and contain no duplicates.")
    run(
        args.repo_root,
        args.output_dir,
        args.pretrain_only,
        seeds,
        args.template,
    )


if __name__ == "__main__":
    main()
