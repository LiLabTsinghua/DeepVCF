"""Plot pooled split-by-seed internal database-ablation results.

Run from the repository root in the deepvcf environment:
    conda run -n deepvcf python figure/internal_evaluation/scripts/plot_database_ablation.py

Inputs: results/internal_evaluation/{embedding_comparison,database_ablation}/.
Outputs: figure/internal_evaluation/outputs/database_ablation/.
Each condition pools 14 hold-outs and five seeds using FoldUnit 1 (70 points).
Bars show mean and sample SD; all observations are displayed. Exports five
metrics as labeled/no-text SVG and PNG, source/summary CSVs, and combined
CSV/TSV result tables. --results-root and --output-dir override paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    BAR_DOT_BAR_WIDTH_MM,
    BAR_DOT_ERROR_CAP_WIDTH_MM,
    BAR_DOT_POINT_DIAMETER_MM,
    EDGE_COLOR,
    create_bar_dot_panel,
    save_panel,
)


RESULTS_ROOT = PROJECT_ROOT / "results" / "internal_evaluation"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/database_ablation'
SEEDS = (1, 2, 3, 4, 5)
SPLITS = (
    "metabolite_hold_out",
    "amino_acid_hold_out",
    "carbohydrate_hold_out",
    "cofactors_and_vitamins_hold_out",
    "lipid_hold_out",
    "nucleotide_hold_out",
    "secondary_metabolites_hold_out",
    "gem_gene_hold_out",
    "gene_type_hold_out",
    "go_metabolism_related_hold_out",
    "go_signal_transduction_hold_out",
    "go_stress_response_hold_out",
    "go_transcriptional_regulation_hold_out",
    "go_dna_replication_repair_hold_out",
)
CONFIGS = (
    ("enhanced", "Full", "#264653"),
    ("enhanced_no_string", "−STRING", "#2A9D8F"),
    ("enhanced_no_gem", "−GEM", "#E9C46A"),
    ("enhanced_no_kegg", "−KEGG", "#F4A261"),
    ("enhanced_no_go", "−GO", "#E76F51"),
)
FIGURES = (
    ("head_corrupt", "auroc", "Head-corruption AUROC"),
    ("head_corrupt", "aupr", "Head-corruption AUPR"),
    ("tail_corrupt", "auroc", "Tail-corruption AUROC"),
    ("tail_corrupt", "aupr", "Tail-corruption AUPR"),
    ("head_corrupt", "hits30", "Head-corruption Hits@30"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def benchmark_path(root: Path, config: str, seed: int) -> Path:
    group = "embedding_comparison" if config == "enhanced" else "database_ablation"
    return root / group / config / f"seed_{seed}" / "benchmark.json"


def read_benchmarks(root: Path) -> dict[tuple[str, int], dict[str, dict]]:
    """Require all 14 current splits for every configuration and seed."""
    benchmarks = {}
    expected = set(SPLITS)
    for config, _, _ in CONFIGS:
        for seed in SEEDS:
            path = benchmark_path(root, config, seed)
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            splits = result.get("splits", [])
            by_split = {item["split"]: item for item in splits}
            if len(splits) != len(expected) or set(by_split) != expected:
                raise ValueError(f"Unexpected or incomplete split set: {path}")
            filter_path = result.get("ranking_filter_path", "")
            if not filter_path.endswith("ECO_before_2023.txt"):
                raise ValueError(f"Unexpected ranking filter: {path}")
            benchmarks[(config, seed)] = by_split
    return benchmarks


def get_fold_one(split_result: dict, corruption: str) -> dict:
    folds = split_result[corruption]["folds"]
    if len(folds) != 5 or {fold["fold"] for fold in folds} != {1, 2, 3, 4, 5}:
        raise ValueError("Expected exactly five distinct folds per split.")
    return next(fold for fold in folds if fold["fold"] == 1)


def collect_records(
    benchmarks: dict[tuple[str, int], dict[str, dict]],
    corruption: str,
    metric: str,
) -> list[dict]:
    records = []
    for config, label, _ in CONFIGS:
        for split in SPLITS:
            for seed in SEEDS:
                fold = get_fold_one(benchmarks[(config, seed)][split], corruption)
                if metric == "hits30":
                    value = fold["target_gene_ranking"]["gene_hit@30"]
                else:
                    value = fold["binary"]["overall"][metric]
                value = float(value)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(
                        f"Invalid {metric}: {config}/{split}/seed_{seed}: {value}"
                    )
                records.append({
                    "configuration": config,
                    "configuration_label": label,
                    "split": split,
                    "seed": seed,
                    "fold": 1,
                    "corruption": corruption,
                    "metric": "gene_hit@30" if metric == "hits30" else metric,
                    "value": value,
                })
    expected_count = len(CONFIGS) * len(SPLITS) * len(SEEDS)
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} values, found {len(records)}")
    return records


def summarize(records: list[dict]) -> list[dict]:
    output = []
    for config, label, _ in CONFIGS:
        values = [r["value"] for r in records if r["configuration"] == config]
        if len(values) != len(SPLITS) * len(SEEDS):
            raise ValueError(f"Incomplete pooled observations for {config}")
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        if not 0 <= mean - sd <= mean + sd <= 1:
            raise ValueError(f"Mean ± SD outside metric domain for {config}")
        output.append({
            "configuration": config,
            "configuration_label": label,
            "n_splits": len(SPLITS),
            "n_seeds": len(SEEDS),
            "n_observations": len(values),
            "mean": mean,
            "sample_sd": sd,
        })
    return output


def write_csv(path: Path, rows: list[dict], delimiter=',') -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(
    output_stem: Path,
    records: list[dict],
    summaries: list[dict],
    axis_label: str,
    show_text: bool,
) -> None:
    """Draw five fixed-width bars and every split-by-seed observation."""
    fig, ax, groups = create_bar_dot_panel((1, 1, 1, 1, 1))
    centers = [group[0] for group in groups]
    point_size = (BAR_DOT_POINT_DIAMETER_MM * 72 / 25.4) ** 2
    cap_half_width_pt = BAR_DOT_ERROR_CAP_WIDTH_MM * 72 / 25.4 / 2
    summary_by_config = {row["configuration"]: row for row in summaries}

    for center, (config, _, color) in zip(centers, CONFIGS):
        values = [r["value"] for r in records if r["configuration"] == config]
        summary = summary_by_config[config]
        ax.bar(
            center, summary["mean"], width=BAR_DOT_BAR_WIDTH_MM,
            facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.8,
            alpha=1.0, zorder=2,
        )
        ax.errorbar(
            center, summary["mean"], yerr=summary["sample_sd"],
            fmt="none", ecolor=EDGE_COLOR, elinewidth=1.0,
            capsize=cap_half_width_pt, capthick=1.0, alpha=1.0, zorder=3,
        )
        rng = random.Random(20260926 + centers.index(center))
        jitter = [rng.uniform(-2.5, 2.5) for _ in values]
        ax.scatter(
            [center + offset for offset in jitter], values,
            s=point_size, marker="o", facecolor="#111111",
            edgecolors="none", alpha=1.0, zorder=4,
        )

    if records[0]["metric"] == "gene_hit@30":
        upper = max(
            max(row["value"] for row in records),
            max(row["mean"] + row["sample_sd"] for row in summaries),
        )
        upper = min(1.0, math.ceil((upper + 0.05) * 10) / 10)
        ax.set_ylim(0, upper)
        ax.set_yticks(tuple(i / 10 for i in range(0, int(upper * 10) + 1, 2)))
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xticks(centers)
    if show_text:
        ax.set_xticklabels(
            [label for _, label, _ in CONFIGS],
            rotation=90, ha="center", va="top", fontsize=9.5,
        )
        ax.set_ylabel(axis_label, fontsize=13)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        output_stem = Path(f"{output_stem}_no_text")
    ax.grid(False)
    save_panel(fig, output_stem)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = read_benchmarks(args.results_root.resolve())
    combined = {
        config: {
            'configuration': config,
            'configuration_label': label,
            'n_observations': len(SPLITS) * len(SEEDS),
        }
        for config, label, _ in CONFIGS
    }
    summaries = {}
    for corruption, metric, label in FIGURES:
        records = collect_records(benchmarks, corruption, metric)
        summary = summarize(records)
        summaries[(corruption, metric)] = summary
        stem = f"database_ablation_{corruption}_{metric}_pooled"
        write_csv(output_dir / f"{stem}_source_data.csv", records)
        write_csv(output_dir / f"{stem}_summary.csv", summary)
        for show_text in (True, False):
            draw_figure(output_dir / stem, records, summary, label, show_text)
    for corruption, metric, title in (
        ('head_corrupt', 'auroc', 'Head AUROC'),
        ('head_corrupt', 'aupr', 'Head AUPR'),
        ('head_corrupt', 'hits30', 'Head Hits@30'),
        ('tail_corrupt', 'auroc', 'Tail AUROC'),
        ('tail_corrupt', 'aupr', 'Tail AUPR'),
    ):
        for row in summaries[(corruption, metric)]:
            combined[row['configuration']][f'{title} mean'] = row['mean']
            combined[row['configuration']][f'{title} SD'] = row['sample_sd']
    write_csv(output_dir / 'database_ablation_results_table.csv', list(combined.values()))
    display_rows = [
        {
            'Configuration': row['configuration_label'],
            **{
                title: f"{row[f'{title} mean']:.4f} ± {row[f'{title} SD']:.4f}"
                for title in ('Head AUROC', 'Head AUPR', 'Head Hits@30', 'Tail AUROC', 'Tail AUPR')
            },
        }
        for row in combined.values()
    ]
    write_csv(output_dir / 'database_ablation_results_table.tsv', display_rows, delimiter='\t')
    print(f"Saved five figures and source data to {output_dir}")


if __name__ == "__main__":
    main()
