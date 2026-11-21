#!/usr/bin/env python
"""Native-candidate ranking.

Run from the repository root:
    conda run -n deepvcf python figure/external_evaluation/scripts/plot_native_candidate_hits.py --plot-only

Environment: deepvcf.
Outputs: figure/external_evaluation/outputs/native_candidates/.
Rank all methods within each mechanistic baseline scored-candidate set. Filtered ranks use pessimistic ties and the four ECO positive datasets. K = 1, 5, 10, 50, 100, 500, 1000; equally spaced categorical positions. FoldUnit 1 for DeepVCF; five-fold ensembles for +BE and +BE+MP, summarized over five seeds. --plot-only redraws saved ranks; omit it and use --device cuda:0 to recompute scores on an out-of-sandbox GPU.
Exports SVG/PNG figures and source tables, with labeled and no-text variants.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deepvcf-matplotlib")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "reproducibility/external_evaluation"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from reproducibility.external_evaluation import common as external
from figure.external_evaluation.scripts.native_candidate_data import (
    COLORS, METHODS, OUTPUT, native_data,
)
from figure.figure_style import (
    create_panel,
    save_panel,
    style_axis,
    DATA_LINEWIDTH,
    MARKER_SIZE,
    MARKER_EDGE_WIDTH,
    ERRORBAR_LINEWIDTH,
    ERRORBAR_CAPSIZE,
    EDGE_COLOR,
)

SEEDS = (1, 2, 3, 4, 5)
K_VALUES = (1, 5, 10, 50, 100, 500, 1000)
VARIANTS = ("DeepVCF", "DeepVCF+BE", "DeepVCF+BE+MP")


def build_tasks(positives, scores, candidates, known):
    tasks, coverage = [], []
    for method in METHODS:
        covered = informative = 0
        for row in positives.itertuples(index=False):
            key = (row.metabolite, row.relation)
            pool = candidates[method].get(key, [])
            if row.gene not in pool:
                continue
            covered += 1
            filtered = [gene for gene in pool if gene == row.gene or gene not in known[key]]
            informative += 1
            tasks.append({"baseline": method, "positive_index": row.positive_index,
                          "gene": row.gene, "relation": row.relation,
                          "metabolite": row.metabolite, "candidate_count": len(filtered),
                          "native_candidate_count": len(pool), "pool": filtered})
        coverage.append({"baseline": method, "total_positive_rows": len(positives),
                         "native_covered_positive_rows": covered,
                         "informative_positive_rows": informative,
                         "singleton_positive_rows": sum(t["candidate_count"] == 1 for t in tasks if t["baseline"] == method),
                         "positive_coverage": covered / len(positives)})
    return tasks, pd.DataFrame(coverage)


def score_models(tasks, device):
    queries = defaultdict(set)
    for task in tasks:
        queries[(task["metabolite"], task["relation"])].update(task["pool"])
    triples = [(met, rel, gene) for (met, rel), genes in sorted(queries.items()) for gene in sorted(genes)]
    result = {}
    for seed in SEEDS:
        for model_name in ("deepvcf", "deepvcf_mp"):
            folder = ROOT / "results/external_evaluation" / model_name / f"seed_{seed}"
            metadata = json.loads((folder / "metrics.json").read_text())
            config = metadata["config"].copy()
            config["device"] = device
            config["kg_path"] = {"ECO": str(ROOT / "data/processed/kg/ECO/kg_enhanced.txt")}
            model = external.DeepVCFModel.load_model(folder / "model.pt", device=device)
            kg_metadata = model.kg_metadata
            model.eval()
            entities, relations = kg_metadata["node_to_id"], kg_metadata["relation_to_id"]
            values = []
            with torch.inference_mode():
                for start in range(0, len(triples), 2048):
                    batch = triples[start:start + 2048]
                    edges = torch.tensor([[entities[g] for _, _, g in batch],
                                          [entities[m] for m, _, _ in batch]], device=device)
                    rels = torch.tensor([relations[r] for _, r, _ in batch], device=device)
                    fold_scores = torch.stack([fold['tpn'].predict_edge_probability(edges, rels).reshape(-1) for fold in model.folds])
                    values.append(torch.stack([fold_scores[0], fold_scores.mean(0)]).cpu().numpy())
            arrays = np.concatenate(values, axis=1)
            names = ("DeepVCF", "DeepVCF+BE") if model_name == "deepvcf" else (None, "DeepVCF+BE+MP")
            for name, array in zip(names, arrays):
                if name:
                    result[(seed, name)] = dict(zip(triples, array))
            # Check recomputed positive scores against the saved external results.
            saved = pd.read_csv(folder / "gene_corruption_ranks.tsv", sep="\t")
            saved = saved.loc[saved.test_set.eq("gem_gene/head_corrupt")]
            for name, selector in ((names[0], "fold_1"), (names[1], "ensemble")):
                if name is None:
                    continue
                for row in saved.loc[saved.model.eq(selector)].itertuples(index=False):
                    key = (row.metabolite, row.relation, row.gene)
                    if key in result[(seed, name)]:
                        if not np.isclose(result[(seed, name)][key], row.score, atol=2e-6, rtol=1e-5):
                            raise ValueError(f"Checkpoint score mismatch: {seed}/{name}/{key}")
            print(f"Verified inference {model_name}, seed {seed}, {len(triples)} candidates", flush=True)
            del model
            torch.cuda.empty_cache()
    return result


def make_ranks(tasks, native, predictions):
    records = []
    for task in tasks:
        for method in (task["baseline"], *VARIANTS):
            for seed in SEEDS:
                scores = native[method] if method == task["baseline"] else predictions[(seed, method)]
                met, rel = task["metabolite"], task["relation"]
                positive = scores[(met, rel, task["gene"])]
                values = np.array([scores[(met, rel, gene)] for gene in task["pool"]])
                # Match the main benchmark's pessimistic handling of ties.
                rank = int((values >= positive).sum())
                if not 1 <= rank <= len(values):
                    raise ValueError("Invalid filtered rank")
                records.append({**{k: v for k, v in task.items() if k != "pool"},
                                "method": method, "seed": seed, "score": float(positive),
                                "filtered_rank": rank, "hit_at_10": int(rank <= 10)})
    return pd.DataFrame(records)


def plot_panel(data, baseline, output_dir, show_text, x_spacing="equal"):
    fig, ax = create_panel("square_v1")
    methods = (baseline, *VARIANTS)
    x = np.arange(len(K_VALUES)) if x_spacing == "equal" else np.array(K_VALUES)
    for method in methods:
        group = data.loc[data.method.eq(method)]
        summary = group.groupby("k").hits_at_k.agg(["mean", "std"]).reindex(K_VALUES)
        means = summary["mean"].to_numpy()
        spreads = summary["std"].fillna(0).to_numpy()
        artists = ax.errorbar(x, means,
                    yerr=np.stack([np.minimum(spreads, means),
                                   np.minimum(spreads, 1 - means)]),
                    color=COLORS[method], marker="o",
                    markeredgecolor=EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH,
                    markersize=MARKER_SIZE, linewidth=DATA_LINEWIDTH,
                    elinewidth=ERRORBAR_LINEWIDTH, capsize=ERRORBAR_CAPSIZE,
                    label=method, zorder=3, alpha=1, clip_on=False)
        line, caps, stems = artists.lines
        for artist in (line, *caps, *stems):
            artist.set_clip_on(False)
    padding = .35 if x_spacing == "equal" else 25
    ax.set_xlim(x[0] - padding, x[-1] + padding)
    ax.set_xticks(x)
    ax.minorticks_off()
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, .2))
    if show_text:
        ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=12)
        ax.set_xlabel("K", fontsize=13)
        ax.set_ylabel("Filtered Hits@K", fontsize=13)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    stem = output_dir / f"native_candidate_hits_curve_{baseline.lower()}{'' if show_text else '_no_text'}"
    return save_panel(fig, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--x-spacing", choices=("equal", "linear"), default="equal")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "seed_rank_source_data.csv"
    if not args.plot_only:
        positives, native, candidates, known = native_data()
        tasks, coverage = build_tasks(positives, native, candidates, known)
        if any(coverage.informative_positive_rows.eq(0)):
            raise ValueError("A mechanistic model has no informative covered positives")
        coverage.to_csv(args.output_dir / "coverage_summary.csv", index=False)
        predictions = score_models(tasks, args.device)
        make_ranks(tasks, native, predictions).to_csv(source_path, index=False)
    source = pd.read_csv(source_path)
    metric_frames = []
    for k in K_VALUES:
        hits = source.assign(hit_at_k=source.filtered_rank.le(k).astype(int))
        metric = hits.groupby(["baseline", "method", "seed"], sort=False).hit_at_k.agg(
            n_positive="size", n_hits="sum", hits_at_k="mean").reset_index()
        metric["k"] = k
        metric_frames.append(metric)
    seed_metrics = pd.concat(metric_frames, ignore_index=True)
    seed_metrics.to_csv(args.output_dir / "hits_curve_seed_metrics.csv", index=False)
    summary = seed_metrics.groupby(["baseline", "method", "k"]).hits_at_k.agg(
        mean="mean", std="std").reset_index()
    summary.to_csv(args.output_dir / "hits_curve_summary.csv", index=False)
    for baseline in METHODS:
        subset = seed_metrics.loc[seed_metrics.baseline.eq(baseline)]
        for show in (True, False):
            for path in plot_panel(subset, baseline, args.output_dir, show, args.x_spacing):
                print(f"Saved {path}", flush=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
