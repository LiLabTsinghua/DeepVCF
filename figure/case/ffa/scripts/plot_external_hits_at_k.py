#!/usr/bin/env python
"""Compare external FFA Hits@10/20/30 with hdca_c mechanistic baselines.

Run from the repository root in deepvcf:
    python figure/case/ffa/scripts/plot_external_hits_at_k.py

Inputs: data/processed/cf/case/ffa/, the ECO basic KG, and the FSEOF/FVSEOF/
iBridge ECO baseline tables for hdca_c. FVSEOF files are under the dhca_c
directory, with hdca_c filenames.
Outputs: figure/case/ffa/outputs/external_dataset_validation/:
    xu_2013_oe/; fang_2021_ko/gem_covered/; fang_2021_ko/gem_uncovered/.
Xu uses OE, Fang uses KO. Covered panels compare all four methods; the
uncovered panel shows DeepVCF alone. DeepVCF ranks its full 4,136-gene list
before GEM grouping. Mechanistic methods rank their scored GEM-covered
recommendations. Ties receive the minimum rank; missing predictions are misses.
Each rate divides hits by all external positive targets in that dataset/group:
15 Xu covered, 18 Fang covered, and 27 Fang uncovered. Seven Fang uncovered
targets have no DeepVCF score. GEM coverage is based on Catalyzes-source
proteins in the ECO basic KG. One deterministic ranking is used per method,
so no seed-level error bars or inferential statistics are shown.
Exports labeled/no-text SVG/600-dpi PNG, per-target ranks, and Hits@K source CSVs."""

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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "reproducibility/external_evaluation"))
from figure.figure_style import (  # noqa: E402
    DATA_LINEWIDTH, EDGE_COLOR, MARKER_EDGE_WIDTH, MARKER_SIZE, METHOD_COLORS,
    create_panel, save_panel, style_axis,
)
from reproducibility.external_evaluation.evaluate_mechanistic_baselines import (  # noqa: E402
    load_gem_gene_pool, load_gene_mapping, sigmoid,
    load_fseof_scores, load_ibridge_scores, load_reaction_gene_mapping,
)

DATA_DIR = ROOT / "data/processed/cf/case/ffa"
FIGURE_DIR = Path(__file__).resolve().parents[1] / 'outputs/external_dataset_validation'
FVSEOF_ROOT = ROOT / "baselines/FVSEOF/ECO/iML1515"
FSEOF_ROOT = ROOT / "baselines/FSEOF/ECO/Targets_iML1515_c_new"
IBRIDGE_ROOT = ROOT / "baselines/iBridge/ECO/application_result_iML1515_c_new"
GENE_INFO_PATH = ROOT / "data/raw/ECO/eco_gene_info.txt"
KG_PATH = ROOT / "data/processed/kg/ECO/kg_basic.txt"
MODEL_PATH = ROOT / "data/raw/ECO/iML1515.json"
METABOLITE = "Metabolite:hdca_c"
KS = (10, 20, 30)
DATASETS = (
    ("Xu 2013", "OE", "xu_2013nc.txt", "xu_2013_oe"),
    ("Fang 2021", "KO", "fang_2021nc.txt", "fang_2021_ko"),
)
METHODS = ("DeepVCF", "FSEOF", "FVSEOF", "iBridge")
COLORS = {
    ("DeepVCF", "GEM-covered"): METHOD_COLORS["DeepVCF+BE"],
    ("DeepVCF", "GEM-uncovered"): METHOD_COLORS["DeepVCF+BE"],
    ("FSEOF", "GEM-covered"): "#F3D9A6",
    ("FVSEOF", "GEM-covered"): "#E6AD87",
    ("iBridge", "GEM-covered"): "#C98A3D",
}


def load_targets(gem_nodes):
    tables = []
    for dataset, modification, filename, _folder in DATASETS:
        data = pd.read_csv(
            DATA_DIR / "external_datasets" / filename,
            sep="\t", header=None,
            names=["protein_node", "gene", "category", "label"],
        )
        if data[["gene", "category", "label"]].isna().any().any():
            raise ValueError(f"Incomplete external target table: {dataset}")
        if not data["label"].isin((0, 1)).all() or data["gene"].duplicated().any():
            raise ValueError(f"Invalid labels or repeated genes: {dataset}")
        data.insert(0, "dataset", dataset)
        data.insert(1, "modification", modification)
        data["gem_group"] = np.where(
            data["protein_node"].isin(gem_nodes), "GEM-covered", "GEM-uncovered",
        )
        tables.append(data)
    return pd.concat(tables, ignore_index=True)


def load_deepvcf_candidates():
    data = pd.read_csv(DATA_DIR / "pred_202505.txt", sep="\t")
    required = {"uniprot_id", "name", "modification", "score"}
    if not required.issubset(data.columns):
        raise ValueError("Prediction table lacks required columns")
    if data[["uniprot_id", "name", "modification", "score"]].isna().any().any():
        raise ValueError("Incomplete DeepVCF predictions")
    if not data["modification"].isin(("OE", "KO")).all():
        raise ValueError("Unexpected DeepVCF intervention")
    if data.duplicated(["uniprot_id", "modification"]).any() or not np.isfinite(data["score"]).all():
        raise ValueError("Repeated genes or invalid DeepVCF scores")
    data["protein_node"] = "ECO_Protein:" + data["uniprot_id"]
    return {
        modification: data.loc[data["modification"].eq(modification),
                               ["protein_node", "score"]].copy()
        for modification in ("OE", "KO")
    }


def load_fvseof_scores_for_hdca(locus_to_protein, gem_nodes):
    """Use files named hdca_c even though this product's directory is misspelled."""
    folder = FVSEOF_ROOT / "hdca_c"
    if not folder.is_dir():
        folder = FVSEOF_ROOT / "dhca_c"
    scores = {}
    for modification, direction, sign in (("OE", "up", 1.0), ("KO", "down", -1.0)):
        path = folder / f"hdca_c_{direction}_targets_sorted.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, index_col=0)
        if frame.columns.tolist() != ["Rank"]:
            raise ValueError(f"Unexpected FVSEOF schema: {path}")
        best = {}
        for locus, raw in frame["Rank"].items():
            signed = sign * float(raw)
            protein = locus_to_protein.get(str(locus))
            if protein in gem_nodes and np.isfinite(signed) and signed > 0:
                best[protein] = max(sigmoid(signed), best.get(protein, 0.0))
            elif not np.isfinite(signed):
                raise ValueError(f"Invalid FVSEOF score for {locus}")
        scores[modification] = best
    return scores


def load_mechanistic_candidates(gem_nodes):
    gem_pool = set(load_gem_gene_pool(KG_PATH))
    if gem_pool != gem_nodes:
        raise ValueError("Current mechanistic GEM pool differs from the FFA coverage definition")
    locus_to_protein = load_gene_mapping(GENE_INFO_PATH, gem_pool)
    reaction_to_genes = load_reaction_gene_mapping(MODEL_PATH, locus_to_protein)
    fseof = load_fseof_scores(FSEOF_ROOT, [METABOLITE], locus_to_protein)
    ibridge = load_ibridge_scores(IBRIDGE_ROOT, [METABOLITE], reaction_to_genes)
    fvseof = load_fvseof_scores_for_hdca(locus_to_protein, gem_pool)
    mechanistic = {}
    for method, score_map in (("FSEOF", fseof), ("iBridge", ibridge)):
        for modification, relation in (("OE", "overexpress"), ("KO", "knock out")):
            best = {
                gene: score for (product, rel, gene), score in score_map.items()
                if product == METABOLITE and rel == relation and gene in gem_pool
            }
            mechanistic[(method, modification)] = best
    for modification, best in fvseof.items():
        mechanistic[("FVSEOF", modification)] = best
    candidates = {}
    for key, best in mechanistic.items():
        if not best or not all(np.isfinite(list(best.values()))):
            raise ValueError(f"No usable mechanistic recommendations: {key}")
        candidates[key] = pd.DataFrame(
            [(gene, score) for gene, score in best.items()],
            columns=["protein_node", "score"],
        )
    return candidates


def rank_candidates(frame):
    if frame["protein_node"].isna().any() or frame["protein_node"].duplicated().any():
        raise ValueError("Invalid candidate identifiers")
    ranked = frame.copy()
    ranked["rank_min"] = ranked["score"].rank(method="min", ascending=False)
    ranked["rank_max"] = ranked["score"].rank(method="max", ascending=False)
    ranked["rank_used"] = ranked["rank_min"]
    return ranked


def build_source():
    gem_nodes = set(load_gem_gene_pool(ROOT / "data/processed/kg/ECO/kg_basic.txt"))
    targets = load_targets(gem_nodes)
    deepvcf = load_deepvcf_candidates()
    mechanistic = load_mechanistic_candidates(gem_nodes)
    detailed = []
    summaries = []
    for dataset, modification, _filename, _folder in DATASETS:
        dataset_targets = targets.loc[targets["dataset"].eq(dataset)]
        for method in METHODS:
            raw = deepvcf[modification] if method == "DeepVCF" else mechanistic[(method, modification)]
            ranked = rank_candidates(raw)
            joined = dataset_targets.merge(
                ranked, on="protein_node", how="left", validate="many_to_one",
            )
            joined.insert(2, "method", method)
            joined["n_candidates"] = len(ranked)
            detailed.append(joined)
            groups = ("GEM-covered", "GEM-uncovered") if method == "DeepVCF" else ("GEM-covered",)
            for group in groups:
                subset = joined.loc[joined["gem_group"].eq(group)]
                if subset.empty:
                    continue
                positives = subset.loc[subset["label"].eq(1)]
                n_total = len(positives)
                n_ranked = int(positives["rank_used"].notna().sum())
                if n_total == 0:
                    continue
                for k in KS:
                    hits = int(positives["rank_used"].le(k).sum())
                    summaries.append({
                        "dataset": dataset,
                        "modification": modification,
                        "method": method,
                        "gem_group": group,
                        "k": k,
                        "hits": hits,
                        "n_positive_total": n_total,
                        "n_positive_ranked": n_ranked,
                        "n_positive_unranked": n_total - n_ranked,
                        "n_candidates": len(ranked),
                        "hits_at_k_all_positives": hits / n_total,
                    })
    detail = pd.concat(detailed, ignore_index=True)
    source = pd.DataFrame(summaries)
    if source["hits_at_k_all_positives"].isna().any() or not source["hits_at_k_all_positives"].between(0, 1).all():
        raise ValueError("Invalid external Hits@K rate")
    return detail, source


def plot_panel(source, dataset, gem_group, output_dir, show_text=True):
    series = [("DeepVCF", gem_group)]
    if gem_group == "GEM-covered":
        series = [
            ("FSEOF", gem_group),
            ("FVSEOF", gem_group),
            ("iBridge", gem_group),
            ("DeepVCF", gem_group),
        ]
    fig, ax = create_panel("square_v1")
    ymax = 0.4 if dataset == "Xu 2013" else (0.3 if gem_group == "GEM-covered" else 0.12)
    for method, group in series:
        rows = source.loc[
            source["dataset"].eq(dataset)
            & source["method"].eq(method)
            & source["gem_group"].eq(group)
        ].sort_values("k")
        if rows["k"].tolist() != list(KS):
            raise ValueError(f"Missing method/coverage series: {dataset}, {method}, {group}")
        values = rows["hits_at_k_all_positives"].to_numpy(dtype=float)
        color = COLORS[(method, group)]
        ax.plot(
            KS, values, color=color,
            marker="o", markerfacecolor=color,
            markeredgecolor=EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=MARKER_SIZE, linewidth=DATA_LINEWIDTH,
            alpha=1.0, label=method, clip_on=False, zorder=4,
        )
    ax.set_xlim(9, 31)
    ax.set_ylim(0, ymax)
    ax.set_xticks(KS)
    if dataset == "Xu 2013":
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    elif gem_group == "GEM-covered":
        ax.set_yticks([0, 0.1, 0.2, 0.3])
    else:
        ax.set_yticks([0, 0.04, 0.08, 0.12])
    if show_text:
        ax.set_xlabel("K")
        ax.set_ylabel("Hits@K")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    style_axis(ax)
    ax.grid(False)
    stem = output_dir / ("external_hits_at_k" if show_text else "external_hits_at_k_no_text")
    return save_panel(fig, stem)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    detail, source = build_source()
    panels = (
        ("Xu 2013", "GEM-covered", "xu_2013_oe"),
        ("Fang 2021", "GEM-covered", "fang_2021_ko/gem_covered"),
        ("Fang 2021", "GEM-uncovered", "fang_2021_ko/gem_uncovered"),
    )
    for dataset, gem_group, folder in panels:
        output_dir = FIGURE_DIR / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        allowed_methods = METHODS if gem_group == "GEM-covered" else ("DeepVCF",)
        detail.loc[
            detail["dataset"].eq(dataset)
            & detail["gem_group"].eq(gem_group)
            & detail["method"].isin(allowed_methods)
        ].to_csv(output_dir / "external_target_ranks_by_method.csv", index=False)
        subset = source.loc[
            source["dataset"].eq(dataset)
            & source["gem_group"].eq(gem_group)
            & source["method"].isin(allowed_methods)
        ]
        subset.to_csv(output_dir / "hits_at_k_source_data.csv", index=False)
        for show_text in (True, False):
            for output in plot_panel(subset, dataset, gem_group, output_dir, show_text):
                print(f"Saved {output}")
        print(subset[["method", "gem_group", "k", "hits", "n_positive_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
