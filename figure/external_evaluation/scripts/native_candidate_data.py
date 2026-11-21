"""Shared mechanistic scored candidates and external positive coverage inputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'reproducibility/external_evaluation'))
from reproducibility.external_evaluation import evaluate_mechanistic_baselines as mech
from figure.figure_style import METHOD_COLORS

METHODS = ("FSEOF", "FVSEOF", "iBridge")


COLORS = {**METHOD_COLORS, "FSEOF": "#F3D9A6", "iBridge": "#C98A3D"}


OUTPUT = Path(__file__).resolve().parents[1] / 'outputs/native_candidates'


def native_data():
    config = mech.SPECIES_CONFIG["ECO"]
    test = mech.read_cf(config["test_path"])
    positives = test.loc[test.label.eq(1)].copy()
    positives["positive_index"] = positives.index
    gem = mech.load_gem_gene_pool(config["kg_path"])
    mapping = mech.load_gene_mapping(config["gene_info_path"], gem)
    metabolites = sorted(set(test.metabolite))
    scores = {
        "FSEOF": mech.load_fseof_scores(config["fseof_root"], metabolites, mapping),
        "FVSEOF": mech.load_fvseof_scores(
            ROOT / "baselines/FVSEOF/ECO/iML1515", metabolites, mapping),
        "iBridge": mech.load_ibridge_scores(config["ibridge_root"], metabolites,
            mech.load_reaction_gene_mapping(config["model_path"], mapping)),
    }
    known = defaultdict(set)
    frame, _ = mech.load_filter_data(config["filter_paths"])
    for gene, relation, metabolite, _ in frame.itertuples(index=False, name=None):
        known[(metabolite, relation)].add(gene)
    candidates = {}
    for method, lookup in scores.items():
        groups = defaultdict(set)
        for metabolite, relation, gene in lookup:
            if gene in gem:
                groups[(metabolite, relation)].add(gene)
        candidates[method] = {key: sorted(genes) for key, genes in groups.items()}
    return positives, scores, candidates, known
