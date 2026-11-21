#!/usr/bin/env python
"""Evaluate FSEOF, FVSEOF, and iBridge on five species' GEM test sets.

The external test files contain gene-corruption triples.  For filtered
gene-ranking metrics, ECO uses the union of ``ECO_before_2023``, ``ECO_2023``,
``ECO_2024``, and ``laser``; each other species uses its own dataset file.

Example:
    python reproducibility/external_evaluation/evaluate_mechanistic_baselines.py \
        --species ECO --methods fseof fvseof ibridge \
        --seeds 1 2 3 4 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Project imports
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_DIR = REPO_ROOT / "code"
REPRODUCIBILITY_DIR = REPO_ROOT / "reproducibility"
for import_path in (CODE_DIR, REPRODUCIBILITY_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


from metric import (  # noqa: E402
    DEFAULT_RANKING_KS, compute_binary_metrics, compute_filtered_target_gene_ranking,
)
from rank_records import write_rank_records  # noqa: E402

from reproducibility.external_evaluation.common import (  # noqa: E402
    MECHANISTIC_BASELINE_METHODS,
    configure_logging, write_json, get_model_spec,
)


LOGGER = logging.getLogger(__name__)
TASK_RELATIONS = ('knock out', 'overexpress')


# =============================================================================
# Experiment layout
# =============================================================================

DATA_ROOT = REPO_ROOT / "data"
CF_ROOT = DATA_ROOT / "processed" / "cf"
KG_ROOT = DATA_ROOT / "processed" / "kg"
RAW_ROOT = DATA_ROOT / "raw"
BASELINES_ROOT = REPO_ROOT / "baselines"
RESULTS_ROOT = REPO_ROOT / "results"
SPECIES_TO_PAIR = {
    "CGL": "ECO_CGL", "SCE": "ECO_SCE",
    "BSU": "ECO_BSU", "PPU": "ECO_PPU",
}

SPECIES_CONFIG = {
    "ECO": {
        "model": "iML1515",
        "kg_path": KG_ROOT / "ECO" / "kg_enhanced.txt",
        "gene_info_path": RAW_ROOT / "ECO" / "eco_gene_info.txt",
        "model_path": RAW_ROOT / "ECO" / "iML1515.json",
        "fseof_root": BASELINES_ROOT / "FSEOF" / "ECO"
        / "Targets_iML1515_c_new",
        "ibridge_root": BASELINES_ROOT / "iBridge" / "ECO"
        / "application_result_iML1515_c_new",
        "test_path": CF_ROOT / "evaluation" / "external_test" / "gem_gene"
        / "head_corrupt_test.txt",
        "filter_paths": (
            CF_ROOT / "datasets" / "ECO_before_2023.txt",
            CF_ROOT / "datasets" / "ECO_2023.txt",
            CF_ROOT / "datasets" / "ECO_2024.txt",
            CF_ROOT / "datasets" / "laser.txt",
        ),
    },
    "CGL": {
        "model": "iCW773",
        "kg_path": KG_ROOT / "CGL" / "kg_enhanced.txt",
        "gene_info_path": RAW_ROOT / "CGL" / "cgl_gene_info.txt",
        "model_path": RAW_ROOT / "CGL" / "iCW773.json",
        "fseof_root": BASELINES_ROOT / "FSEOF" / "CGL"
        / "Targets_iCW773_c_new",
        "ibridge_root": BASELINES_ROOT / "iBridge" / "CGL"
        / "application_result_iCW773_c_new",
        "test_path": CF_ROOT / "evaluation" / "cross_species_transfer" / "CGL"
        / "gem_gene" / "head_corrupt_test.txt",
        "filter_paths": (CF_ROOT / "datasets" / "CGL_dataset.txt",),
    },
    "SCE": {
        "model": "iMM904",
        "kg_path": KG_ROOT / "SCE" / "kg_enhanced.txt",
        "gene_info_path": RAW_ROOT / "SCE" / "sce_gene_info.txt",
        "model_path": RAW_ROOT / "SCE" / "iMM904.json",
        "fseof_root": BASELINES_ROOT / "FSEOF" / "SCE"
        / "Targets_iMM904_c_new",
        "ibridge_root": BASELINES_ROOT / "iBridge" / "SCE"
        / "application_result_iMM904_c_new",
        "test_path": CF_ROOT / "evaluation" / "cross_species_transfer" / "SCE"
        / "gem_gene" / "head_corrupt_test.txt",
        "filter_paths": (CF_ROOT / "datasets" / "SCE_dataset.txt",),
    },
    "BSU": {
        "model": "iYO844",
        "kg_path": KG_ROOT / "BSU" / "kg_enhanced.txt",
        "gene_info_path": RAW_ROOT / "BSU" / "uniprot_B_subtilis.tsv",
        "model_path": RAW_ROOT / "BSU" / "iYO844.json",
        "fseof_root": BASELINES_ROOT / "FSEOF" / "BSU"
        / "Targets_iYO844_c_new",
        "ibridge_root": BASELINES_ROOT / "iBridge" / "BSU"
        / "application_result_iYO844_c_new",
        "test_path": CF_ROOT / "evaluation" / "cross_species_transfer" / "BSU"
        / "gem_gene" / "head_corrupt_test.txt",
        "filter_paths": (CF_ROOT / "datasets" / "BSU_dataset.txt",),
    },
    "PPU": {
        "model": "iJN1463",
        "kg_path": KG_ROOT / "PPU" / "kg_enhanced.txt",
        "gene_info_path": RAW_ROOT / "PPU" / "uniprot_mapping.tsv",
        "model_path": RAW_ROOT / "PPU" / "iJN1463.json",
        "fseof_root": BASELINES_ROOT / "FSEOF" / "PPU"
        / "Targets_iJN1463_c_new",
        "ibridge_root": BASELINES_ROOT / "iBridge" / "PPU"
        / "application_result_iJN1463_c_new",
        "test_path": CF_ROOT / "evaluation" / "cross_species_transfer" / "PPU"
        / "gem_gene" / "head_corrupt_test.txt",
        "filter_paths": (CF_ROOT / "datasets" / "PPU_dataset.txt",),
    },
}

DEFAULT_SPECIES = tuple(SPECIES_CONFIG)
DEFAULT_METHODS = MECHANISTIC_BASELINE_METHODS
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
RANKING_METRICS = (
    "gene_mrr",
    "gene_hit@10",
    "gene_hit@20",
    "gene_hit@30",
)


# =============================================================================
# Input processing
# =============================================================================

def read_cf(path):
    """Read a four-column counterfactual data file.

    Args:
        path: Counterfactual data path.

    Returns:
        pandas.DataFrame: Data with gene, relation, metabolite, and label
        columns.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If its shape, labels, or task relations are invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Counterfactual data not found: {path}')
    frame = pd.read_csv(path, sep='\t', header=None, dtype=str)
    if frame.shape[1] != 4:
        raise ValueError(f'{path} must contain four tab-separated columns.')
    frame.columns = ['gene', 'relation', 'metabolite', 'label']
    frame['label'] = pd.to_numeric(frame['label'], errors='raise').astype(int)
    if not frame['label'].isin((0, 1)).all():
        raise ValueError(f'{path} labels must contain only 0 and 1.')
    unknown_relations = set(frame['relation']) - set(TASK_RELATIONS)
    if unknown_relations:
        raise ValueError(
            f'{path} contains unknown task relations: '
            f'{sorted(unknown_relations)}.'
        )
    return frame


def read_positive_dataset(path):
    """Read a positive counterfactual dataset used for rank filtering.

    Args:
        path: Three-column ``gene, relation, metabolite`` dataset, or a
            four-column labeled counterfactual file.

    Returns:
        pandas.DataFrame: Positive rows with ``gene``, ``relation``,
            ``metabolite``, and ``label`` columns.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file shape or labels are invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Filter dataset not found: {path}")
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if frame.shape[1] == 3:
        frame.columns = ["gene", "relation", "metabolite"]
        frame["label"] = 1
    elif frame.shape[1] == 4:
        frame.columns = ["gene", "relation", "metabolite", "label"]
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
        if not frame["label"].isin((0, 1)).all():
            raise ValueError(f"{path} labels must contain only 0 and 1.")
        frame = frame.loc[frame["label"] == 1].copy()
    else:
        raise ValueError(f"{path} must contain three or four tab-separated columns.")
    unknown_relations = set(frame["relation"]) - set(TASK_RELATIONS)
    if unknown_relations:
        raise ValueError(
            f"{path} contains unknown task relations: {sorted(unknown_relations)}."
        )
    return frame[["gene", "relation", "metabolite", "label"]].copy()


def load_filter_data(paths):
    """Combine and deduplicate positive triples used by filtered ranking.

    Args:
        paths: Iterable of positive dataset paths.

    Returns:
        tuple[pandas.DataFrame, dict]: Combined positive frame and per-file
            row counts for result metadata.
    """
    frames = []
    counts = {}
    for path in paths:
        frame = read_positive_dataset(path)
        frames.append(frame)
        counts[str(path)] = int(len(frame))
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["gene", "relation", "metabolite"],
    )
    combined["label"] = 1
    return combined.reset_index(drop=True), counts


def load_gene_mapping(path, gem_gene_pool):
    """Load the ECO locus-tag to KG-protein mapping.

    Args:
        path: Tab-separated ECO gene-information file.
        gem_gene_pool: Current GEM-gene identifiers used to resolve aliases.

    Returns:
        dict: Locus tag to ``ECO_Protein`` identifier mapping.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If a locus tag maps to multiple current GEM genes.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Gene mapping not found: {path}')
    frame = pd.read_csv(path, sep='\t', header=None, dtype=str)
    candidates = {}
    for uniprot_id, locus_tag in frame.iloc[:, [0, 2]].itertuples(index=False):
        protein = f'ECO_Protein:{uniprot_id}'
        candidates.setdefault(locus_tag, []).append(protein)
    gem_gene_pool = set(gem_gene_pool)
    mapping = {}
    for locus_tag, proteins in candidates.items():
        gem_candidates = sorted(set(proteins) & gem_gene_pool)
        if len(gem_candidates) > 1:
            raise ValueError(
                f'Locus tag {locus_tag!r} maps to multiple current GEM genes: '
                f'{gem_candidates}.'
            )
        mapping[locus_tag] = (
            gem_candidates[0] if gem_candidates else proteins[-1]
        )
    return mapping


def _species_gene_pattern(species):
    """Return the gene-ID pattern used by one species' GEM model."""
    patterns = {
        "ECO": r"b\d+",
        "CGL": r"(?:cg|Cgl)\d+",
        "SCE": r"Y[A-Z0-9]+(?:-[A-Z0-9]+)?",
        "BSU": r"BSU\d+",
        "PPU": r"PP_[A-Za-z0-9_-]+",
    }
    try:
        return re.compile(patterns[species])
    except KeyError as exc:
        raise ValueError(f"Unsupported species {species!r}.") from exc


def load_species_gene_mapping(species, path, gem_gene_pool):
    """Map species model gene IDs to KG protein identifiers.

    Args:
        species: Species key (``ECO``, ``CGL``, ``SCE``, ``BSU``, or ``PPU``).
        path: Tab-separated UniProt-to-gene mapping file.
        gem_gene_pool: Current KG GEM protein identifiers.

    Returns:
        dict[str, str]: Model gene/locus ID to KG protein identifier mapping.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError: If one model ID maps to multiple current GEM proteins.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Gene mapping not found: {path}")
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    pattern = _species_gene_pattern(species)
    candidates = {}
    # The compact mapping tables use stable identifier columns for ECO/CGL/
    # SCE.  BSU and PPU contain several alternative locus columns, so those
    # two are scanned in full; restricting the first three species avoids
    # matching locus-like text in protein descriptions.
    identifier_columns = {
        "ECO": (2,),
        "CGL": (2, 3),
        "SCE": (2,),
        "BSU": None,
        "PPU": None,
    }[species]
    for row in frame.itertuples(index=False, name=None):
        if not row:
            continue
        uniprot = str(row[0]).strip()
        if not uniprot or uniprot.lower() in {"nan", "entry"}:
            continue
        protein = f"{species}_Protein:{uniprot}"
        values = (
            row[1:]
            if identifier_columns is None
            else (row[index] for index in identifier_columns if index < len(row))
        )
        for value in values:
            if value is None or pd.isna(value):
                continue
            for token in str(value).split():
                if pattern.fullmatch(token):
                    candidates.setdefault(token, set()).add(protein)

    gem_gene_pool = set(gem_gene_pool)
    mapping = {}
    for gene_id, proteins in candidates.items():
        gem_candidates = sorted(proteins & gem_gene_pool)
        if len(gem_candidates) > 1:
            raise ValueError(
                f"Model gene {gene_id!r} maps to multiple current GEM proteins: "
                f"{gem_candidates}."
            )
        mapping[gene_id] = gem_candidates[0] if gem_candidates else sorted(proteins)[0]
    return mapping


def load_gem_gene_pool(path):
    """Load the current ECO GEM-gene candidate pool from the KG.

    Args:
        path: Enhanced ECO knowledge-graph path.

    Returns:
        list[str]: Sorted GEM protein identifiers.

    Raises:
        FileNotFoundError: If the KG does not exist.
        ValueError: If no GEM genes are found.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Knowledge graph not found: {path}')
    kg = pd.read_csv(path, sep='\t', header=None, names=['head', 'rel', 'tail'])
    genes = sorted(set(kg.loc[kg['rel'] == 'Catalyzes', 'head']))
    if not genes:
        raise ValueError(f'No Catalyzes-source GEM genes found in {path}.')
    return genes


def load_reaction_gene_mapping(model_path, gene_mapping):
    """Map GEM reactions to KG protein identifiers using gene rules.

    Args:
        model_path: COBRA JSON model path.
        gene_mapping: Model gene ID to KG protein mapping.

    Returns:
        dict[str, list[str]]: Reaction ID to sorted protein IDs.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If the model does not contain a reaction list.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"GEM model not found: {model_path}")
    with model_path.open(encoding="utf-8") as handle:
        model = json.load(handle)
    if "reactions" not in model:
        raise ValueError(f"GEM model has no reactions list: {model_path}")
    reaction_to_genes = {}
    for reaction in model["reactions"]:
        rule = reaction.get("gene_reaction_rule", "") or ""
        tokens = set(re.findall(r"[A-Za-z0-9_][A-Za-z0-9_-]*", rule))
        reaction_to_genes[reaction["id"]] = sorted({
            gene_mapping[token] for token in tokens if token in gene_mapping
        })
    return reaction_to_genes


# =============================================================================
# Method scores
# =============================================================================

def sigmoid(value):
    """Convert a finite raw method score to a probability-like value.

    Args:
        value: Raw numeric score.

    Returns:
        float: Logistic transform in ``[0, 1]``.
    """
    value = float(np.clip(value, -700.0, 700.0))
    return 1.0 / (1.0 + math.exp(-value))


def complete_scores(ranked_scores, metabolites, gene_pool, seed):
    """Assign reproducible low scores to unranked GEM genes.

    Args:
        ranked_scores: Deterministic method scores.
        metabolites: Full KG metabolite identifiers.
        gene_pool: Full current ECO GEM-gene candidate pool.
        seed: Random-padding seed.

    Returns:
        dict: Complete score lookup for both task relations.
    """
    rng = random.Random(seed)
    scores = dict(ranked_scores)
    for metabolite in sorted(metabolites):
        for relation in TASK_RELATIONS:
            for gene in gene_pool:
                key = (metabolite, relation, gene)
                if key not in scores:
                    scores[key] = sigmoid(rng.uniform(-1.0, 0.0))
    return scores


def load_fseof_scores(root, metabolites, gene_mapping):
    """Load positive FSEOF scores for requested metabolites and directions.

    Args:
        root: FSEOF result directory.
        metabolites: KG metabolite identifiers.
        gene_mapping: FSEOF locus/model ID to KG protein mapping.

    Returns:
        dict: ``(metabolite, relation, gene)`` to deterministic score.
    """
    ranked = {}
    specifications = (
        ("knock out", "_Down_regulation_targets.tsv", -1.0),
        ("overexpress", "_Up_regulation_targets.tsv", 1.0),
    )
    for metabolite in sorted(metabolites):
        metabolite_id = metabolite.removeprefix("Metabolite:")
        for relation, suffix, sign in specifications:
            path = Path(root) / metabolite_id / f"{metabolite_id}{suffix}"
            if not path.is_file():
                continue
            frame = pd.read_csv(path, sep="\t")
            if "Rank" not in frame.columns:
                raise ValueError(f"FSEOF file lacks a Rank column: {path}")
            gene_column = frame.columns[0]
            for locus_tag, raw_rank in frame[[gene_column, "Rank"]].itertuples(
                index=False,
            ):
                rank = pd.to_numeric(raw_rank, errors="coerce")
                if pd.isna(rank):
                    continue
                signed_rank = sign * float(rank)
                gene = gene_mapping.get(str(locus_tag))
                if signed_rank > 0 and gene is not None:
                    key = (metabolite, relation, gene)
                    ranked[key] = max(float(sigmoid(signed_rank)), ranked.get(key, 0.0))
    return ranked


def load_fvseof_scores(root, metabolites, gene_mapping):
    """Read FVSEOF CSVs using FSEOF's signed-Rank scoring convention.

    Missing targets are left for seeded score completion. Duplicate mapped
    genes retain their strongest recommendation. Invalid scores fail loudly.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'FVSEOF predictions not found: {root}')
    ranked = {}
    found = 0
    for metabolite in sorted(metabolites):
        target = metabolite.removeprefix('Metabolite:')
        for relation, direction, sign in (
            ('knock out', 'down', -1.0), ('overexpress', 'up', 1.0),
        ):
            path = root / target / f'{target}_{direction}_targets_sorted.csv'
            if not path.is_file():
                continue
            found += 1
            frame = pd.read_csv(path, index_col=0)
            if list(frame.columns) != ['Rank']:
                raise ValueError(f'Expected gene index and Rank column: {path}')
            for locus, raw in frame['Rank'].items():
                rank = float(raw)
                if not math.isfinite(rank):
                    raise ValueError(f'Non-finite Rank for {locus} in {path}')
                gene = gene_mapping.get(str(locus))
                if sign * rank > 0 and gene is not None:
                    key = (metabolite, relation, gene)
                    ranked[key] = max(sigmoid(sign * rank), ranked.get(key, 0.0))
    if not found:
        raise ValueError(f'No FVSEOF prediction files for requested targets: {root}')
    return ranked


def load_ibridge_scores(root, metabolites, reaction_to_genes):
    """Load positive iBridge reaction scores and project them to genes.

    Args:
        root: iBridge result directory.
        metabolites: KG metabolite identifiers.
        reaction_to_genes: GEM reaction-to-protein mapping.

    Returns:
        dict: ``(metabolite, relation, gene)`` to deterministic score.

    Raises:
        ValueError: If an iBridge file is an intermediate metabolite-score
            table and therefore lacks the required gene/reaction columns.
    """
    ranked = {}
    specifications = (
        ("knock out", "down", "Native_down"),
        ("overexpress", "up", "Native"),
    )
    required = {
        "Gene source", "Positive score", "Negative score", "Reaction",
    }
    expected_files = 0
    found_files = 0
    for metabolite in sorted(metabolites):
        metabolite_id = metabolite.removeprefix("Metabolite:")
        for relation, direction, gene_source in specifications:
            expected_files += 1
            filename = (
                f"New_reaction_candidate_{direction}_Candidate_Final_"
                f"MetScore_DM_{metabolite_id}.txt"
            )
            path = Path(root) / filename
            if not path.is_file():
                continue
            found_files += 1
            frame = pd.read_csv(path, sep="\t")
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(
                    f"iBridge file {path} lacks gene-level columns {missing}; "
                    "it appears to be an intermediate metabolite-score table."
                )
            frame = frame.loc[frame["Gene source"] == gene_source].copy()
            frame["difference"] = (
                pd.to_numeric(frame["Positive score"], errors="coerce")
                - pd.to_numeric(frame["Negative score"], errors="coerce")
            )
            frame = frame.loc[frame["difference"] > 0]
            best_by_gene = {}
            for reaction, difference in frame[["Reaction", "difference"]].itertuples(
                index=False,
            ):
                if pd.isna(reaction) or pd.isna(difference):
                    continue
                for gene in reaction_to_genes.get(str(reaction), ()):
                    best_by_gene[gene] = max(
                        float(difference),
                        best_by_gene.get(gene, -np.inf),
                    )
            for gene, difference in best_by_gene.items():
                ranked[(metabolite, relation, gene)] = sigmoid(difference)
    if expected_files and not found_files:
        existing_files = sorted(Path(root).glob("*.txt"))
        if existing_files:
            raise ValueError(
                f"iBridge root {root} contains no direction-specific gene-level "
                "files named New_reaction_candidate_{down,up}_...; the available "
                "files appear to be intermediate metabolite-score tables."
            )
        raise FileNotFoundError(
            f"No iBridge prediction files found under {root} for the requested "
            "test metabolites."
        )
    return ranked


# =============================================================================
# Ranking and metrics
# =============================================================================

def build_index_data(filter_data, test, gene_pool):
    """Create compact tensors required by the project ranking implementation.

    Args:
        filter_data: Explicit positive-filter data; only positive rows are used.
        test: GEM-gene external head-corruption data.
        gene_pool: Current ECO GEM-gene candidate pool.

    Returns:
        dict: Entity mappings, tensors, and filtered positive set.

    Raises:
        ValueError: If a test gene is absent from the GEM pool.
    """
    missing = sorted(set(test['gene']) - set(gene_pool))
    if missing:
        raise ValueError(
            'Mechanistic baselines support only GEM-gene evaluation, but '
            f'{len(missing)} test genes are absent from the current GEM '
            f'candidate pool; examples: {missing[:10]}.'
        )
    entities = sorted(
        set(gene_pool)
        | set(filter_data['gene']) | set(filter_data['metabolite'])
        | set(test['gene']) | set(test['metabolite'])
    )
    entity_to_index = {entity: index for index, entity in enumerate(entities)}
    relation_to_index = {
        relation: index for index, relation in enumerate(TASK_RELATIONS)
    }

    def tensors(frame):
        edge_index = torch.from_numpy(np.stack([
            frame['gene'].map(entity_to_index).to_numpy(),
            frame['metabolite'].map(entity_to_index).to_numpy(),
        ])).long()
        edge_type = torch.tensor(
            frame['relation'].map(relation_to_index).to_numpy(),
            dtype=torch.long,
        )
        labels = torch.tensor(frame['label'].to_numpy(), dtype=torch.float32)
        return edge_index, edge_type, labels

    filter_edges, filter_relations, filter_labels = tensors(filter_data)
    test_edges, test_relations, test_labels = tensors(test)
    full_positive_set = set()
    positive = filter_labels.bool()
    positive_edges = filter_edges[:, positive]
    positive_relations = filter_relations[positive]
    for index in range(positive_edges.size(1)):
        full_positive_set.add((
            int(positive_edges[0, index]),
            int(positive_relations[index]),
            int(positive_edges[1, index]),
        ))

    gem_pool_tensor = torch.tensor(
        [entity_to_index[gene] for gene in gene_pool], dtype=torch.long,
    )
    return {
        'index_to_entity': entities,
        'test_edges': test_edges,
        'test_relations': test_relations,
        'test_labels': test_labels,
        'gem_pool': gem_pool_tensor,
        'full_positive_set': full_positive_set,
    }


def evaluate(scores, indexed, return_records=False, device='cpu'):
    """Evaluate one complete method score lookup with project metrics.

    Args:
        scores: ``(metabolite, relation, gene)`` score lookup.
        indexed: Output of :func:`build_index_data`.
        return_records: Whether to also return one rank record per positive
            test edge.
        device: Torch device for score tensors and metric evaluation.

    Returns:
        dict or tuple: Binary and filtered target-gene ranking metrics; when
        ``return_records`` is true, the second item is the list of rank
        records returned by ``compute_filtered_target_gene_ranking``.
    """
    device = torch.device(device)
    indexed = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in indexed.items()
    }
    entities = indexed['index_to_entity']

    def score_fn(edge_index, edge_type):
        values = []
        for (gene_index, metabolite_index), relation_index in zip(
            edge_index.t().cpu().tolist(), edge_type.cpu().tolist(),
        ):
            gene = entities[gene_index]
            metabolite = entities[metabolite_index]
            relation = TASK_RELATIONS[relation_index]
            values.append(scores[(metabolite, relation, gene)])
        return torch.tensor(values, dtype=torch.float32, device=device)

    test_edges = indexed['test_edges']
    test_relations = indexed['test_relations']
    test_labels = indexed['test_labels']
    binary = compute_binary_metrics(
        score_fn(test_edges, test_relations), test_labels,
    )
    positive = test_labels.bool()
    ranking_result = compute_filtered_target_gene_ranking(
        test_edges[:, positive],
        test_relations[positive],
        score_fn,
        indexed['gem_pool'],
        torch.empty(0, dtype=torch.long, device=device),
        indexed['full_positive_set'],
        ks=DEFAULT_RANKING_KS,
        task_rel=torch.tensor([0, 1], dtype=torch.long, device=device),
        return_records=return_records,
    )
    if return_records:
        ranking, rank_records = ranking_result
        return {
            'binary': binary,
            'target_gene_ranking': ranking,
        }, rank_records
    return {'binary': binary, 'target_gene_ranking': ranking_result}


# =============================================================================
# Serialization
# =============================================================================

# =============================================================================
# Experiment entry point
# =============================================================================

def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence used by tests.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", choices=DEFAULT_SPECIES,
                        default=list(DEFAULT_SPECIES))
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def evaluate_one(species, method, seed, args, config, test, filter_data,
                 gene_pool, deterministic_scores, indexed):
    """Evaluate one species/method/seed combination and write its artifacts.

    Args:
        species: Species key.
        method: Baseline key.
        seed: Random-padding seed.
        args: Parsed command-line arguments.
        config: Species configuration dictionary.
        test: External GEM-gene test frame.
        filter_data: Positive triples used for ranking filtering.
        gene_pool: Sorted KG GEM-gene pool.
        deterministic_scores: Seed-independent baseline scores.
        indexed: Indexed tensors from :func:`build_index_data`.

    Returns:
        dict: Binary and ranking metrics for this run.
    """
    if species == "ECO":
        output_dir = args.output_root / "external_evaluation" / method / f"seed_{seed}"
    else:
        output_dir = (
            args.output_root / "cross_species_transfer" / "zero_shot"
            / SPECIES_TO_PAIR[species] / f"seed_{seed}" / method
        )
    metrics_path = output_dir / "metrics.json"
    rank_path = output_dir / "gene_corruption_ranks.tsv"
    if metrics_path.is_file() and not args.overwrite:
        print(f'[reuse] {species}/{method} seed={seed}', flush=True)
        return json.loads(metrics_path.read_text())['metrics']
    configure_logging(output_dir / "run.log")
    LOGGER.info("Evaluating %s/%s seed %d.", species, method, seed)

    metabolites = sorted(set(test["metabolite"]))
    scores = complete_scores(
        deterministic_scores, metabolites, gene_pool, seed,
    )
    result, rank_records = evaluate(
        scores, indexed, return_records=True, device=args.device,
    )
    relation_to_index = {relation: index for index, relation in enumerate(TASK_RELATIONS)}
    entity_to_index = {
        entity: index for index, entity in enumerate(indexed["index_to_entity"])
    }
    write_rank_records(
        rank_path,
        [(method, rank_records)],
        entity_to_index,
        relation_to_index,
        metadata={
            "seed": seed,
            "test_set": "gem_gene/head_corrupt",
            "gene_group": "gem_gene",
            "corruption": "head_corrupt",
        },
    )
    write_json(
        metrics_path,
        {
            "method": method,
            "model_family": get_model_spec(method).family,
            "model_label": get_model_spec(method).label,
            "seed": seed,
            "species": species,
            "model": config["model"],
            "device": args.device,
            "gene_group": "gem_gene",
            "corruption": "head_corrupt",
            "randomness": "seed-controlled low-score padding for unranked GEM genes",
            "score_semantics": "method-native ranked scores with seed-controlled "
            "low-score padding for unranked GEM genes",
            "test_path": str(config["test_path"]),
            "filter_paths": [str(path) for path in config["filter_paths"]],
            "ranking_filter_base": "explicit_positive_datasets",
            "filter_rows_by_file": filter_data[1],
            "num_filter_rows": int(len(filter_data[0])),
            "num_test_rows": int(len(test)),
            "num_positive_test_rows": int(test["label"].sum()),
            "num_gem_candidates": int(len(gene_pool)),
            "num_deterministic_scores": int(len(deterministic_scores)),
            "metrics": result,
        },
    )
    return result


def main(argv=None):
    """Load each method's scores, evaluate them, and save metrics and ranks."""
    args = parse_args(argv)
    args.output_root = args.output_root.resolve()
    if args.dry_run:
        for species in args.species:
            config = SPECIES_CONFIG[species]
            print(species, config["test_path"])
            for method in args.methods:
                for seed in args.seeds:
                    if species == 'ECO':
                        destination = args.output_root / 'external_evaluation' / method / f'seed_{seed}'
                    else:
                        destination = (args.output_root / 'cross_species_transfer/zero_shot'
                                       / SPECIES_TO_PAIR[species] / f'seed_{seed}' / method)
                    print(f'[plan] {species}/{method} seed={seed} -> {destination}')
        return

    for species in args.species:
        config = SPECIES_CONFIG[species]
        test = read_cf(config['test_path'])
        filter_frame, filter_counts = load_filter_data(config['filter_paths'])
        gene_pool = load_gem_gene_pool(config['kg_path'])
        indexed = build_index_data(filter_frame, test, gene_pool)
        metabolites = sorted(set(test['metabolite']))
        gene_mapping = load_species_gene_mapping(species, config['gene_info_path'], gene_pool)
        for method in args.methods:
            if method == 'fvseof':
                deterministic = load_fvseof_scores(
                    BASELINES_ROOT / 'FVSEOF' / species / config['model'],
                    metabolites, gene_mapping,
                )
            elif method == 'fseof':
                deterministic = load_fseof_scores(config['fseof_root'], metabolites, gene_mapping)
            else:
                reaction_to_genes = load_reaction_gene_mapping(config['model_path'], gene_mapping)
                deterministic = load_ibridge_scores(config['ibridge_root'], metabolites, reaction_to_genes)
            for seed in args.seeds:
                evaluate_one(species, method, seed, args, config, test,
                             (filter_frame, filter_counts), gene_pool, deterministic, indexed)
    print('Per-seed metrics and ranks written beside the external/zero-shot runs.')


if __name__ == "__main__":
    main()
