"""Experiment paths, test selection, and ranking filters for species transfer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# DeepVCF Cross-Species Transfer
# =============================================================================

DEFAULT_PAIRS = ("ECO_SCE", "ECO_CGL", "ECO_BSU", "ECO_PPU")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
KG_GROUP_SPECIES = {
    "ECO_SCE": ("ECO", "SCE"),
    "ECO_CGL": ("ECO", "CGL"),
    "ECO_BSU": ("ECO", "BSU"),
    "ECO_PPU": ("ECO", "PPU"),
    "ECO_SCE_CGL": ("ECO", "SCE", "CGL"),
}
ECO_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "cf" / "train" / "train.txt"
TARGET_TEST_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "cf"
    / "evaluation"
    / "cross_species_transfer"
)
TARGET_DATASET_ROOT = REPO_ROOT / "data" / "processed" / "cf" / "datasets"
KG_ROOT = REPO_ROOT / "data" / "processed" / "kg"
RESULTS_ROOT = REPO_ROOT / "results" / "cross_species_transfer"
DEFAULT_KG_LEVEL = "enhanced"
GENE_GROUPS = ("gem_gene", "non_gem_gene")
CORRUPTIONS = ("head_corrupt", "tail_corrupt")


# =============================================================================
# Experiment Paths
# =============================================================================

def get_target_species(pair):
    """Return the target species for a two-species KG pair."""
    return KG_GROUP_SPECIES[pair][1]


def get_zero_shot_run_dir(output_root, pair, seed):
    """Return the output directory for one zero-shot pair and seed.

    Args:
        output_root (str or Path): Cross-species result root.
        pair (str): Pair key.
        seed (int): Experiment seed.

    Returns:
        pathlib.Path: Directory containing model and evaluation artifacts.
    """
    return Path(output_root) / "zero_shot" / pair / f"seed_{seed}"


def get_zero_shot_evaluation_dir(output_root, kg_group, seed, target_species):
    """Return the target-specific evaluation directory for one model.

    Args:
        output_root (str or Path): Cross-species result root.
        kg_group (str): Species group used to build the KG/DRN.
        seed (int): Experiment seed.
        target_species (str): Species used for candidate scoping and testing.

    Returns:
        pathlib.Path: Directory containing target-specific metrics and ranks.
    """
    return (
        get_zero_shot_run_dir(output_root, kg_group, seed)
        / "evaluation"
        / target_species
    )


def get_few_shot_run_dir(output_root, pair, shot, seed):
    """Return the output directory for one few-shot pair, shot, and seed.

    Args:
        output_root (str or Path): Cross-species result root.
        pair (str): Pair key.
        shot (int): Number of target-species positive examples in the support
            file.
        seed (int): Experiment seed.

    Returns:
        pathlib.Path: Directory containing model and evaluation artifacts.
    """
    return (
        Path(output_root) / "few_shot" / pair / f"shot_{shot}"
        / f"seed_{seed}"
    )


def discover_test_sets(target_species, root=TARGET_TEST_ROOT):
    """Select the available prepared tests for the requested target species.

    Some species have fewer test sets, so only existing files are included.
    Head-corruption tests also receive ranking evaluation.
    """
    species_root = Path(root) / target_species
    specs = []
    for group in GENE_GROUPS:
        for corruption in CORRUPTIONS:
            path = species_root / group / f'{corruption}_test.txt'
            if path.is_file():
                specs.append({
                    'name': f'{group}/{corruption}', 'gene_group': group,
                    'corruption': corruption, 'path': path,
                    'include_ranking': corruption == 'head_corrupt',
                })
    if not specs:
        raise FileNotFoundError(f'No cross-species test files found in {species_root}')
    return specs


# =============================================================================
# Target-Species Ranking Filter
# =============================================================================

def load_target_filter_set(path, kg_metadata):
    """Index covered target positives for ranking, using the model's mappings.

    The target dataset only filters other known positives from rankings.
    Three-column files contain positives; four-column files include labels.
    Triples outside this model's vocabulary cannot be candidates and are skipped.
    """
    frame = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
    if frame.shape[1] == 4:
        frame = frame.loc[frame[3].astype(int) == 1, [0, 1, 2]]
    frame.columns = ['h', 'r', 't']
    frame = frame.apply(lambda column: column.str.strip())
    node_to_id = kg_metadata['node_to_id']
    relation_to_id = kg_metadata['relation_to_id']
    return {
        (node_to_id[row.h], relation_to_id[row.r], node_to_id[row.t])
        for row in frame.itertuples(index=False)
        if row.h in node_to_id and row.t in node_to_id and row.r in relation_to_id
    }
