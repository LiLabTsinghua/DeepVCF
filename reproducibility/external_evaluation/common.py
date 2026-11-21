"""Shared construction and result helpers for external evaluation."""

from dataclasses import dataclass
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
REPRODUCIBILITY_DIR = REPO_ROOT / 'reproducibility'
if str(REPRODUCIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(REPRODUCIBILITY_DIR))

from model import DeepVCFModel  # noqa: E402
from rank_records import build_rank_rows, write_rank_rows  # noqa: E402


# =============================================================================
# Experiment Layout
# =============================================================================

TRAIN_PATH = REPO_ROOT / 'data' / 'processed' / 'cf' / 'train' / 'train.txt'
EXTERNAL_TEST_ROOT = (
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'evaluation' / 'external_test'
)
EXTERNAL_RANKING_FILTER_PATHS = (
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'datasets' / 'ECO_2023.txt',
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'datasets' / 'ECO_2024.txt',
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'datasets'
    / 'ECO_before_2023.txt',
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'datasets' / 'laser.txt',
)
DRN_ROOT = REPO_ROOT / 'results' / 'drn' / 'ECO' / 'enhanced'
EXTERNAL_EVALUATION_ROOT = REPO_ROOT / 'results' / 'external_evaluation'
RESULTS_ROOT = EXTERNAL_EVALUATION_ROOT
GENE_GROUPS = ('gem_gene', 'non_gem_gene')
CORRUPTIONS = ('head_corrupt', 'tail_corrupt')


def get_drn_checkpoint(seed):
    """Return the enhanced ECO DRN checkpoint for one experiment seed.

    Args:
        seed: Experiment random seed.

    Returns:
        Seed-specific DRN checkpoint path.
    """
    return DRN_ROOT / f'seed_{seed}' / 'drn_trained_model.pt'


def load_external_filter_positive_set(paths, kg_metadata):
    """Index the supplied positive datasets with the model's mappings.

    Three-column files contain positive triples. Four-column files include
    labels, so only their positive rows contribute to filtered ranking.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    node_to_id = kg_metadata['node_to_id']
    relation_to_id = kg_metadata['relation_to_id']
    positives, rows_by_file = set(), {}
    for path in paths:
        frame = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
        if frame.shape[1] == 4:
            frame = frame.loc[frame[3].astype(int) == 1, [0, 1, 2]]
        frame.columns = ['h', 'r', 't']
        frame = frame.apply(lambda column: column.str.strip())
        rows_by_file[str(path)] = len(frame)
        positives.update(
            (node_to_id[row.h], relation_to_id[row.r], node_to_id[row.t])
            for row in frame.itertuples(index=False)
        )
    return positives, rows_by_file


# =============================================================================
# External Test Discovery
# =============================================================================

def discover_test_sets(root=EXTERNAL_TEST_ROOT, corruptions=CORRUPTIONS):
    """Describe the prepared external tests for each gene group."""
    return [
        {
            'name': f'{group}/{corruption}', 'gene_group': group,
            'corruption': corruption,
            'path': Path(root) / group / f'{corruption}_test.txt',
            'include_ranking': corruption == 'head_corrupt',
        }
        for group in GENE_GROUPS for corruption in corruptions
    ]


def discover_head_corrupt_test_sets(root=EXTERNAL_TEST_ROOT):
    """Select the two external head-corruption tests."""
    return discover_test_sets(root, corruptions=('head_corrupt',))


# =============================================================================
# Logging and Serialization
# =============================================================================

def configure_logging(log_path=None, verbose=False):
    """Configure console logging and an optional file log.

    Args:
        log_path: Optional log-file path.
        verbose: Whether console logging should use INFO level.
    """
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=handlers,
        force=True,
    )
    handlers[0].setLevel(logging.INFO if verbose else logging.WARNING)


def to_jsonable(value):
    """Convert nested tensors, NumPy values, and paths for JSON output."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    """Write deterministic, human-readable JSON.

    Args:
        path: Destination path.
        payload: JSON-compatible or convertible payload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def compact_deepvcf_result(binary_result, ranking_result=None):
    """Retain reportable per-fold and ensemble metrics from both test methods.

    Args:
        binary_result (dict): Output of ``evaluate_deepvcf(test_type='binary')``.
        ranking_result (dict or None): Optional ranking evaluation output.

    Returns:
        dict: Ensemble and per-fold binary metrics, with overall and per-group
            ranking metrics when ranking results are supplied.
    """
    groups = ('overall', 'gem', 'non_gem', 'heterologous')
    summary = {
        'ensemble': {'binary': {
            group: binary_result['ensemble']['binary'][group] for group in groups
        }},
        'folds': [
            {'fold': item['fold'], 'binary': {
                group: item['binary'][group] for group in groups
            }}
            for item in binary_result['fold_results']
        ],
    }
    if ranking_result is not None:
        ranking_fields = (
            'target_gene_ranking', 'target_gene_ranking_by_gene_group',
            'target_metabolite_ranking', 'target_metabolite_ranking_by_gene_group',
        )
        summary['ensemble'].update({
            name: ranking_result['ensemble'][name] for name in ranking_fields
        })
        for target, item in zip(summary['folds'], ranking_result['fold_results']):
            target.update({name: item[name] for name in ranking_fields})
    return summary


def build_tpn_rank_rows(
    evaluation,
    spec,
    seed,
    kg_metadata,
):
    """Build fold and ensemble rank rows for one test set.

    Args:
        evaluation: Result returned by ``DeepVCFModel.evaluate_deepvcf`` with
            ``return_rank_records=True``.
        spec: External test-set descriptor.
        seed: Experiment seed.
        kg_metadata: Model metadata containing node and relation mappings.

    Returns:
        list[dict]: Serialized rank rows; empty when ranking was not computed.
    """
    records_by_model = [
        (f"fold_{item['fold']}", item['target_gene_rank_records'])
        for item in evaluation['fold_results']
    ]
    records_by_model.append(('ensemble', evaluation['ensemble']['target_gene_rank_records']))
    return build_rank_rows(
        records_by_model,
        kg_metadata['node_to_id'],
        kg_metadata['relation_to_id'],
        metadata={
            'seed': seed,
            'test_set': spec['name'],
            'gene_group': spec['gene_group'],
            'corruption': spec['corruption'],
        },
    )



# Result identifiers shared by the three evaluation entry points.
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
MODE_TO_METHOD = {
    'default': 'deepvcf',
    'pretrain_only': 'deepvcf_pretrain_only',
    'mp': 'deepvcf_mp',
    'joint_training': 'deepvcf_joint_training',
}
ML_MODEL_NAMES = ('lr', 'mlp', 'rf')
EMBEDDING_SOURCES = ('drn', 'random')
MECHANISTIC_BASELINE_METHODS = ('fseof', 'fvseof', 'ibridge')


@dataclass(frozen=True)
class ModelSpec:
    method: str
    label: str
    family: str


MODEL_SPECS = {
    method: ModelSpec(method, label, 'deepvcf')
    for method, label in zip(MODE_TO_METHOD.values(), (
        'DeepVCF / default', 'DeepVCF / pretraining only',
        'DeepVCF / pretrain → fine-tune', 'DeepVCF / joint training',
    ))
}
for _name in ML_MODEL_NAMES:
    MODEL_SPECS[_name] = ModelSpec(_name, _name.upper() + ' / DRN embedding', 'ml_baseline')
    _method = _name + '_random_embedding'
    MODEL_SPECS[_method] = ModelSpec(_method, _name.upper() + ' / random embedding', 'ml_baseline')
for _name, _label in (('fseof', 'FSEOF'), ('fvseof', 'FVSEOF'), ('ibridge', 'iBridge')):
    MODEL_SPECS[_name] = ModelSpec(_name, _label, 'mechanistic_baseline')


def get_model_spec(method):
    return MODEL_SPECS[method]


def result_method_for_baseline(model_name, embedding_source):
    if model_name not in ML_MODEL_NAMES or embedding_source not in EMBEDDING_SOURCES:
        raise ValueError(f'Unknown ML condition: {model_name}/{embedding_source}')
    return model_name if embedding_source == 'drn' else f'{model_name}_random_embedding'



