"""Shared configuration and result helpers for internal evaluation."""

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))



# =============================================================================
# Experiment Layout
# =============================================================================

INTERNAL_DATA_ROOT = (
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'evaluation' / 'internal_test'
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / 'results' / 'internal_evaluation'
RANKING_FILTER_PATH = (
    REPO_ROOT / 'data' / 'processed' / 'cf' / 'datasets'
    / 'ECO_before_2023.txt'
)
DEFAULT_DRN_ROOT = REPO_ROOT / 'results' / 'drn' / 'ECO'


EMBEDDING_VARIANTS = ('random', 'basic', 'enhanced')
DATABASE_ABLATION_VARIANTS = (
    'enhanced_no_string', 'enhanced_no_gem',
    'enhanced_no_kegg', 'enhanced_no_go',
)


def get_experiment_group(variant):
    """Return the comparison group for an internal-evaluation variant."""
    return (
        'embedding_comparison' if variant in EMBEDDING_VARIANTS
        else 'database_ablation'
    )


def load_ranking_filter_positives(path, kg_metadata):
    """Index known positives using the model's node and relation mappings.

    Basic KG has a smaller entity vocabulary than Enhanced KG. Positives
    outside the model vocabulary cannot be ranking candidates and are skipped.
    """
    node_to_id = kg_metadata['node_to_id']
    relation_to_id = kg_metadata['relation_to_id']
    positives = set()
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.reader(handle, delimiter='\t'):
            head, relation, tail = (value.strip() for value in row)
            if head in node_to_id and tail in node_to_id and relation in relation_to_id:
                positives.add((node_to_id[head], relation_to_id[relation], node_to_id[tail]))
    return positives


def get_drn_checkpoint(variant, seed):
    """Locate the complete model saved after DRN training for one variant."""
    root = (DEFAULT_DRN_ROOT / 'database_ablation'
            if variant in DATABASE_ABLATION_VARIANTS else DEFAULT_DRN_ROOT)
    return root / variant / f'seed_{seed}' / 'drn_trained_model.pt'


# =============================================================================
# Split Discovery
# =============================================================================

def get_split_group(split_name):
    """Classify an internal split by its required KG coverage.

    Args:
        split_name: Split-directory name.

    Returns:
        Either ``'core'`` or ``'enhanced_only'``.
    """
    if split_name.startswith('go_') or split_name in {
        'gene_type_hold_out', 'non_gem_gene_hold_out',
    }:
        return 'enhanced_only'
    return 'core'


def discover_splits(variant, internal_root=INTERNAL_DATA_ROOT, requested=None):
    """Select prepared split directories supported by the model variant.

    Basic uses core splits. The remaining variants also use enhanced-only
    splits. Explicitly requested split names retain the same selection rule.
    """
    root = Path(internal_root)
    splits = ([root / name for name in requested] if requested is not None
              else sorted(path for path in root.iterdir() if path.is_dir()))
    if variant == 'basic':
        if requested and any(get_split_group(path.name) == 'enhanced_only' for path in splits):
            raise ValueError('Basic KG supports only core splits.')
        splits = [path for path in splits if get_split_group(path.name) == 'core']
    return sorted(splits)


# =============================================================================
# Logging and Result Serialization
# =============================================================================

def configure_logging(log_path=None, verbose=False):
    """Configure concise console logging and an optional file log.

    Args:
        log_path: Optional log-file path.
        verbose: Whether console logging should use INFO instead of WARNING.
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
    """Convert nested tensor, NumPy, and path values for JSON output."""
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
    """Write one deterministic, human-readable JSON file.

    Args:
        path: Destination path.
        payload: JSON-compatible or convertible payload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def extract_metric_summary(binary_result, ranking_result=None):
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


def iter_metric_rows(split_result):
    """Flatten one split result into long-form benchmark rows.

    Args:
        split_result: Compact result returned by
            ``train_and_evaluate_split``.

    Yields:
        Dictionaries suitable for the aggregate TSV file.
    """
    base = {
        'variant': split_result['variant'],
        'split': split_result['split'],
        'split_group': split_result['split_group'],
        'seed': split_result['seed'],
    }
    for corruption in ('head_corrupt', 'tail_corrupt'):
        evaluation = split_result[corruption]
        model_results = [('ensemble', evaluation['ensemble'])]
        model_results.extend(
            (f"fold_{item['fold']}", item) for item in evaluation['folds']
        )
        for model_name, model_result in model_results:
            # Collect binary and ranking metrics with their report group names.
            metric_groups = list(model_result['binary'].items())
            for target in ('gene', 'metabolite'):
                metric_groups.append((
                    f'target_{target}', model_result.get(f'target_{target}_ranking'),
                ))
                metric_groups.extend(model_result.get(
                    f'target_{target}_ranking_by_gene_group', {}
                ).items())
            for group, metrics in metric_groups:
                if metrics is None:
                    continue
                for metric, value in metrics.items():
                    yield {
                        **base, 'corruption': corruption, 'model': model_name,
                        'group': group, 'metric': metric, 'value': value,
                    }


def write_benchmark_tsv(path, split_results):
    """Write a long-form aggregate benchmark table.

    Args:
        path: Destination TSV path.
        split_results: Iterable of compact split results.
    """
    fieldnames = (
        'variant', 'split', 'split_group', 'seed', 'corruption',
        'model', 'group', 'metric', 'value',
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for split_result in split_results:
            writer.writerows(iter_metric_rows(split_result))


