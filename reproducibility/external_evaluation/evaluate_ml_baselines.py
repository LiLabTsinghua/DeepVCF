#!/usr/bin/env python
r"""Fit and evaluate LR/MLP/RF across seeds and embedding sources.

Run in deepvcf on a GPU outside the sandbox, from the repository root:
    python reproducibility/external_evaluation/evaluate_ml_baselines.py \
        --models lr mlp rf --embedding-sources drn random \
        --seeds 1 2 3 4 5 --device cuda:0

DRN embeddings use fold 0. Random embeddings use the same dimensionality
and indexing with an independent seed-specific normalized embedding table.
The classifiers fit the full CF training dataset. Evaluation uses the two
ECO external head-corruption tests and the explicit four-file ranking filter.
--checkpoint accepts a DRN path with {seed}; the default is the seed-specific
enhanced ECO DRN. Each condition writes metrics.json and per-positive ranks.
--dry-run prints the plan; --overwrite replaces selected results.
"""

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reproducibility.external_evaluation.common import (
    DEFAULT_SEEDS,
    EMBEDDING_SOURCES,
    ML_MODEL_NAMES,
    get_model_spec,
    result_method_for_baseline,
    EXTERNAL_RANKING_FILTER_PATHS,
    EXTERNAL_TEST_ROOT,
    RESULTS_ROOT,
    TRAIN_PATH,
    DeepVCFModel,
    configure_logging,
    discover_head_corrupt_test_sets,
    get_drn_checkpoint,
    build_rank_rows,
    load_external_filter_positive_set,
    write_rank_rows,
    write_json,
)

from cf_data import CFDataProcessor, CFTrainDataProcessor
from metric import (
    DEFAULT_RANKING_KS,
    compute_binary_metrics_by_gene_group,
    compute_filtered_target_gene_ranking,
)
from utils import set_seeds


LOGGER = logging.getLogger(__name__)
ML_MODELS = ML_MODEL_NAMES


# =============================================================================
# Command Line
# =============================================================================

def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence used by tests.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=ML_MODELS,
                        default=list(ML_MODELS))
    parser.add_argument(
        '--embedding-sources', '--embedding-source', nargs='+', choices=EMBEDDING_SOURCES,
        default=list(EMBEDDING_SOURCES),
        help=(
            'Node representation used in baseline features. Random embeddings '
            'are seed-specific, fixed, and L2-normalized (default: drn).'
        ),
    )
    parser.add_argument('--checkpoint', help='DRN path, optionally containing {seed}.')
    parser.add_argument('--train-path', type=Path, default=TRAIN_PATH)
    parser.add_argument('--external-root', type=Path,
                        default=EXTERNAL_TEST_ROOT)
    parser.add_argument(
        '--ranking-filter-paths', nargs='+', type=Path,
        default=list(EXTERNAL_RANKING_FILTER_PATHS),
        help=(
            'External positive datasets used for filtered gene ranking. '
            'No train/test labels are added implicitly.'
        ),
    )
    parser.add_argument('--output-root', type=Path, default=RESULTS_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    if args.checkpoint and len(args.seeds) > 1 and '{seed}' not in args.checkpoint:
        parser.error('multiple seeds require {seed} in --checkpoint')
    for name in ('train_path', 'external_root', 'output_root'):
        setattr(args, name, getattr(args, name).resolve())
    args.ranking_filter_paths = [path.resolve() for path in args.ranking_filter_paths]
    return args


# =============================================================================
# Feature Construction and Estimators
# =============================================================================

def extract_features(node_embeddings, edge_index, edge_type, task_rel):
    """Build ``[head embedding | tail embedding | relation one-hot]`` features.

    Args:
        node_embeddings: Normalized node embeddings with shape ``[N, D]``.
        edge_index: Indexed edges with shape ``[2, E]``.
        edge_type: Relation indices with shape ``[E]``.
        task_rel: The two task-relation indices.

    Returns:
        NumPy feature matrix with shape ``[E, 2D + 2]``.

    Raises:
        ValueError: If an edge contains a non-task relation.
    """
    embeddings = node_embeddings.detach().cpu().numpy()
    edges = edge_index.detach().cpu().numpy()
    relations = edge_type.detach().cpu().numpy()
    relation_to_column = {
        int(relation): index
        for index, relation in enumerate(task_rel.detach().cpu().tolist())
    }
    try:
        relation_columns = np.asarray(
            [relation_to_column[int(relation)] for relation in relations],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f'Feature data contains non-task relation index {exc.args[0]}.'
        ) from exc
    relation_one_hot = np.eye(len(relation_to_column), dtype=np.float32)[
        relation_columns
    ]
    return np.concatenate([
        embeddings[edges[0]],
        embeddings[edges[1]],
        relation_one_hot,
    ], axis=1)


def build_node_embeddings(model, kg_metadata, hidden_dim, source, seed):
    """Return the fixed node representation used by a sklearn baseline.

    Random embeddings match the DRN dimensionality and node indexing, but use
    no learned DRN values. A separate deterministic embedding table is sampled
    for every experiment seed and L2-normalized exactly once before training.

    Args:
        model: Checkpoint-aligned DeepVCF model used for DRN embeddings.
        kg_metadata: KG metadata containing the entity index mapping.
        hidden_dim: Embedding dimension from checkpoint metadata.
        source: Either ``drn`` or ``random``.
        seed: Experiment seed used for deterministic random sampling.

    Returns:
        CPU float tensor with shape ``[num_nodes, hidden_dim]``.
    """
    if source == 'drn':
        return model.get_drn_embeddings(fold_index=0).detach().cpu()
    if source != 'random':
        raise ValueError(f'Unknown embedding source {source!r}.')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    embeddings = torch.randn(
        len(kg_metadata['node_to_id']),
        hidden_dim,
        generator=generator,
        dtype=torch.float32,
    )
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def result_method_name(model_name, embedding_source):
    """Return the result ID for one estimator and embedding source."""
    return result_method_for_baseline(model_name, embedding_source)


def build_estimator(name, seed):
    """Construct one fixed sklearn baseline.

    Args:
        name: Baseline name: ``lr``, ``mlp``, or ``rf``.
        seed: Estimator random seed.

    Returns:
        Unfitted sklearn estimator.

    Raises:
        ValueError: If the ML model name is unknown.
    """
    if name == 'lr':
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=seed),
        )
    if name == 'mlp':
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(300, 300),
                max_iter=300,
                early_stopping=False,
                random_state=seed,
            ),
        )
    if name == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f'Unknown ML model {name!r}.')


# =============================================================================
# Training and Evaluation
# =============================================================================

def evaluate_estimator(
    estimator,
    test_data,
    node_embeddings,
    task_rel,
    candidate_pools,
    filter_pos_set,
    include_ranking,
):
    """Evaluate one fitted sklearn estimator on one external test set.

    Args:
        estimator: Fitted sklearn classifier with ``predict_proba``.
        test_data: Indexed external test Data object.
        node_embeddings: Frozen normalized DRN embeddings.
        task_rel: Two task-relation indices.
        candidate_pools: ECO candidate pools used for gene ranking.
        filter_pos_set: Caller-supplied positive triples used for filtering.
        include_ranking: Whether to compute filtered target-gene ranking.

    Returns:
        Dictionary containing binary metrics and optional ranking metrics.
    """
    edge_index = test_data.test_edge_index.cpu()
    edge_type = test_data.test_edge_type.cpu()
    labels = test_data.test_label.cpu()

    def score_fn(query_edge_index, query_edge_type):
        features = extract_features(
            node_embeddings,
            query_edge_index,
            query_edge_type,
            task_rel,
        )
        probabilities = estimator.predict_proba(features)[:, 1].astype(
            np.float32,
        )
        return torch.from_numpy(probabilities).to(query_edge_index.device)

    predictions = score_fn(edge_index, edge_type)
    overall, gem, non_gem, heterologous = compute_binary_metrics_by_gene_group(
        predictions,
        labels,
        edge_index[0],
        candidate_pools,
    )
    result = {
        'binary': {'overall': overall, 'gem': gem, 'non_gem': non_gem,
                   'heterologous': heterologous},
    }
    if include_ranking:
        positive = labels.bool()
        ranking, rank_records = compute_filtered_target_gene_ranking(
            edge_index[:, positive],
            edge_type[positive],
            score_fn,
            candidate_pools['gem_genes'].cpu(),
            candidate_pools['non_gem_genes'].cpu(),
            filter_pos_set,
            ks=DEFAULT_RANKING_KS,
            task_rel=task_rel.cpu(),
            return_records=True,
            heterologous_gene_pool=candidate_pools['heterologous_genes'].cpu(),
        )
        result['target_gene_ranking'] = ranking
        result['target_gene_rank_records'] = rank_records
    return result


def run(args, seed, embedding_source, checkpoint):
    """Fit selected classical ML models and write external-test metrics.

    Args:
        args: Parsed command-line arguments.
        seed: Experiment seed.
        embedding_source: DRN or random node embeddings.
        checkpoint: Complete model saved after DRN training.
    """
    specs = discover_head_corrupt_test_sets(args.external_root)
    if args.dry_run:
        print('models:', ', '.join(args.models))
        print('embedding source:', embedding_source)
        print('train:', args.train_path)
        print('checkpoint:', checkpoint)
        print('ranking filter paths:')
        for path in args.ranking_filter_paths:
            print(' ', path)
        for spec in specs:
            print(f"test: {spec['name']} -> {spec['path']}")
        for name in args.models:
            method = result_method_name(name, embedding_source)
            print(
                'output:',
                args.output_root / method / f'seed_{seed}',
            )
        return

    pending = []
    for name in args.models:
        method = result_method_name(name, embedding_source)
        output_dir = args.output_root / method / f'seed_{seed}'
        if (output_dir / 'metrics.json').is_file() and not args.overwrite:
            print(f'[reuse] {method} seed={seed}', flush=True)
        else:
            pending.append(name)
    if not pending:
        return
    set_seeds(seed)
    model = DeepVCFModel.load_model(checkpoint, device=args.device)
    kg_metadata = model.kg_metadata
    config = model.config
    full_train_data, _ = CFTrainDataProcessor(
        train_path=args.train_path,
        index_root=checkpoint.parent,
        seed=seed,
    ).process_full()
    train_edge_index = full_train_data.train_edge_index
    train_edge_type = full_train_data.train_edge_type
    train_labels = full_train_data.train_label
    # Learned baselines use fold 0's frozen DRN representation. Random
    # controls use only the checkpoint-aligned indexing and embedding width.
    node_embeddings = build_node_embeddings(
        model,
        kg_metadata,
        config['hidden_dim'],
        embedding_source,
        seed,
    )
    train_features = extract_features(
        node_embeddings,
        train_edge_index,
        train_edge_type,
        model.task_rel,
    )
    train_targets = train_labels.numpy().astype(np.int64)
    ranking_filter_pos_set, ranking_filter_rows_by_file = (
        load_external_filter_positive_set(
            args.ranking_filter_paths,
            kg_metadata,
        )
    )
    candidate_pools = kg_metadata['node_group_ids_by_species']['ECO']

    for name in pending:
        method = result_method_name(name, embedding_source)
        output_dir = args.output_root / method / f'seed_{seed}'
        result_path = output_dir / 'metrics.json'
        configure_logging(output_dir / 'run.log', verbose=args.verbose)
        LOGGER.info(
            'Fitting %s with %s embeddings on all %d training rows.',
            name, embedding_source, len(train_targets),
        )
        estimator = build_estimator(name, seed)
        estimator.fit(train_features, train_targets)

        test_results = {}
        rank_rows = []
        for spec in specs:
            test_data = CFDataProcessor(
                spec['path'], checkpoint.parent, split='test',
            ).process()
            evaluation = evaluate_estimator(
                estimator,
                test_data,
                node_embeddings,
                model.task_rel,
                candidate_pools,
                ranking_filter_pos_set,
                spec['include_ranking'],
            )
            rank_records = evaluation.pop('target_gene_rank_records', None)
            if rank_records is not None:
                rank_rows.extend(build_rank_rows(
                    [('single', rank_records)],
                    kg_metadata['node_to_id'],
                    kg_metadata['relation_to_id'],
                    metadata={
                        'seed': seed,
                        'test_set': spec['name'],
                        'gene_group': spec['gene_group'],
                        'corruption': spec['corruption'],
                    },
                ))
            test_results[spec['name']] = evaluation
        rank_path = output_dir / 'gene_corruption_ranks.tsv'
        write_rank_rows(rank_path, rank_rows)
        LOGGER.info('Wrote per-positive gene ranks to %s.', rank_path)
        write_json(result_path, {
            'method': method,
            'model_family': get_model_spec(method).family,
            'model_label': get_model_spec(method).label,
            'seed': seed,
            'checkpoint': checkpoint,
            'embedding_source': embedding_source,
            'embedding_dim': int(node_embeddings.shape[1]),
            'embedding_seed': (
                seed if embedding_source == 'random' else None
            ),
            'train_path': args.train_path,
            'num_train_rows': len(train_targets),
            'ranking_filter_paths': args.ranking_filter_paths,
            'ranking_filter_rows_by_file': ranking_filter_rows_by_file,
            'ranking_filter_base': 'external_positive_datasets',
            'num_ranking_filter_base_positives': len(ranking_filter_pos_set),
            'validation_early_stopping': False,
            'test_sets': test_results,
        })


def main(argv=None):
    """Fit and evaluate selected ML models for each seed and embedding source."""
    args = parse_args(argv)
    for seed in args.seeds:
        checkpoint = (Path(args.checkpoint.format(seed=seed)) if args.checkpoint
                      else get_drn_checkpoint(seed)).resolve()
        for source in args.embedding_sources:
            run(args, seed, source, checkpoint)


if __name__ == '__main__':
    main()
