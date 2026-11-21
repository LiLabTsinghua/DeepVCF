#!/usr/bin/env python
r"""Train and evaluate ECO internal hold-outs using prepared KGs and DRNs.

Run from the repository root in deepvcf, on a GPU outside the sandbox:
    python reproducibility/internal_evaluation/evaluate_deepvcf.py \
        --group embedding_comparison --seeds 1 2 3 4 5 --device cuda:0
    python reproducibility/internal_evaluation/evaluate_deepvcf.py \
        --group database_ablation --seeds 1 --dry-run
    python reproducibility/internal_evaluation/evaluate_deepvcf.py \
        --variants basic enhanced --seeds 1 --splits amino_acid_hold_out

Each hold-out calls training/train_tpn.py with its train.txt, then evaluates
head/tail tests. Head tests additionally report filtered target-gene ranking
using ECO_before_2023.txt. Basic supports eight splits; other variants support
all fourteen. DRNs are under results/drn/ECO/{variant}/seed_{seed}/drn_trained_model.pt,
with database-ablation variants under ECO/database_ablation/.

Outputs: results/internal_evaluation/{group}/{variant}/seed_{seed}/ contains
benchmark.json, benchmark.tsv, and splits/{split}/metrics.json. Each split's
training/seed_{seed}/ contains model.pt, index_mappings/, and training records.
Split metrics and saved training models are reused. --overwrite
replaces selected outputs; --dry-run prints commands without creating files.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from common import (
    DATABASE_ABLATION_VARIANTS, DEFAULT_RESULTS_ROOT, EMBEDDING_VARIANTS,
    INTERNAL_DATA_ROOT, RANKING_FILTER_PATH, REPO_ROOT,
    configure_logging, discover_splits, extract_metric_summary,
    get_experiment_group, get_split_group, get_drn_checkpoint,
    load_ranking_filter_positives, write_benchmark_tsv, write_json,
)

from cf_data import CFDataProcessor
from model import DeepVCFModel

GROUPS = {
    'embedding_comparison': EMBEDDING_VARIANTS,
    'database_ablation': DATABASE_ABLATION_VARIANTS,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--group', choices=tuple(GROUPS))
    selection.add_argument('--variants', '--variant', nargs='+', choices=sorted(EMBEDDING_VARIANTS + DATABASE_ABLATION_VARIANTS))
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--checkpoint', type=str,
                        help='Complete model after DRN training, optionally containing {seed}.')
    parser.add_argument('--internal-root', type=Path, default=INTERNAL_DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--ranking-filter-path', type=Path, default=RANKING_FILTER_PATH)
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tpn-lr', type=float, default=5e-5)
    parser.add_argument('--tpn-wd', type=float, default=1e-5)
    parser.add_argument('--tpn-batch-size', type=int, default=128)
    parser.add_argument('--tpn-epochs', type=int, default=500)
    parser.add_argument('--tpn-patience', type=int, default=30)
    parser.add_argument('--tpn-loss', choices=('CE', 'BCE'), default='CE')
    parser.add_argument('--tpn-early-stop-metric',
                        choices=('mean_group_aupr', 'unfiltered_gene_mrr'),
                        default='unfiltered_gene_mrr')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    args.variants = list(GROUPS[args.group]) if args.group else args.variants
    if len(args.variants) != len(set(args.variants)):
        parser.error('--variants must be distinct')
    if len(args.seeds) != len(set(args.seeds)) or any(
            seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('--seeds must be distinct and in [0, 2**32)')
    if args.checkpoint and (len(args.variants) != 1 or (
            len(args.seeds) > 1 and '{seed}' not in args.checkpoint)):
        parser.error('--checkpoint requires one variant and {seed} for multiple seeds')
    for name in ('internal_root', 'output_root', 'ranking_filter_path'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def training_command(args, seed, checkpoint, split_dir, output_dir):
    """Build the command that trains TPNs for this internal split."""
    command = [
        sys.executable, str(REPO_ROOT / 'reproducibility/training/train_tpn.py'),
        '--model-checkpoint', str(checkpoint),
        '--train-path', str(split_dir / 'train.txt'),
        '--output-dir', str(output_dir / 'training'),
        '--seeds', str(seed), '--device', args.device, '--species', 'ECO',
    ]
    for name in ('tpn_lr', 'tpn_wd',
                 'tpn_batch_size', 'tpn_epochs', 'tpn_patience', 'tpn_loss',
                 'tpn_early_stop_metric'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    if args.overwrite:
        command.append('--overwrite')
    if args.verbose:
        command.append('--verbose')
    return command


def train_and_evaluate_split(args, variant, seed, checkpoint, split_dir,
                             output_dir, command):
    """Train TPNs, evaluate the two test sets, and save one split's metrics."""
    result_path = output_dir / 'metrics.json'
    if result_path.is_file() and not args.overwrite:
        print(f'[reuse] {variant} seed={seed} split={split_dir.name}', flush=True)
        return json.loads(result_path.read_text())

    training_dir = output_dir / 'training' / f'seed_{seed}'
    model_path = training_dir / 'model.pt'
    if model_path.is_file() and not args.overwrite:
        print(f'[reuse training] {model_path}', flush=True)
    else:
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    configure_logging(output_dir / 'run.log', verbose=args.verbose)
    model = DeepVCFModel.load_model(model_path, device=args.device)
    # load_model restores the saved config, including its original device.
    model.config['device'] = args.device
    filters = load_ranking_filter_positives(args.ranking_filter_path, model.kg_metadata)
    head = CFDataProcessor(split_dir / 'head_corrupt_test.txt', training_dir, split='test').process()
    tail = CFDataProcessor(split_dir / 'tail_corrupt_test.txt', training_dir, split='test').process()
    head_binary = model.evaluate_deepvcf(
        species='ECO', test_data=head, labels=head.test_label, test_type='binary',
    )
    head_ranking = model.evaluate_deepvcf(
        species='ECO', test_data=head, labels=head.test_label, test_type='ranking',
        filter_pos_set=filters,
    )
    tail_binary = model.evaluate_deepvcf(
        species='ECO', test_data=tail, labels=tail.test_label, test_type='binary',
    )
    result = {
        'variant': variant, 'seed': seed, 'split': split_dir.name,
        'split_group': get_split_group(split_dir.name),
        'checkpoint': str(checkpoint), 'model_checkpoint': str(model_path),
        'ranking_filter_path': str(args.ranking_filter_path),
        'config': model.config,
        'head_corrupt': extract_metric_summary(head_binary, head_ranking),
        'tail_corrupt': extract_metric_summary(tail_binary),
    }
    write_json(result_path, result)
    return result


def main(argv=None):
    args = parse_args(argv)
    for seed in args.seeds:
        for variant in args.variants:
            checkpoint = (Path(args.checkpoint.format(seed=seed)) if args.checkpoint
                          else get_drn_checkpoint(variant, seed)).resolve()
            variant_dir = args.output_root / get_experiment_group(variant) / variant / f'seed_{seed}'
            split_dirs = discover_splits(variant, args.internal_root, args.splits)
            aggregate_path = variant_dir / 'benchmark.json'
            results = []
            for split_dir in split_dirs:
                output_dir = variant_dir / 'splits' / split_dir.name
                command = training_command(args, seed, checkpoint, split_dir, output_dir)
                if args.dry_run:
                    print(f'[plan] {variant} seed={seed} split={split_dir.name}')
                    print(shlex.join(command))
                    print(f'  metrics: {output_dir / "metrics.json"}')
                    continue
                results.append(train_and_evaluate_split(
                    args, variant, seed, checkpoint, split_dir, output_dir, command,
                ))
            if args.dry_run:
                continue
            write_json(aggregate_path, {
                'variant': variant, 'seed': seed, 'checkpoint': checkpoint,
                'ranking_filter_path': args.ranking_filter_path,
                'splits': results,
            })
            write_benchmark_tsv(variant_dir / 'benchmark.tsv', results)
            print(f'[complete] {variant} seed={seed}: {variant_dir}', flush=True)


if __name__ == '__main__':
    main()
