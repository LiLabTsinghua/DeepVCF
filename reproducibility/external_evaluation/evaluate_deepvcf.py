#!/usr/bin/env python
r"""Train and evaluate DeepVCF on the ECO external head-corruption tests.

Run from the repository root in deepvcf, on a GPU outside the sandbox:
    python reproducibility/external_evaluation/evaluate_deepvcf.py \
        --mode default --seeds 1 2 3 4 5 --device cuda:0
    python reproducibility/external_evaluation/evaluate_deepvcf.py \
        --modes default pretrain_only mp joint_training --seeds 1 --dry-run

Each mode/seed calls training/train_deepvcf.py to train DRNs and TPNs from
the supplied KG and labeled data, then evaluates the external tests. --train-path and --external-root select
CF training and test inputs. --pretrain-path supplies mechanistic data.
--kg accepts SPECIES=PATH inputs and defaults to the enhanced ECO KG.

Outputs under results/external_evaluation/{method}/seed_{seed}/:
metrics.json, gene_corruption_ranks.tsv, run.log, and training/seed_{seed}/
containing model.pt, index_mappings/, and training records. Saved models
and metrics are reused. --overwrite replaces selected outputs; --dry-run
prints commands without loading a model or creating files.
"""

import argparse
from pathlib import Path
import shlex
import sys
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproducibility.external_evaluation.common import (
    DEFAULT_SEEDS, EXTERNAL_RANKING_FILTER_PATHS, EXTERNAL_TEST_ROOT,
    MODE_TO_METHOD, RESULTS_ROOT, TRAIN_PATH, DeepVCFModel, build_tpn_rank_rows,
    compact_deepvcf_result, configure_logging,
    discover_head_corrupt_test_sets, get_model_spec,
    load_external_filter_positive_set,
    write_json, write_rank_rows,
)
from cf_data import CFDataProcessor



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', '--mode', nargs='+', choices=tuple(MODE_TO_METHOD),
                        default=['default'])
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--kg', nargs='+', default=[
        'ECO=' + str(REPO_ROOT / 'data/processed/kg/ECO/kg_enhanced.txt')])
    parser.add_argument('--train-path', type=Path,
                        help='CF data; defaults to the formal ECO train set when used.')
    parser.add_argument('--pretrain-path', type=Path, default=(
        REPO_ROOT / 'data/processed/cf/mechanistic_pretraining/ECO_FSEOF_1_3.txt'))
    parser.add_argument('--pretrain-epochs', type=int, default=50)
    parser.add_argument('--pretrain-batch-size', type=int, default=1024)
    parser.add_argument('--reset-cls-after-pretrain', action='store_true')
    parser.add_argument('--filter-train-missing', action='store_true')
    parser.add_argument('--external-root', type=Path, default=EXTERNAL_TEST_ROOT)
    parser.add_argument('--splits', nargs='+', help='External test names, e.g. gem_gene/head_corrupt.')
    parser.add_argument('--ranking-filter-paths', nargs='+', type=Path,
                        default=list(EXTERNAL_RANKING_FILTER_PATHS))
    parser.add_argument('--output-root', type=Path, default=RESULTS_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--hidden-channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drn-lr', type=float, default=1e-3)
    parser.add_argument('--drn-wd', type=float, default=1e-5)
    parser.add_argument('--drn-batch-size', type=int, default=1024)
    parser.add_argument('--drn-epochs', type=int, default=200)
    parser.add_argument('--drn-eval-interval', type=int, default=1)
    parser.add_argument('--drn-patience', type=int, default=20)
    parser.add_argument('--drn-num-neg', type=int, default=100)
    parser.add_argument('--drn-adv-temperature', type=float, default=1.0)
    parser.add_argument('--drn-margin', type=float, default=1.0)
    parser.add_argument('--drn-neg-sampling', choices=('adversarial', 'uniform'),
                        default='adversarial')
    parser.add_argument('--no-inverse-relations', nargs='*')
    parser.add_argument('--tpn-lr', type=float, default=5e-5)
    parser.add_argument('--tpn-wd', type=float, default=1e-5)
    parser.add_argument('--tpn-batch-size', type=int, default=128)
    parser.add_argument('--tpn-epochs', type=int, default=500)
    parser.add_argument('--tpn-patience', type=int, default=30)
    parser.add_argument('--tpn-loss', choices=('CE', 'BCE'), default='CE')
    parser.add_argument('--tpn-early-stop-metric',
                        choices=('mean_group_aupr', 'unfiltered_gene_mrr'), default='unfiltered_gene_mrr')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    for name in ('external_root', 'output_root', 'pretrain_path'):
        setattr(args, name, getattr(args, name).resolve())
    if args.train_path:
        args.train_path = args.train_path.resolve()
    args.ranking_filter_paths = [path.resolve() for path in args.ranking_filter_paths]
    return args


def training_command(args, mode, seed, output_dir):
    command = [
        sys.executable, str(REPO_ROOT / 'reproducibility/training/train_deepvcf.py'),
        '--kg', *args.kg,
        '--mode', mode, '--output-dir', str(output_dir / 'training'),
        '--species', 'ECO', '--seeds', str(seed), '--device', args.device,
    ]
    if mode != 'pretrain_only':
        command.extend(['--train-path', str(args.train_path or TRAIN_PATH)])
    if mode != 'default':
        command.extend(['--pretrain-path', str(args.pretrain_path)])
    if mode in ('pretrain_only', 'mp'):
        for name in ('pretrain_epochs', 'pretrain_batch_size'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    if args.no_inverse_relations is not None:
        command.extend(['--no-inverse-relations', *args.no_inverse_relations])
    for name in ('k', 'hidden_dim', 'hidden_channels', 'dropout',
                 'drn_lr', 'drn_wd', 'drn_batch_size', 'drn_epochs',
                 'drn_eval_interval', 'drn_patience', 'drn_num_neg',
                 'drn_adv_temperature', 'drn_margin', 'drn_neg_sampling',
                 'tpn_lr', 'tpn_wd',
                 'tpn_batch_size', 'tpn_epochs', 'tpn_patience', 'tpn_loss', 'tpn_early_stop_metric'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    for name in ('reset_cls_after_pretrain', 'filter_train_missing', 'overwrite', 'verbose'):
        if getattr(args, name):
            command.append('--' + name.replace('_', '-'))
    return command


def train_and_evaluate(args, mode, seed, specs, command):
    method = MODE_TO_METHOD[mode]
    output_dir = args.output_root / method / f'seed_{seed}'
    result_path = output_dir / 'metrics.json'
    rank_path = output_dir / 'gene_corruption_ranks.tsv'
    if result_path.is_file() and not args.overwrite:
        print(f'[reuse] {method} seed={seed}', flush=True)
        return
    checkpoint = output_dir / 'training' / f'seed_{seed}' / 'model.pt'
    if not checkpoint.is_file() or args.overwrite:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    metadata = {
        'mode': mode, 'method': method, 'seed': seed,
        'model_checkpoint': str(checkpoint),
        'ranking_filter_paths': args.ranking_filter_paths,
        'external_root': args.external_root,
    }
    configure_logging(output_dir / 'run.log', verbose=args.verbose)
    model = DeepVCFModel.load_model(checkpoint, device=args.device)
    model.config['device'] = args.device
    model.eval()
    filters, counts = load_external_filter_positive_set(
        args.ranking_filter_paths, kg_metadata=model.kg_metadata,
    )
    test_results, rank_rows = {}, []
    index_root = checkpoint.parent
    for spec in specs:
        test_data = CFDataProcessor(spec['path'], index_root, split='test').process()
        binary_result = model.evaluate_deepvcf(
            species='ECO', test_data=test_data,
            labels=test_data.test_label, test_type='binary',
        )
        ranking_result = None
        if spec['include_ranking']:
            ranking_result = model.evaluate_deepvcf(
                species='ECO', test_data=test_data,
                labels=test_data.test_label, test_type='ranking',
                filter_pos_set=filters, return_rank_records=True,
            )
            rank_rows.extend(build_tpn_rank_rows(
                ranking_result, spec, seed, model.kg_metadata,
            ))
        test_results[spec['name']] = compact_deepvcf_result(
            binary_result, ranking_result,
        )
    write_rank_rows(rank_path, rank_rows)
    spec = get_model_spec(method)
    write_json(result_path, {
        **metadata, 'model_family': spec.family, 'model_label': spec.label,
        'config': model.config,
        'ranking_filter_rows_by_file': counts,
        'ranking_filter_base': 'external_positive_datasets',
        'num_ranking_filter_base_positives': len(filters),
        'test_sets': test_results,
    })
    print(f'[done] {method} seed={seed}: {result_path}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    specs = discover_head_corrupt_test_sets(args.external_root)
    if args.splits:
        available = {spec['name']: spec for spec in specs}
        specs = [available[name] for name in args.splits]
    for mode in args.modes:
        for seed in args.seeds:
            output_dir = args.output_root / MODE_TO_METHOD[mode] / f'seed_{seed}'
            command = training_command(args, mode, seed, output_dir)
            if args.dry_run:
                print(f'[plan] {mode} seed={seed}\n  training: {shlex.join(command)}')
                for spec in specs:
                    print(f"  test: {spec['name']} -> {spec['path']}")
                print(f'  output: {output_dir}')
                continue
            train_and_evaluate(args, mode, seed, specs, command)


if __name__ == '__main__':
    main()
