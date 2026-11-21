#!/usr/bin/env python
r"""Train and evaluate ECO-to-target zero-shot and few-shot DeepVCF.

Run from the repository root in deepvcf, on a GPU outside the sandbox:
    python reproducibility/cross_species_transfer/evaluate_deepvcf.py \
        --mode zero_shot --seeds 1 2 3 4 5 --device cuda:0
    python reproducibility/cross_species_transfer/evaluate_deepvcf.py \
        --mode few_shot --kg-groups ECO_SCE --shots 10 --seeds 1 --dry-run
    python reproducibility/cross_species_transfer/evaluate_deepvcf.py \
        --mode zero_shot --evaluation eco_external --stage evaluate \
        --kg-groups ECO_SCE ECO_CGL ECO_SCE_CGL

Each run calls training/train_deepvcf.py with the ECO and target KGs to train
a complete model. TPN uses --species ECO. Few-shot adds target support through
--always-train-path; CF validation remains ECO-only. Target
evaluation switches candidate species to the target species and uses its own
positive dataset for filtering. ECO external evaluation uses the four explicit
ECO positive datasets and is available for zero-shot models.

--evaluation selects target (default), eco_external, or both. --stage selects
both (default), train, or evaluate. --dry-run prints paths without training or
prediction. --overwrite replaces selected artifacts. Models retain their paths
under results/cross_species_transfer/{zero_shot,few_shot}/; target and ECO
external metrics and rank tables retain their existing result locations.
"""

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproducibility.cross_species_transfer.common import (
    DEFAULT_KG_LEVEL, DEFAULT_PAIRS, DEFAULT_SEEDS, KG_ROOT, ECO_TRAIN_PATH,
    KG_GROUP_SPECIES, RESULTS_ROOT, TARGET_DATASET_ROOT, TARGET_TEST_ROOT,
    discover_test_sets, get_few_shot_run_dir,
    get_target_species, get_zero_shot_evaluation_dir, get_zero_shot_run_dir,
    load_target_filter_set,
)
from reproducibility.external_evaluation.common import (
    EXTERNAL_RANKING_FILTER_PATHS, EXTERNAL_TEST_ROOT,
    discover_head_corrupt_test_sets,
    load_external_filter_positive_set, build_tpn_rank_rows, compact_deepvcf_result,
    configure_logging, write_json,
)
from cf_data import CFDataProcessor
from model import DeepVCFModel
from rank_records import write_rank_rows

FEW_SHOT_PAIRS = ('ECO_SCE', 'ECO_CGL')
DEFAULT_SHOTS = (10, 20, 30, 40, 50)
ECO_EXTERNAL_GROUPS = ('ECO_SCE', 'ECO_CGL', 'ECO_SCE_CGL')
ECO_EXPERIMENT = 'eco_external_without_target_cf'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('zero_shot', 'few_shot'), required=True)
    parser.add_argument('--evaluation', choices=('target', 'eco_external', 'both'), default='target')
    parser.add_argument('--stage', choices=('both', 'train', 'evaluate'), default='both')
    parser.add_argument('--kg-groups', nargs='+', choices=tuple(KG_GROUP_SPECIES))
    parser.add_argument('--shots', nargs='+', type=int, default=list(DEFAULT_SHOTS))
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--kg-level', choices=('basic', 'enhanced'), default=DEFAULT_KG_LEVEL)
    parser.add_argument('--train-path', type=Path, default=ECO_TRAIN_PATH)
    parser.add_argument('--shot-root', type=Path, default=TARGET_TEST_ROOT)
    parser.add_argument('--test-root', type=Path, default=TARGET_TEST_ROOT)
    parser.add_argument('--target-dataset-root', type=Path, default=TARGET_DATASET_ROOT)
    parser.add_argument('--external-root', type=Path, default=EXTERNAL_TEST_ROOT)
    parser.add_argument('--ranking-filter-paths', nargs='+', type=Path,
                        default=list(EXTERNAL_RANKING_FILTER_PATHS))
    parser.add_argument('--kg-root', type=Path, default=KG_ROOT)
    parser.add_argument('--output-root', '--model-root', type=Path, default=RESULTS_ROOT)
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
    parser.add_argument('--tpn-early-stop-metric',
                        choices=('mean_group_aupr', 'unfiltered_gene_mrr'), default='unfiltered_gene_mrr')
    parser.add_argument('--ks', nargs='+', type=int, default=[10, 20, 30])
    parser.add_argument('--no-target-filter', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'few_shot' and args.evaluation != 'target':
        parser.error('ECO external evaluation uses zero_shot models')
    args.kg_groups = args.kg_groups or list(
        FEW_SHOT_PAIRS if args.mode == 'few_shot' else
        ECO_EXTERNAL_GROUPS if args.evaluation == 'eco_external' else DEFAULT_PAIRS)
    if args.mode == 'few_shot' and set(args.kg_groups) - set(FEW_SHOT_PAIRS):
        parser.error('few_shot supports ECO_SCE and ECO_CGL')
    for name in ('train_path', 'shot_root', 'test_root', 'target_dataset_root',
                 'external_root', 'kg_root', 'output_root'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.ranking_filter_paths = [path.resolve() for path in args.ranking_filter_paths]
    return args


def run_dir(args, group, seed, shot):
    return (get_zero_shot_run_dir(args.output_root, group, seed) if shot is None
            else get_few_shot_run_dir(args.output_root, group, shot, seed))


def support_path(args, group, shot):
    return (args.shot_root / get_target_species(group) / f'{shot}_shot.txt'
            if shot is not None else None)


def training_command(args, group, seed, shot):
    destination = run_dir(args, group, seed, shot)
    command = [
        sys.executable, str(REPO_ROOT / 'reproducibility/training/train_deepvcf.py'),
        '--kg', *(f'{species}={args.kg_root / species / f"kg_{args.kg_level}.txt"}'
                  for species in KG_GROUP_SPECIES[group]),
        '--train-path', str(args.train_path), '--species', 'ECO',
        '--output-dir', str(destination.parent), '--seeds', str(seed),
        '--device', args.device,
    ]
    support = support_path(args, group, shot)
    if support is not None:
        command.extend(['--always-train-path', str(support)])
    if args.no_inverse_relations is not None:
        command.extend(['--no-inverse-relations', *args.no_inverse_relations])
    for name in ('k', 'hidden_dim', 'hidden_channels', 'dropout',
                 'drn_lr', 'drn_wd', 'drn_batch_size', 'drn_epochs',
                 'drn_eval_interval', 'drn_patience', 'drn_num_neg',
                 'drn_adv_temperature', 'drn_margin', 'drn_neg_sampling',
                 'tpn_lr', 'tpn_wd', 'tpn_batch_size',
                 'tpn_epochs', 'tpn_patience', 'tpn_early_stop_metric'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    if args.overwrite:
        command.append('--overwrite')
    if args.verbose:
        command.append('--verbose')
    return command


def train(args, group, seed, shot):
    command = training_command(args, group, seed, shot)
    model_path = run_dir(args, group, seed, shot) / 'model.pt'
    if model_path.is_file() and not args.overwrite:
        print(f'[reuse training] {model_path}', flush=True)
    else:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return model_path


def evaluate(args, group, seed, shot, target, *, eco_external=False):
    model_path = run_dir(args, group, seed, shot) / 'model.pt'
    if eco_external:
        destination = args.output_root / ECO_EXPERIMENT / group / f'seed_{seed}'
        specs = discover_head_corrupt_test_sets(args.external_root)
        filter_paths = args.ranking_filter_paths
        experiment = ECO_EXPERIMENT
    else:
        destination = (get_zero_shot_evaluation_dir(args.output_root, group, seed, target)
                       if shot is None else run_dir(args, group, seed, shot))
        specs = discover_test_sets(target, args.test_root)
        filter_paths = ([] if args.no_target_filter else
                        [args.target_dataset_root / f'{target}_dataset.txt'])
        experiment = 'zero_shot_evaluation' if shot is None else 'few_shot_evaluation'
    result_path, rank_path = destination / 'metrics.json', destination / 'gene_corruption_ranks.tsv'
    metadata = {
        'experiment': experiment, 'source_species': 'ECO', 'target_species': target,
        'seed': seed, 'model_checkpoint': model_path,
        'candidate_species': target, 'ranking_ks': args.ks,
    }
    if shot is None or eco_external:
        metadata.update(kg_group=group, kg_species=KG_GROUP_SPECIES[group])
    else:
        metadata.update(pair=group, shot=shot)
    if result_path.is_file() and not args.overwrite:
        print(f'[reuse] {group} seed={seed} shot={shot} target={target}', flush=True)
        return
    configure_logging(destination / 'evaluate.log', args.verbose)
    model = DeepVCFModel.load_model(model_path, device=args.device)
    model.config['device'] = args.device
    model.eval()
    if eco_external:
        positive_filter, _ = load_external_filter_positive_set(
            filter_paths, kg_metadata=model.kg_metadata)
    elif filter_paths:
        positive_filter = load_target_filter_set(filter_paths[0], model.kg_metadata)
    else:
        positive_filter = set()
    metrics, rank_rows = {}, []
    index_root = model_path.parent
    for spec in specs:
        data = CFDataProcessor(spec['path'], index_root, split='test').process()
        binary_result = model.evaluate_deepvcf(
            species=target, test_data=data,
            labels=data.test_label, test_type='binary',
        )
        ranking_result = None
        if spec['include_ranking']:
            ranking_result = model.evaluate_deepvcf(
                species=target, test_data=data,
                labels=data.test_label, test_type='ranking',
                filter_pos_set=positive_filter, ks=tuple(args.ks),
                return_rank_records=True,
            )
            rank_rows.extend(build_tpn_rank_rows(
                ranking_result, spec, seed, model.kg_metadata,
            ))
        metrics[spec['name']] = compact_deepvcf_result(binary_result, ranking_result)
    write_rank_rows(rank_path, rank_rows)
    write_json(result_path, {
        **metadata, 'test_sets': metrics, 'ranking_filter_paths': filter_paths,
        'num_ranking_filter_positives': len(positive_filter),
    })
    print(f'[evaluated] {result_path}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    for group in args.kg_groups:
        targets = [species for species in KG_GROUP_SPECIES[group] if species != 'ECO']
        for shot in ([None] if args.mode == 'zero_shot' else args.shots):
            for seed in args.seeds:
                if args.dry_run:
                    print(f'[plan] {args.mode} {group} shot={shot} seed={seed}')
                    print(f'  model: {run_dir(args, group, seed, shot) / "model.pt"}')
                    print(f'  ECO train: {args.train_path}')
                    if shot is not None:
                        print(f'  support: {support_path(args, group, shot)}')
                    if args.stage != 'train':
                        if args.evaluation != 'eco_external':
                            for target in targets:
                                for spec in discover_test_sets(target, args.test_root):
                                    print(f"  test ({target}): {spec['path']}")
                        if args.evaluation != 'target':
                            for spec in discover_head_corrupt_test_sets(args.external_root):
                                print(f"  ECO external: {spec['path']}")
                    continue

                # Train with ECO data; few-shot support is added only to training.
                if args.stage != 'evaluate':
                    train(args, group, seed, shot)
                if args.stage != 'train':
                    if args.evaluation != 'eco_external':
                        for target in targets:
                            evaluate(args, group, seed, shot, target)
                    if args.evaluation != 'target':
                        evaluate(args, group, seed, shot, 'ECO', eco_external=True)


if __name__ == '__main__':
    main()
