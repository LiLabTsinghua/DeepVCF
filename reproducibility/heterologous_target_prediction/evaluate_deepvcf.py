#!/usr/bin/env python
r"""Train complete DeepVCF models and predict heterologous enzyme targets.

Run from the repository root in deepvcf, on a GPU outside the sandbox:
    python reproducibility/heterologous_target_prediction/evaluate_deepvcf.py \
        --variants basic_extended --seeds 1 2 3 4 5 6 7 8 9 10 \
        --query-metabolites Metabolite:ptrc_c --device cuda:0

Each variant/seed calls training/train_deepvcf.py to train DRNs and TPNs
from the extended ECO KG and CF training data, without MP. basic_extended
filters CF rows outside its KG; enhanced_extended uses all CF training rows.

Outputs under results/heterologous_target_prediction/{variant}/seed_{seed}/:
model.pt, index_mappings/, run.log, summary.json, prediction_summary.json,
and predictions/{metabolite}.tsv with per-fold probabilities and ensemble
mean/variance. Candidates are ECO heterologous proteins; both model task
modifications are scored. The default query is Metabolite:ptrc_c.

Existing models and query predictions are reused. --overwrite retrains the
selected models and regenerates predictions. --dry-run prints training
commands and prediction paths without loading models or creating files.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproducibility.heterologous_target_prediction.common import (
    DEFAULT_SEEDS, DEFAULT_VARIANTS, KG_ROOT, QUERY_METABOLITES,
    RESULTS_ROOT, TRAIN_PATH, VARIANT_KG_FILES, write_json,
)

sys.path.insert(0, str(REPO_ROOT / 'code'))
from model import DeepVCFModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', '--variant', nargs='+', choices=DEFAULT_VARIANTS,
                        default=list(DEFAULT_VARIANTS))
    parser.add_argument('--seeds', '--seed', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--kg-root', type=Path, default=KG_ROOT)
    parser.add_argument('--train-path', type=Path, default=TRAIN_PATH)
    parser.add_argument('--output-root', type=Path, default=RESULTS_ROOT)
    parser.add_argument('--query-metabolites', nargs='+', default=list(QUERY_METABOLITES))
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
                        choices=('mean_group_aupr', 'unfiltered_gene_mrr'),
                        default='unfiltered_gene_mrr')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    for name in ('kg_root', 'train_path', 'output_root'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def training_command(args, variant, seed):
    """Train a complete model from the selected extended KG and CF data."""
    command = [
        sys.executable, str(REPO_ROOT / 'reproducibility/training/train_deepvcf.py'),
        '--kg', f'ECO={args.kg_root / VARIANT_KG_FILES[variant]}',
        '--train-path', str(args.train_path), '--species', 'ECO',
        '--mode', 'default', '--output-dir', str(args.output_root / variant),
        '--seeds', str(seed), '--device', args.device,
    ]
    if variant == 'basic_extended':
        command.append('--filter-train-missing')
    if args.no_inverse_relations is not None:
        command.extend(['--no-inverse-relations', *args.no_inverse_relations])
    for name in ('k', 'hidden_dim', 'hidden_channels', 'dropout',
                 'drn_lr', 'drn_wd', 'drn_batch_size', 'drn_epochs',
                 'drn_eval_interval', 'drn_patience', 'drn_num_neg',
                 'drn_adv_temperature', 'drn_margin', 'drn_neg_sampling',
                 'tpn_lr', 'tpn_wd', 'tpn_batch_size', 'tpn_epochs',
                 'tpn_patience', 'tpn_loss', 'tpn_early_stop_metric'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    for name in ('overwrite', 'verbose'):
        if getattr(args, name):
            command.append('--' + name)
    return command


def train_and_predict(args, variant, seed, command):
    output_dir = args.output_root / variant / f'seed_{seed}'
    checkpoint = output_dir / 'model.pt'
    prediction_dir = output_dir / 'predictions'

    # Reuse existing query results unless regeneration is requested.
    pending = [metabolite for metabolite in args.query_metabolites
               if args.overwrite or not (prediction_dir / f'{metabolite}.tsv').is_file()]
    if not pending:
        print(f'[reuse] {variant} seed={seed}: {prediction_dir}', flush=True)
        return

    if not checkpoint.is_file() or args.overwrite:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    else:
        print(f'[reuse training] {checkpoint}', flush=True)

    # The model selects the species' heterologous pool and writes all scores.
    model = DeepVCFModel.load_model(checkpoint, device=args.device)
    model.config['device'] = args.device
    model.eval()
    model.genome_scale_prediction(
        species='ECO', query_met=pending,
        gene_group='heterologous', output_dir=prediction_dir,
    )
    write_json(output_dir / 'prediction_summary.json', {
        'variant': variant, 'seed': seed, 'model_checkpoint': str(checkpoint),
        'kg_path': str(args.kg_root / VARIANT_KG_FILES[variant]),
        'train_path': str(args.train_path),
        'species': 'ECO', 'gene_group': 'heterologous',
        'query_metabolites': args.query_metabolites,
        'num_folds': len(model.folds),
    })
    print(f'[done] {variant} seed={seed}: {prediction_dir}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    for variant in args.variants:
        for seed in args.seeds:
            command = training_command(args, variant, seed)
            if args.dry_run:
                output_dir = args.output_root / variant / f'seed_{seed}'
                print(f'[plan] {variant} seed={seed}\n  training: {shlex.join(command)}')
                print(f'  model: {output_dir / "model.pt"}')
                for metabolite in args.query_metabolites:
                    print(f'  prediction: {output_dir / "predictions" / (metabolite + ".tsv")}')
                continue
            train_and_predict(args, variant, seed, command)


if __name__ == '__main__':
    main()
