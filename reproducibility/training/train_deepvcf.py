#!/usr/bin/env python
"""Train DeepVCF from knowledge graphs and labeled cell-factory data.

Run in deepvcf, outside the sandbox on a GPU, from the repository root:
    python reproducibility/training/train_deepvcf.py \
        --species ECO \
        --kg ECO=data/processed/kg/ECO/kg_enhanced.txt \
        --train-path data/processed/cf/train/train.txt \
        --output-dir results/training/ECO --seeds 1

Train DRN then TPN. --epochs 0 saves initialized DRNs and trains their TPNs.
Use train_tpn.py to train TPNs from existing DRN checkpoints. Both entry points
support --mode default, pretrain_only, mp, and joint_training:
  default: CF training only (the default).
  pretrain_only: mechanistic pretraining only.
  mp: mechanistic pretraining followed by CF fine-tuning.
  joint_training: CF folds with mechanistic rows added to every training fold.

--pretrain-path supplies the mechanistic data for the latter three modes.
Pretraining defaults to 50 epochs. pretrain_only uses the mechanistic
labels supplied by --pretrain-path.

--train-path accepts one or more four-column, tab-separated CF files, without
headers: Protein, task relation, cytoplasmic Metabolite, binary label.
CF folds provide validation for early stopping: k-fold when k > 1, otherwise
an 80/20 split. --species explicitly selects the candidate-pool species for
training and validation and is required for every run.

Each seed saves model.pt, index_mappings/, run.log, and summary.json. Pretraining also saves pretrained_model.pt. All snapshots
contain the complete model. Evaluation remains in the experiment directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from . import train_drn
else:
    import train_drn

from cf_data import CFDataProcessor, CFTrainDataProcessor


MODES = ('default', 'pretrain_only', 'mp', 'joint_training')

def create_parser():
    """Define KG, DRN, and TPN options for full DeepVCF training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kg', nargs='+', required=True, metavar='SPECIES=PATH')
    parser.add_argument('--species', required=True, help='Species used for candidate pools.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--drn-lr', type=float, default=1e-3)
    parser.add_argument('--drn-wd', type=float, default=1e-5)
    parser.add_argument('--drn-batch-size', type=int, default=1024)
    parser.add_argument('--drn-epochs', '--epochs', type=int, default=200)
    parser.add_argument('--drn-eval-interval', type=int, default=1)
    parser.add_argument('--drn-patience', type=int, default=20)
    parser.add_argument('--drn-num-neg', type=int, default=100)
    parser.add_argument('--drn-adv-temperature', type=float, default=1.0)
    parser.add_argument('--drn-margin', type=float, default=1.0)
    parser.add_argument('--drn-neg-sampling', choices=('adversarial', 'uniform'),
                        default='adversarial')
    parser.add_argument('--no-inverse-relations', nargs='*', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mode', choices=MODES, default='default')
    parser.add_argument('--train-path', nargs='+', type=Path)
    parser.add_argument('--always-train-path', nargs='+', type=Path,
                        help='Support data added only to the training side of every CF fold.')
    parser.add_argument('--drop-always-train-overlap', action='store_true',
                        help='Drop support triples overlapping --train-path; default: reject overlap.')
    parser.add_argument('--pretrain-path', type=Path,
                        help='Labeled mechanistic data for pretraining or joint training.')
    parser.add_argument('--pretrain-epochs', type=int, default=50)
    parser.add_argument('--pretrain-batch-size', type=int, default=1024)
    parser.add_argument('--reset-cls-after-pretrain', action='store_true')
    parser.add_argument('--filter-train-missing', action='store_true',
                        help='Filter CF rows outside KG coverage.')
    parser.add_argument('--hidden-channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--tpn-lr', type=float, default=5e-5)
    parser.add_argument('--tpn-wd', type=float, default=1e-5)
    parser.add_argument('--tpn-batch-size', type=int, default=128)
    parser.add_argument('--tpn-epochs', type=int, default=500)
    parser.add_argument('--tpn-patience', type=int, default=30)
    parser.add_argument('--tpn-loss', choices=('CE', 'BCE'), default='CE')
    parser.add_argument('--tpn-early-stop-metric',
                        choices=('unfiltered_gene_mrr', 'mean_group_aupr'),
                        default='unfiltered_gene_mrr')
    return parser


def parse_args(argv=None):
    return create_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kg = train_drn.graph_paths(args.kg)
    for seed in args.seeds:
        config = train_drn.build_config(args, kg, seed)
        config.update(hidden_channels=args.hidden_channels, dropout=args.dropout)
        config.update({
            'device': args.device, 'seed': seed,
            'tpn_lr': args.tpn_lr, 'tpn_wd': args.tpn_wd,
            'tpn_batch_size': args.tpn_batch_size, 'tpn_num_epochs': args.tpn_epochs,
            'tpn_patience': args.tpn_patience, 'tpn_loss': args.tpn_loss,
            'tpn_early_stop_metric': args.tpn_early_stop_metric,
        })
        run_dir = args.output_dir.resolve() / f'seed_{seed}'
        checkpoint = run_dir / 'model.pt'
        if args.dry_run:
            print(json.dumps({'config': config, 'mode': args.mode,
                              'checkpoint': str(checkpoint)}, indent=2))
            continue
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(f'Training output already exists in {run_dir}; use --overwrite.')
        run_dir.mkdir(parents=True, exist_ok=True)
        train_drn.configure_logging(run_dir / 'run.log', args.verbose)
        train_drn.set_seeds(seed)

        # Build a new complete model, train its DRNs, then train its TPNs.
        model, _, splits = train_drn.build_model_and_splits(config, run_dir, args.species)
        if splits is not None:
            model.train_drn(splits)

        # Pretraining uses the same TPN training method with fixed epochs.
        if args.mode in ('pretrain_only', 'mp'):
            data = CFDataProcessor(args.pretrain_path, run_dir, split='train').process()
            model.train_tpn(
                [data] * len(model.folds), fixed_epochs=args.pretrain_epochs,
                batch_size=args.pretrain_batch_size,
            )
            model.save_model(run_dir / 'pretrained_model.pt')

        if args.mode != 'pretrain_only':
            # Joint training adds mechanistic rows only to each training fold.
            support = (args.pretrain_path if args.mode == 'joint_training'
                       else args.always_train_path)
            folds, _ = CFTrainDataProcessor(
                train_path=args.train_path, index_root=run_dir,
                ensemble=config['ensemble'], k=config['k'], seed=seed,
                filter_train_missing=args.filter_train_missing,
                always_train_path=support,
                drop_always_train_overlap=(args.mode == 'joint_training'
                                           or args.drop_always_train_overlap),
            ).process()
            train_drn.set_seeds(seed)
            if args.reset_cls_after_pretrain:
                model.reset_tpn_cls_layer()
                train_drn.set_seeds(seed)
            model.train_tpn(folds)

        model.save_model(checkpoint)
        (run_dir / 'summary.json').write_text(json.dumps({
            'seed': seed, 'mode': args.mode, 'species': args.species,
            'config': config, 'checkpoint': str(checkpoint),
        }, indent=2) + '\n')
        print(f'Saved {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
