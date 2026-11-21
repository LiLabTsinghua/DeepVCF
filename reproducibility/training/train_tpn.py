#!/usr/bin/env python
"""Load a complete DeepVCF model and train its TPNs with frozen DRNs.

Run from the repository root in deepvcf, outside the sandbox on a GPU:
    python reproducibility/training/train_tpn.py \
        --species ECO \
        --model-checkpoint 'results/drn/ECO/enhanced/seed_{seed}/drn_trained_model.pt' \
        --train-path data/processed/cf/train/train.txt \
        --output-dir results/training/ECO --seeds 1 2 3 4 5

--model-checkpoint is required. Use {seed} for seed-matched checkpoints, or
a literal path with one --seeds value. DRN parameters remain fixed; only TPN
is trained. Network architecture, dropout, and fold count are restored from
the checkpoint. Configure TPN optimization and early stopping with --tpn-*.

--mode selects default (CF only), pretrain_only (mechanistic only), mp
(mechanistic then CF), or joint_training (CF plus mechanistic data in each
training fold). Supply --pretrain-path for the latter three modes.
pretrain_only uses mechanistic labels supplied by --pretrain-path.
--filter-train-missing filters CF rows outside KG coverage. CF indexing uses
the mappings stored in the model checkpoint.

--always-train-path supplies support rows included in every CF training fold.
Validation remains on --train-path. Support/main triples must be disjoint;
--drop-always-train-overlap explicitly drops overlapping support triples.
These support options work with default and mp modes.

Output per seed: model.pt (DRN and trained TPN), index_mappings/,
run.log, and summary.json. Mechanistic pretraining also writes
pretrained_model.pt before any CF fine-tuning. All snapshots contain the
complete model and use DeepVCFModel.load_model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import train_drn
else:
    import train_drn


from cf_data import CFDataProcessor, CFTrainDataProcessor
from model import DeepVCFModel



def create_parser():
    """Define options for TPN training from a complete model checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-checkpoint', required=True,
                        help='Complete model path, optionally containing {seed}.')
    parser.add_argument('--species', required=True, help='Species used for candidate pools.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mode', choices=('default', 'pretrain_only', 'mp', 'joint_training'), default='default')
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
    for seed in args.seeds:
        source = Path(args.model_checkpoint.format(seed=seed)).resolve()
        run_dir = args.output_dir.resolve() / f'seed_{seed}'
        checkpoint = run_dir / 'model.pt'
        if args.dry_run:
            print(f'{source} -> {checkpoint} (mode={args.mode})')
            continue
        if source in (checkpoint, run_dir / 'pretrained_model.pt'):
            raise ValueError('Use an output path separate from the input checkpoint.')
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(f'{checkpoint} already exists; use --overwrite.')
        run_dir.mkdir(parents=True, exist_ok=True)
        train_drn.configure_logging(run_dir / 'run.log', args.verbose)
        train_drn.set_seeds(seed)

        # Restore the complete model with its trained DRNs and index mappings.
        model = DeepVCFModel.load_model(source, device=args.device)
        model.set_species(args.species)
        config = model.config
        config.update({
            'device': args.device, 'seed': seed,
            'tpn_lr': args.tpn_lr, 'tpn_wd': args.tpn_wd,
            'tpn_batch_size': args.tpn_batch_size, 'tpn_num_epochs': args.tpn_epochs,
            'tpn_patience': args.tpn_patience, 'tpn_loss': args.tpn_loss,
            'tpn_early_stop_metric': args.tpn_early_stop_metric,
        })
        mapping_dir = run_dir / 'index_mappings'
        mapping_dir.mkdir(parents=True, exist_ok=True)
        for name in ('node_to_id', 'relation_to_id'):
            np.save(mapping_dir / f'{name}.npy', model.kg_metadata[name])

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
