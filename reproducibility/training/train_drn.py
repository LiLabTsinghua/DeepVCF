#!/usr/bin/env python
"""Construct a complete DeepVCF model and train its DRNs from knowledge graphs.

Run from the repository root in deepvcf, outside the sandbox on a GPU:
    python reproducibility/training/train_drn.py \
        --species ECO \
        --kg ECO=data/processed/kg/ECO/kg_enhanced.txt \
        --output-dir results/drn/ECO/enhanced --seeds 1 2 3 4 5

--kg accepts one or more SPECIES=PATH inputs. Equal non-species-scoped entity
IDs are shared when multiple graphs are supplied. Graph construction is a
separate step; supply the prepared KG for each experiment.

Outputs per seed: drn_trained_model.pt (complete model after DRN training), index_mappings/,
summary.json, and run.log.
TPNs retain their initialized weights in the saved model.
The output directory is chosen explicitly so existing downstream checkpoint
paths can be retained. --epochs 0 saves initialized DRNs without training.
PPI relations are treated as symmetric when present in the mapping graph;
--no-inverse-relations overrides that rule (an empty list adds all inverses).
--dry-run prints the configuration and output path without training.
Existing checkpoints/summaries require --overwrite to be replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'code'))
from knowledge import KGProcessor  # noqa: E402
from model import DeepVCFModel  # noqa: E402
from utils import set_seeds  # noqa: E402

LOGGER = logging.getLogger(__name__)


def create_parser():
    """Define KG and DRN training options for this entry point."""
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
    return parser


def parse_args(argv=None):
    return create_parser().parse_args(argv)


def graph_paths(entries):
    """Convert SPECIES=PATH arguments to the KG path dictionary."""
    return {
        species: str(Path(path).resolve())
        for species, path in (entry.split('=', 1) for entry in entries)
    }


def build_config(args, kg, seed):
    """Collect DRN settings and the graph's symmetric relations."""
    excluded = args.no_inverse_relations
    if excluded is None:
        relations = set()
        for path in kg.values():
            with Path(path).open(newline='', encoding='utf-8') as handle:
                relations.update(row[1] for row in csv.reader(handle, delimiter='\t'))
        excluded = {'PPI'} & relations
    return {
        'device': args.device, 'seed': seed, 'kg_path': kg,
        'no_inverse_relations': sorted(excluded),
        'ensemble': args.k > 1, 'k': args.k, 'hidden_dim': args.hidden_dim,
        # Initialize each fold with a DRN and TPN.
        'hidden_channels': 300, 'dropout': 0.1,
        'drn_lr': args.drn_lr, 'drn_wd': args.drn_wd,
        'drn_batch_size': args.drn_batch_size, 'drn_num_epochs': args.drn_epochs,
        'drn_eval_interval': args.drn_eval_interval, 'drn_patience': args.drn_patience,
        'drn_neg_sampling': args.drn_neg_sampling, 'drn_num_neg': args.drn_num_neg,
        'drn_adv_temperature': args.drn_adv_temperature, 'drn_margin': args.drn_margin,
    }


def build_model_and_splits(config, index_root, species):
    processor = KGProcessor(
        kg_path=config['kg_path'],
        index_root=index_root, no_inverse_relations=config['no_inverse_relations'],
        seed=config['seed'],
    )
    if config['drn_num_epochs']:
        graph, kg_metadata, splits = processor.process(num_drn_folds=config['k'])
    else:
        graph, kg_metadata = processor.process()
        splits = None
    model = DeepVCFModel(
        config=config, num_nodes=graph.num_nodes, num_relations=graph.num_edge_type,
        task_rel=graph.task_rel, kg_metadata=kg_metadata, species=species,
    ).to(config['device'])
    return model, graph, splits


def configure_logging(path, verbose):
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[console, logging.FileHandler(path, mode='w')], force=True,
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    kg = graph_paths(args.kg)
    for seed in args.seeds:
        config = build_config(args, kg, seed)
        run_dir = args.output_dir.resolve() / f'seed_{seed}'
        checkpoint = run_dir / 'drn_trained_model.pt'
        if args.dry_run:
            print(json.dumps({'config': config, 'checkpoint': str(checkpoint)}, indent=2))
            continue
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(f'{checkpoint} already exists; use --overwrite.')
        run_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(run_dir / 'run.log', args.verbose)
        set_seeds(seed)

        # Construct the complete model and train only its DRNs.
        model, _, splits = build_model_and_splits(config, run_dir, args.species)
        if splits is not None:
            model.train_drn(splits)
        model.save_model(checkpoint)
        (run_dir / 'summary.json').write_text(json.dumps({
            'seed': seed, 'species': args.species, 'config': config,
            'checkpoint': str(checkpoint),
        }, indent=2) + '\n')
        print(f'Saved {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
