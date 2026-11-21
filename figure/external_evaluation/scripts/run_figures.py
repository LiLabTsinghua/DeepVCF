"""Reproduce external-evaluation figures by analysis.

Run from the repository root in deepvcf:
    python figure/external_evaluation/scripts/run_figures.py
    python figure/external_evaluation/scripts/run_figures.py --analysis ablation tyrr
    python figure/external_evaluation/scripts/run_figures.py --dry-run

The default runs all six analyses. Native-candidate ranking uses the saved
seed_rank_source_data.csv in outputs/native_candidates/. To recompute those
scores, pass --recompute-native-scores --device cuda:0 and run outside the
sandbox on a GPU. No model training is performed.
Each plotting script describes its inputs, outputs, and statistical conventions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ANALYSES = {
    'mechanistic_comparison': (
        'plot_integrated_external_benchmark.py',
        'plot_all_methods_overview.py',
        'plot_target_coverage.py',
    ),
    'ablation': ('plot_embedding_ablation.py',),
    'training_strategy': ('plot_training_strategy_ablation.py',),
    'target_distance': ('plot_hit30_graph_distance.py',),
    'tyrr': ('plot_tyrr_3dhsk_network.py', 'plot_tyrr_rank_across_seeds.py'),
    'native_candidates': (
        'plot_native_candidate_hits.py',
        'plot_native_candidate_count_distribution.py',
        'plot_native_target_counts.py',
        'plot_native_target_coverage.py',
    ),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', nargs='+', choices=tuple(ANALYSES),
                        default=list(ANALYSES))
    parser.add_argument('--recompute-native-scores', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(args.analysis) != len(set(args.analysis)):
        parser.error('--analysis must not contain duplicates')
    for analysis in args.analysis:
        for filename in ANALYSES[analysis]:
            command = [sys.executable, '-B', str(SCRIPT_DIR / filename)]
            if filename == 'plot_native_candidate_hits.py':
                command.extend(['--device', args.device] if args.recompute_native_scores
                               else ['--plot-only'])
            print(f'[{analysis}] {shlex.join(command)}', flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == '__main__':
    main()
