"""Shared data processing and 2 x 7 plotting for internal embedding comparisons.

Each bar summarizes five seeds and five FoldUnits (25 observations), with
sample SD and all observations shown. Basic KG supports eight hold-outs;
the other conditions support all 14. Statistical CSVs retain exploratory
one-sided Welch comparisons without multiple-testing correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/deepvcf-matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy import stats


# =============================================================================
# DeepVCF Internal-Evaluation Figures
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / 'results' / 'internal_evaluation' / 'embedding_comparison'
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs/embedding_comparison'
SEEDS = (1, 2, 3, 4, 5)
VARIANTS = ('random', 'basic', 'enhanced')
VARIANT_LABELS = {
    'untrained': 'Untrained DRN + TPN',
    'random': 'Random DRN',
    'basic': 'Basic KG',
    'enhanced': 'Enhanced KG',
}
VARIANT_COLORS = {
    'untrained': '#D1D5DB',
    'random': '#8B8F94',
    'basic': '#5B7FCA',
    'enhanced': '#B64342',
}

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_style import (  # noqa: E402
    BAR_EDGE_WIDTH,
    EDGE_COLOR,
    ERRORBAR_CAPSIZE,
    ERRORBAR_LINEWIDTH,
    create_small_multiples,
    save_panel,
    scatter_size_for_bar,
    style_axis,
)
METABOLITE_SPLITS = (
    'metabolite_hold_out',
    'amino_acid_hold_out',
    'carbohydrate_hold_out',
    'cofactors_and_vitamins_hold_out',
    'lipid_hold_out',
    'nucleotide_hold_out',
    'secondary_metabolites_hold_out',
)
GENE_SPLITS = (
    'gem_gene_hold_out',
    'gene_type_hold_out',
    'go_metabolism_related_hold_out',
    'go_signal_transduction_hold_out',
    'go_stress_response_hold_out',
    'go_transcriptional_regulation_hold_out',
    'go_dna_replication_repair_hold_out',
)
SPLIT_LABELS = {
    'metabolite_hold_out': 'Metabolites',
    'amino_acid_hold_out': 'Amino acids',
    'carbohydrate_hold_out': 'Carbohydrates',
    'cofactors_and_vitamins_hold_out': 'Cofactors &\nvitamins',
    'lipid_hold_out': 'Lipids',
    'nucleotide_hold_out': 'Nucleotides',
    'secondary_metabolites_hold_out': 'Secondary\nmetabolites',
    'gem_gene_hold_out': 'GEM genes',
    'gene_type_hold_out': 'Gene type',
    'go_metabolism_related_hold_out': 'GO:\nmetabolism-\nrelated',
    'go_signal_transduction_hold_out': 'GO: signal\ntransduction',
    'go_stress_response_hold_out': 'GO: stress\nresponse',
    'go_transcriptional_regulation_hold_out': 'GO:\ntranscriptional\nregulation',
    'go_dna_replication_repair_hold_out': 'GO: DNA\nreplication\n& repair',
}


def parse_args(default_output_dir: Path, argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results-root', type=Path, default=DEFAULT_RESULTS_ROOT,
        help='Root containing <variant>/seed_<seed>/benchmark.json files.',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=default_output_dir,
        help='Destination for the figure and source data.',
    )
    return parser.parse_args(argv)


def apply_publication_style() -> None:
    """Apply the shared publication style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.frameon': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
    })


def load_results(results_root: Path, variants=VARIANTS) -> dict[int, dict[str, dict[str, dict]]]:
    """Load seed-specific benchmark results.

    Args:
        results_root: Root containing seed and variant result directories.

    Returns:
        Results indexed by seed, variant, and split.

    Raises:
        FileNotFoundError: If a required benchmark file is missing.
        ValueError: If a benchmark file contains no split results.
    """
    results = {}
    plotted_splits = set((*METABOLITE_SPLITS, *GENE_SPLITS))
    for seed in SEEDS:
        results[seed] = {}
        for variant in variants:
            path = results_root / variant / f'seed_{seed}' / 'benchmark.json'
            if not path.is_file():
                raise FileNotFoundError(f'Missing benchmark file: {path}')
            with path.open(encoding='utf-8') as handle:
                benchmark = json.load(handle)
            split_results = benchmark.get('splits', [])
            if not split_results:
                raise ValueError(f'No split results found in {path}.')
            by_split = {
                result['split']: result for result in split_results
            }
            expected = plotted_splits if variant != 'basic' else {
                *METABOLITE_SPLITS, 'gem_gene_hold_out',
            }
            if set(by_split) != expected or len(by_split) != len(split_results):
                raise ValueError(
                    f'Unexpected split set in {path}: '
                    f'missing={sorted(expected - set(by_split))}, '
                    f'extra={sorted(set(by_split) - expected)}'
                )
            results[seed][variant] = by_split
    return results


def collect_fold_records(
    results: dict[int, dict[str, dict[str, dict]]],
    corruption: str,
    metric_key: str,
    metric_kind: str = 'aupr',
    variants=VARIANTS,
) -> list[dict]:
    """Collect fold-level metric values without reading ensemble results.

    Args:
        results: Results indexed by seed, variant, and split.
        corruption: Corruption-result key in each split result.
        metric_key: Output record key for the requested metric values.
        metric_kind: Metric stored in each fold result. Supported values are
            ``'aupr'``, ``'auroc'``, and ``'hits30'``.

    Returns:
        Long-form fold-level metric records.

    Raises:
        ValueError: If a result does not contain exactly five folds.
        ValueError: If ``metric_kind`` is unsupported.
    """
    if metric_kind not in {'aupr', 'auroc', 'hits30'}:
        raise ValueError(f'Unsupported internal metric kind: {metric_kind}.')
    records = []
    plotted_splits = set((*METABOLITE_SPLITS, *GENE_SPLITS))
    for seed in SEEDS:
        for variant in variants:
            for split, split_result in results[seed][variant].items():
                if split not in plotted_splits:
                    continue
                folds = split_result[corruption]['folds']
                if len(folds) != 5 or {fold['fold'] for fold in folds} != {1, 2, 3, 4, 5}:
                    raise ValueError(
                        f'Expected five folds for seed {seed}/{variant}/{split}, '
                        f'found {len(folds)}.'
                    )
                for fold_result in folds:
                    if metric_kind in {'aupr', 'auroc'}:
                        value = fold_result['binary']['overall'][metric_kind]
                    else:
                        value = fold_result['target_gene_ranking']['gene_hit@30']
                    value = float(value)
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(
                            f'Invalid {metric_kind} value for '
                            f'{seed}/{variant}/{split}: {value}.'
                        )
                    records.append({
                        'seed': seed,
                        'fold': int(fold_result['fold']),
                        'split': split,
                        'split_label': SPLIT_LABELS[split].replace('\n', ' '),
                        'variant': variant,
                        'variant_label': VARIANT_LABELS[variant],
                        metric_key: value,
                    })
    return records


def values_for(
    records: list[dict], split: str, variant: str, metric_key: str,
) -> np.ndarray:
    """Return available seed-fold values for one group.

    Args:
        records: Long-form fold-level records.
        split: Hold-out split name.
        variant: Model variant name.
        metric_key: Record key containing the requested metric values.

    Returns:
        Metric values ordered by seed and fold.
    """
    return np.asarray([
        row[metric_key] for row in records
        if row['split'] == split and row['variant'] == variant
    ], dtype=float)


def compute_statistics(records: list[dict], metric_key: str, variants=VARIANTS) -> list[dict]:
    """Compare Enhanced KG with available baselines.

    Args:
        records: Long-form fold-level records.
        metric_key: Record key containing the requested metric values.

    Returns:
        One-sided Welch t-test records without multiplicity correction.
    """
    output = []
    for split in (*METABOLITE_SPLITS, *GENE_SPLITS):
        enhanced = values_for(records, split, 'enhanced', metric_key)
        for baseline in (variant for variant in variants if variant != 'enhanced'):
            reference = values_for(records, split, baseline, metric_key)
            if not len(reference):
                continue
            test = stats.ttest_ind(
                enhanced, reference, equal_var=False, alternative='greater',
            )
            output.append({
                'split': split,
                'comparison': f'enhanced_vs_{baseline}',
                'enhanced_mean': float(np.mean(enhanced)),
                'baseline_mean': float(np.mean(reference)),
                'mean_difference': float(
                    np.mean(enhanced) - np.mean(reference)
                ),
                't_statistic': float(test.statistic),
                'p_value': float(test.pvalue),
                'n_enhanced': len(enhanced),
                'n_baseline': len(reference),
                'test': 'one-sided Welch t-test',
                'alternative': 'enhanced > baseline',
                'multiple_testing_correction': 'none',
            })
    return output


def summarize(records: list[dict], metric_key: str, variants=VARIANTS) -> list[dict]:
    """Summarize each available seed-fold distribution.

    Args:
        records: Long-form fold-level records.
        metric_key: Record key containing the requested metric values.

    Returns:
        Mean and sample-standard-deviation records.
    """
    output = []
    for split in (*METABOLITE_SPLITS, *GENE_SPLITS):
        for variant in variants:
            values = values_for(records, split, variant, metric_key)
            if not len(values):
                continue
            output.append({
                'split': split,
                'split_label': SPLIT_LABELS[split].replace('\n', ' '),
                'variant': variant,
                'variant_label': VARIANT_LABELS[variant],
                'n_seed_fold_points': len(values),
                f'mean_{metric_key}': float(np.mean(values)),
                f'sd_{metric_key}': float(np.std(values, ddof=1)),
            })
    return output


def write_csv(path: Path, records: list[dict]) -> None:
    """Write records to CSV.

    Args:
        path: Destination path.
        records: Rows with consistent fields.

    Raises:
        ValueError: If no rows are supplied.
    """
    if not records:
        raise ValueError(f'No records available for {path}.')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def draw_split_panel(
    ax: plt.Axes,
    split: str,
    records: list[dict],
    show_xlabels: bool,
    metric_key: str,
    variants=VARIANTS,
) -> None:
    """Draw bars and all seed-fold points for one split.

    Args:
        ax: Target axes.
        split: Hold-out split name.
        records: Long-form fold-level records.
        show_xlabels: Whether to show condition tick labels.
        metric_key: Record key containing the requested metric values.
    """
    style_axis(ax)
    bar_width = 0.66
    ax.set_xlim(-0.55, len(variants) - 0.45)
    ax.set_ylim(0, 1)
    marker_area = scatter_size_for_bar(
        ax, bar_width, max_size=10, min_size=4,
        diameter_fraction=0.18,
    )
    for position, variant in enumerate(variants):
        values = values_for(records, split, variant, metric_key)
        if len(values) == 0:
            continue
        if len(values) != len(SEEDS) * 5:
            raise ValueError(
                f'Expected 25 fold × seed values for {split}/{variant}, '
                f'found {len(values)}.'
            )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        # Raw observations are validated above.  Symmetric SD intervals can
        # cross a bounded metric's domain near 0 or 1, so only the rendered
        # whisker is clipped; the exported summary retains the unmodified SD.
        bounded_yerr = np.array([
            [min(std, mean)],
            [min(std, 1.0 - mean)],
        ])
        ax.bar(
            position, mean, width=bar_width,
            color=VARIANT_COLORS[variant], alpha=1.0,
            edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=2,
        )
        ax.errorbar(
            position, mean, yerr=bounded_yerr, fmt='none',
            ecolor=EDGE_COLOR, elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE * 0.55,
            capthick=ERRORBAR_LINEWIDTH, zorder=3,
        )
        jitter = np.linspace(-bar_width * 0.28, bar_width * 0.28, len(values))
        ax.scatter(
            position + jitter, values, s=marker_area,
            color=EDGE_COLOR, alpha=1.0,
            edgecolor='none', zorder=4,
        )

    ax.set_title(SPLIT_LABELS[split], pad=3, fontsize=9.0, fontweight='normal')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(
        tuple(variant[0].upper() for variant in variants)
        if show_xlabels else ('',) * len(variants),
        fontsize=8.0,
    )
    ax.set_yticks((0, 0.5, 1.0))
    ax.tick_params(axis='both', pad=2, labelsize=8.0)
    ax.grid(False)


def plot_figure(
    output_stem: Path,
    records: list[dict],
    metric_key: str,
    y_label: str,
    show_text: bool,
    variants=VARIANTS,
) -> None:
    """Plot a two-row internal-evaluation small-multiple figure.

    Args:
        output_stem: Output path without extension.
        records: Long-form fold-level records.
        metric_key: Record key containing the requested metric values.
        y_label: Shared vertical-axis label.
    """
    n_columns = 7
    fig, axes = create_small_multiples(
        'small_multiples_v1', nrows=2, ncols=n_columns,
        sharex=False, sharey=True, panel_width_mm=34.4 if 'untrained' in variants else 26.0,
    )
    # Text-bearing panels need room for the first-row condition ticks and wrapped
    # second-row titles.  For the no-text version, retain the original row
    # spacing while trimming only the outer vertical whitespace.
    if show_text:
        fig.subplots_adjust(hspace=0.42)
    else:
        no_text_height_mm = 78.0
        outer_margin_mm = 3.0
        fig.set_figheight(no_text_height_mm / 25.4)
        fig.subplots_adjust(
            bottom=outer_margin_mm / no_text_height_mm,
            top=1.0 - outer_margin_mm / no_text_height_mm,
            hspace=0.42,
        )
    for column, split in enumerate(METABOLITE_SPLITS):
        draw_split_panel(
            axes[0, column], split, records, show_xlabels=show_text,
            metric_key=metric_key, variants=variants,
        )
    for column, split in enumerate(GENE_SPLITS):
        draw_split_panel(
            axes[1, column], split, records, show_xlabels=show_text,
            metric_key=metric_key, variants=variants,
        )

    for row in range(2):
        for column in range(n_columns):
            ax = axes[row, column]
            if show_text and column:
                ax.tick_params(labelleft=False)
            if not show_text:
                ax.set_title('')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    handles = [
        Patch(
            facecolor=VARIANT_COLORS[variant], edgecolor='#202020',
            linewidth=0.65, label=VARIANT_LABELS[variant],
        )
        for variant in variants
    ]
    handles.append(Line2D(
        [], [], linestyle='none', marker='o', markersize=3,
        markerfacecolor='#202020', markeredgecolor='none',
        label='Fold × seed (n = 25)',
    ))
    if show_text:
        fig.legend(
            handles=handles, loc='upper center', bbox_to_anchor=(0.54, 0.987),
            ncol=len(variants) + 1, borderaxespad=0, handletextpad=0.45,
            columnspacing=1.0, fontsize=9.0,
        )
        fig.text(
            0.018, 0.5, y_label, rotation=90,
            ha='center', va='center', fontsize=13,
        )
    if not show_text:
        output_stem = Path(f'{output_stem}_no_text')
    return save_panel(fig, output_stem)


def main(argv=None, *, include_untrained=False) -> None:
    """Generate the multi-seed fold-level internal-evaluation figure."""
    variants = ('untrained', *VARIANTS) if include_untrained else VARIANTS
    default_output = (DEFAULT_OUTPUT_DIR.with_name('embedding_comparison_with_untrained')
                      if include_untrained else DEFAULT_OUTPUT_DIR)
    args = parse_args(default_output, argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_publication_style()
    results = load_results(args.results_root.resolve(), variants)
    figure_specs = (
        (
            'head_corrupt', 'head_aupr', 'Gene-corruption AUPR',
            'internal_head_aupr_fold_seed_comparison', 'aupr',
        ),
        (
            'head_corrupt', 'head_auroc', 'Gene-corruption AUROC',
            'internal_head_auroc_fold_seed_comparison', 'auroc',
        ),
        (
            'tail_corrupt', 'tail_aupr', 'Tail-corruption AUPR',
            'internal_tail_aupr_fold_seed_comparison', 'aupr',
        ),
        (
            'tail_corrupt', 'tail_auroc', 'Tail-corruption AUROC',
            'internal_tail_auroc_fold_seed_comparison', 'auroc',
        ),
        (
            'head_corrupt', 'head_hits30', 'Gene-corruption Hits@30',
            'internal_head_hits30_fold_seed_comparison', 'hits30',
        ),
    )
    for corruption, metric_key, y_label, output_name, metric_kind in figure_specs:
        records = collect_fold_records(
            results, corruption, metric_key, metric_kind, variants,
        )
        statistical_records = compute_statistics(records, metric_key, variants)
        summary_records = summarize(records, metric_key, variants)

        write_csv(output_dir / f'{metric_key}_source_data.csv', records)
        write_csv(
            output_dir / f'{metric_key}_summary_data.csv', summary_records,
        )
        write_csv(
            output_dir / f'{metric_key}_statistical_tests.csv',
            statistical_records,
        )
        for show_text in (True, False):
            plot_figure(
                output_dir / output_name,
                records,
                metric_key,
                y_label,
                show_text,
                variants,
            )
    print(f'Figure and source data saved to {output_dir}')


if __name__ == '__main__':
    main()
