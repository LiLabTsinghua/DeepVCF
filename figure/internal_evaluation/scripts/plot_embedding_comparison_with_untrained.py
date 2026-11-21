"""Plot Untrained, Random, Basic, and Enhanced internal results in a 2 x 7 layout.

Run from the repository root in the deepvcf environment:
    conda run -n deepvcf python figure/internal_evaluation/scripts/plot_embedding_comparison_with_untrained.py

Inputs: results/internal_evaluation/embedding_comparison/<variant>/seed_<seed>/benchmark.json.
Outputs: figure/internal_evaluation/outputs/embedding_comparison_with_untrained/.
Each condition uses five seeds and five FoldUnits; bars show mean and sample SD.
Exports five metrics as labeled/no-text SVG and PNG, with source, summary,
and exploratory statistical CSVs. --results-root and --output-dir override paths.
"""

from embedding_plotting import main


if __name__ == '__main__':
    main(include_untrained=True)
