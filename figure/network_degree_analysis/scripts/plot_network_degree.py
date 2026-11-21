#!/usr/bin/env python3
"""Plot ECO knowledge-graph degree distributions and power-law tail fits.

Run from the repository root in the deepvcf environment:
    conda run -n deepvcf python figure/network_degree_analysis/scripts/plot_network_degree.py

Outputs: labeled and no-text SVG/PNG figures, source data, and fit statistics
under figure/network_degree_analysis/outputs/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import powerlaw
from scipy.special import zeta


# Keep text editable in SVG exports.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "kg" / "ECO"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'

METHODS = (
    ("Basic KG", "kg_basic.txt", "#5B7FCA"),
    ("Enhanced KG", "kg_enhanced.txt", "#B64342"),
)

sys.path.insert(0, str(PROJECT_ROOT))
from figure.figure_io import write_csv
from figure.figure_style import create_panel, save_panel  # noqa: E402


@dataclass(frozen=True)
class DegreeFit:
    """Empirical degree distribution and fitted power-law tail."""

    method: str
    color: str
    nodes: int
    edges: int
    degrees: np.ndarray
    degree_values: np.ndarray
    counts: np.ndarray
    empirical_ccdf: np.ndarray
    xmin: int
    alpha: float
    tail_n: int
    tail_fraction: float
    fit_x: np.ndarray
    fitted_ccdf: np.ndarray
    log10_ccdf_r2: float
    ks_distance: float
    llr_powerlaw_vs_exponential: float
    p_powerlaw_vs_exponential: float
    llr_powerlaw_vs_lognormal: float
    p_powerlaw_vs_lognormal: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing kg_basic.txt and kg_enhanced.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for figures and source-data tables.",
    )
    return parser.parse_args(argv)


def load_undirected_graph(path: Path) -> nx.Graph:
    """Load a three-column KG file as a simple undirected graph."""
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-graph file not found: {path}")

    graph = nx.Graph()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"Expected three tab-separated fields in {path} "
                    f"at line {line_number}; found {len(fields)}."
                )
            head, _relation, tail = fields
            if not head or not tail:
                raise ValueError(
                    f"Empty endpoint in {path} at line {line_number}."
                )
            graph.add_edge(head, tail)

    if graph.number_of_nodes() == 0:
        raise ValueError(f"No graph nodes were loaded from {path}.")
    return graph


def log10_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return R-squared between observed and predicted positive probabilities."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted arrays must be matching vectors.")
    if np.any(observed <= 0) or np.any(predicted <= 0):
        raise ValueError("Log-space R-squared requires positive probabilities.")
    log_observed = np.log10(observed)
    log_predicted = np.log10(predicted)
    denominator = np.sum((log_observed - log_observed.mean()) ** 2)
    if denominator == 0:
        raise ValueError("R-squared is undefined for a constant observed vector.")
    return float(1.0 - np.sum((log_observed - log_predicted) ** 2) / denominator)


def fit_degree_distribution(
    graph: nx.Graph,
    method: str,
    color: str,
) -> DegreeFit:
    """Fit a discrete power law and assemble plotting/statistical records."""
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=int)
    if degrees.ndim != 1 or degrees.size == 0 or np.any(degrees < 1):
        raise ValueError(f"{method} degrees must be non-empty positive integers.")

    degree_values, counts = np.unique(degrees, return_counts=True)
    empirical_ccdf = np.cumsum(counts[::-1])[::-1].astype(float) / degrees.size
    if not np.isclose(empirical_ccdf[0], 1.0):
        raise ValueError(f"{method} empirical CCDF does not begin at one.")
    if (
        np.any(empirical_ccdf <= 0)
        or np.any(empirical_ccdf > 1)
        or np.any(np.diff(empirical_ccdf) > 0)
    ):
        raise ValueError(f"{method} empirical CCDF is outside (0, 1] or increases.")

    fitted = powerlaw.Fit(degrees, discrete=True, verbose=False)
    xmin = int(fitted.power_law.xmin)
    alpha = float(fitted.power_law.alpha)
    if xmin < 1 or not np.isfinite(alpha) or alpha <= 1:
        raise ValueError(f"Invalid fitted parameters for {method}.")

    tail_n = int(np.count_nonzero(degrees >= xmin))
    tail_fraction = tail_n / degrees.size
    normalization = float(zeta(alpha, xmin))
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(f"Invalid discrete normalization for {method}.")

    fit_x = np.arange(xmin, int(degrees.max()) + 1, dtype=int)
    fitted_conditional_ccdf = zeta(alpha, fit_x) / normalization
    fitted_ccdf = tail_fraction * fitted_conditional_ccdf
    if (
        np.any(~np.isfinite(fitted_ccdf))
        or np.any(fitted_ccdf <= 0)
        or np.any(fitted_ccdf > 1)
        or np.any(np.diff(fitted_ccdf) > 0)
    ):
        raise ValueError(f"{method} fitted CCDF is outside (0, 1] or increases.")

    observed_tail_mask = degree_values >= xmin
    observed_tail_ccdf = empirical_ccdf[observed_tail_mask] / tail_fraction
    predicted_tail_ccdf = (
        zeta(alpha, degree_values[observed_tail_mask]) / normalization
    )
    r2 = log10_r2(observed_tail_ccdf, predicted_tail_ccdf)

    llr_exp, p_exp = fitted.distribution_compare(
        "power_law", "exponential", normalized_ratio=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No valid fits found for distribution lognormal."
        )
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        llr_lognormal, p_lognormal = fitted.distribution_compare(
            "power_law", "lognormal", normalized_ratio=True,
        )
    comparison_values = (llr_exp, p_exp, llr_lognormal, p_lognormal)
    if any(not np.isfinite(value) for value in comparison_values):
        raise ValueError(f"Non-finite distribution comparison for {method}.")
    if not (0 <= p_exp <= 1 and 0 <= p_lognormal <= 1):
        raise ValueError(f"Invalid comparison p-value for {method}.")

    return DegreeFit(
        method=method,
        color=color,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        degrees=degrees,
        degree_values=degree_values,
        counts=counts,
        empirical_ccdf=empirical_ccdf,
        xmin=xmin,
        alpha=alpha,
        tail_n=tail_n,
        tail_fraction=tail_fraction,
        fit_x=fit_x,
        fitted_ccdf=fitted_ccdf,
        log10_ccdf_r2=r2,
        ks_distance=float(fitted.power_law.D),
        llr_powerlaw_vs_exponential=float(llr_exp),
        p_powerlaw_vs_exponential=float(p_exp),
        llr_powerlaw_vs_lognormal=float(llr_lognormal),
        p_powerlaw_vs_lognormal=float(p_lognormal),
    )


def export_tables(fits: tuple[DegreeFit, ...], output_dir: Path) -> None:
    """Export empirical/fitted distributions and fit summaries."""
    source_rows: list[dict] = []
    for result in fits:
        fitted_by_degree = dict(zip(result.fit_x, result.fitted_ccdf))
        for degree, count, probability in zip(
            result.degree_values, result.counts, result.empirical_ccdf,
        ):
            source_rows.append({
                "method": result.method,
                "degree": int(degree),
                "node_count": int(count),
                "empirical_ccdf": float(probability),
                "fitted_ccdf": fitted_by_degree.get(int(degree), ""),
            })

    statistics_rows = [
        {
            "method": result.method,
            "nodes": result.nodes,
            "edges": result.edges,
            "mean_degree": float(result.degrees.mean()),
            "maximum_degree": int(result.degrees.max()),
            "xmin": result.xmin,
            "power_law_alpha": result.alpha,
            "tail_nodes": result.tail_n,
            "tail_fraction": result.tail_fraction,
            "log10_ccdf_r_squared": result.log10_ccdf_r2,
            "ks_distance": result.ks_distance,
            "llr_powerlaw_vs_exponential": (
                result.llr_powerlaw_vs_exponential
            ),
            "p_powerlaw_vs_exponential": result.p_powerlaw_vs_exponential,
            "llr_powerlaw_vs_lognormal": result.llr_powerlaw_vs_lognormal,
            "p_powerlaw_vs_lognormal": result.p_powerlaw_vs_lognormal,
        }
        for result in fits
    ]
    write_csv(output_dir / "network_degree_source_data.csv", source_rows)
    write_csv(output_dir / "network_degree_fit_statistics.csv", statistics_rows)


def plot_degree_distribution(
    fits: tuple[DegreeFit, ...],
    output_stem: Path,
    show_text: bool,
) -> tuple[Path, Path]:
    """Render the square_v1 log-log degree-distribution panel."""
    fig, ax = create_panel("square_v1")

    for result in fits:
        marker = "o"
        ax.plot(
            result.degree_values,
            result.empirical_ccdf,
            linestyle="none",
            marker=marker,
            markersize=1.7,
            markerfacecolor=result.color,
            markeredgecolor=result.color,
            alpha=1.0,
            clip_on=False,
            zorder=3,
        )
        ax.plot(
            result.fit_x,
            result.fitted_ccdf,
            color=result.color,
            linewidth=1.6,
            alpha=1.0,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, max(result.degrees.max() for result in fits) * 1.25)
    minimum_probability = min(
        float(result.fitted_ccdf.min()) for result in fits
    )
    lower_decade = 10 ** np.floor(np.log10(minimum_probability))
    ax.set_ylim(lower_decade, 1.0)
    ax.grid(False, which="both")

    if show_text:
        ax.set_xlabel("Degree, $k$")
        ax.set_ylabel("CCDF, $P(K \\geq k)$")
        handles = [
            Line2D(
                [], [], color=result.color, linewidth=1.6,
                marker="o",
                markersize=2.8,
                markerfacecolor=result.color,
                markeredgecolor=result.color,
                label=(
                    f"{result.method.replace(' KG', '')}: "
                    f"($R^2$ = {result.log10_ccdf_r2:.2f}, "
                    f"$D_{{KS}}$ = {result.ks_distance:.3f})"
                ),
            )
            for result in fits
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=6.2,
            handlelength=1.6,
            borderaxespad=0.35,
            labelspacing=0.45,
        )
    else:
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
        output_stem = Path(f"{output_stem}_no_text")

    return save_panel(fig, output_stem)


def run_analysis(data_root: Path, output_dir: Path) -> tuple[DegreeFit, ...]:
    """Load data, fit both networks, and export tables and figure variants."""
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fits = tuple(
        fit_degree_distribution(
            load_undirected_graph(data_root / filename), method, color,
        )
        for method, filename, color in METHODS
    )
    export_tables(fits, output_dir)
    for show_text in (True, False):
        plot_degree_distribution(
            fits,
            output_dir / "network_degree_distribution",
            show_text=show_text,
        )
    return fits


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    fits = run_analysis(args.data_root, args.output_dir)
    for result in fits:
        print(
            f"{result.method}: nodes={result.nodes}, edges={result.edges}, "
            f"gamma={result.alpha:.3f}, xmin={result.xmin}, "
            f"log10 CCDF R^2={result.log10_ccdf_r2:.3f}"
        )
    print(f"Figure and source data saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
