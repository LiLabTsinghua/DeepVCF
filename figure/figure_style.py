"""Shared publication-panel templates and visual constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


MM_PER_INCH = 25.4


@dataclass(frozen=True)
class PanelTemplate:
    """Fixed geometry for one independently exported manuscript panel."""

    name: str
    width_mm: float
    height_mm: float
    axes_rect: tuple[float, float, float, float]
    axes_box_aspect: float


@dataclass(frozen=True)
class SmallMultiplesTemplate:
    """Geometry shared by a configurable grid of homogeneous subplots."""

    name: str
    panel_width_mm: float
    panel_height_mm: float
    horizontal_gap_mm: float
    vertical_gap_mm: float
    left_mm: float
    right_mm: float
    bottom_mm: float
    top_mm: float


SQUARE_V1 = PanelTemplate(
    name="square_v1",
    width_mm=90.0,
    height_mm=78.0,
    axes_rect=(0.18, 0.26, 0.79, 0.71),
    axes_box_aspect=0.86,
)

WIDE_V1 = PanelTemplate(
    name="wide_v1",
    width_mm=120.0,
    height_mm=78.0,
    # Preserve square_v1's absolute margins and plotting height while adding
    # 30 mm of horizontal plotting space.
    axes_rect=(0.135, 0.26, 0.8425, 0.71),
    axes_box_aspect=0.55,
)

NINE_METHOD_WIDE_V1 = PanelTemplate(
    name="nine_method_wide_v1",
    width_mm=94.7,
    height_mm=78.0,
    # Nine-category companion to wide_v1. It preserves the same absolute
    # margins and approximately the same physical width per category as the
    # 12-category wide panel.
    axes_rect=(0.1711, 0.26, 0.8004, 0.71),
    axes_box_aspect=0.73,
)

NARROW_V1 = PanelTemplate(
    name="narrow_v1",
    width_mm=60.0,
    height_mm=78.0,
    # Three-method companion to square_v1: identical height, with enough
    # absolute left margin for the shared y-axis labels and a narrower body.
    axes_rect=(0.27, 0.26, 0.68, 0.71),
    axes_box_aspect=1.36,
)

TWO_BAR_V1 = PanelTemplate(
    name="two_bar_v1",
    width_mm=47.0,
    height_mm=78.0,
    # Two-bar companion to narrow_v1. Absolute left/right and vertical
    # margins are retained while redundant horizontal space is removed.
    axes_rect=(0.345, 0.26, 0.591, 0.71),
    axes_box_aspect=1.99,
)

TALL_V1 = PanelTemplate(
    name="tall_v1",
    width_mm=85.0,
    height_mm=105.0,
    # A taller single panel with extra left space for dense category labels.
    axes_rect=(0.24, 0.19, 0.73, 0.785),
    axes_box_aspect=1.20,
)

PANEL_TEMPLATES = {
    SQUARE_V1.name: SQUARE_V1,
    WIDE_V1.name: WIDE_V1,
    NINE_METHOD_WIDE_V1.name: NINE_METHOD_WIDE_V1,
    NARROW_V1.name: NARROW_V1,
    TWO_BAR_V1.name: TWO_BAR_V1,
    TALL_V1.name: TALL_V1,
}

SMALL_MULTIPLES_V1 = SmallMultiplesTemplate(
    name="small_multiples_v1",
    panel_width_mm=38.0,
    panel_height_mm=42.0,
    horizontal_gap_mm=4.0,
    vertical_gap_mm=7.0,
    left_mm=18.0,
    right_mm=4.0,
    bottom_mm=23.0,
    top_mm=16.0,
)

SMALL_MULTIPLES_TEMPLATES = {
    SMALL_MULTIPLES_V1.name: SMALL_MULTIPLES_V1,
}

METHOD_ORDER = (
    "FSEOF",
    "FVSEOF",
    "iBridge",
    "LR_sklearn",
    "RF_sklearn",
    "MLP_sklearn",
    "DeepVCF",
    "DeepVCF+BE",
    "DeepVCF+BE+MP",
)

METHOD_TICK_LABELS = {
    "FSEOF": "FSEOF",
    "FVSEOF": "FVSEOF",
    "iBridge": "iBridge",
    "LR_sklearn": "LR",
    "RF_sklearn": "RF",
    "MLP_sklearn": "MLP",
    "DeepVCF": "DeepVCF",
    "DeepVCF+BE": "+BE",
    "DeepVCF+BE+MP": "+BE+MP",
}

METHOD_COLORS = {
    "FSEOF": "#B0B0B0",
    "FVSEOF": "#E6AD87",
    "iBridge": "#777777",
    "LR_sklearn": "#60A5FA",
    "RF_sklearn": "#3B82F6",
    "MLP_sklearn": "#1D4ED8",
    "DeepVCF": "#BF7EEE",
    "DeepVCF+BE": "#773ADA",
    "DeepVCF+BE+MP": "#451594",
}

EDGE_COLOR = "#111111"
FONT_SIZE = 13
AXIS_LABEL_SIZE = 13
NUMERIC_TICK_SIZE = 12
CATEGORY_TICK_SIZE = 9.5
AXIS_LINEWIDTH = 0.8
TICK_WIDTH = 0.8
TICK_LENGTH = 4.0
BAR_EDGE_WIDTH = 0.8
BAR_WIDTH = 0.72
SCATTER_SIZE = 34
SCATTER_SIZE_MIN = 10
SCATTER_BAR_DIAMETER_FRACTION = 0.32
DATA_LINEWIDTH = 2.5
MARKER_SIZE = 8.5
MARKER_EDGE_WIDTH = 0.75
ERRORBAR_LINEWIDTH = 1.3
ERRORBAR_CAPSIZE = 4.0
PNG_DPI = 600

# Physical geometry for bar-and-seed-dot figures. One x-data unit is one mm.
BAR_DOT_HEIGHT_MM = 78.0
BAR_DOT_LEFT_MM = 18.0
BAR_DOT_RIGHT_MM = 4.0
BAR_DOT_BOTTOM_MM = 21.0
BAR_DOT_TOP_MM = 3.0
BAR_DOT_BAR_WIDTH_MM = 8.0
BAR_DOT_WITHIN_GAP_MM = 1.2
BAR_DOT_GROUP_GAP_MM = 4.0
BAR_DOT_SIDE_PADDING_MM = 2.5
BAR_DOT_POINT_DIAMETER_MM = 1.5
BAR_DOT_ERROR_CAP_WIDTH_MM = 2.5
BAR_DOT_ERROR_LINEWIDTH = 1.0


def create_bar_dot_panel(group_sizes):
    """Create a 78-mm-high panel whose width follows fixed-width grouped bars.

    Returns the figure, axes, and one tuple of bar centers (in mm) per group.
    """
    sizes = tuple(group_sizes)
    if not sizes or any(not isinstance(size, int) or size < 1 for size in sizes):
        raise ValueError("group_sizes must contain positive integers.")

    centers = []
    cursor_mm = BAR_DOT_SIDE_PADDING_MM
    for group_index, size in enumerate(sizes):
        group_centers = []
        for bar_index in range(size):
            group_centers.append(cursor_mm + BAR_DOT_BAR_WIDTH_MM / 2)
            cursor_mm += BAR_DOT_BAR_WIDTH_MM
            if bar_index < size - 1:
                cursor_mm += BAR_DOT_WITHIN_GAP_MM
        centers.append(tuple(group_centers))
        if group_index < len(sizes) - 1:
            cursor_mm += BAR_DOT_GROUP_GAP_MM
    plot_width_mm = cursor_mm + BAR_DOT_SIDE_PADDING_MM
    canvas_width_mm = BAR_DOT_LEFT_MM + plot_width_mm + BAR_DOT_RIGHT_MM

    apply_publication_style()
    fig = plt.figure(
        figsize=(canvas_width_mm / MM_PER_INCH, BAR_DOT_HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    ax = fig.add_axes((
        BAR_DOT_LEFT_MM / canvas_width_mm,
        BAR_DOT_BOTTOM_MM / BAR_DOT_HEIGHT_MM,
        plot_width_mm / canvas_width_mm,
        (BAR_DOT_HEIGHT_MM - BAR_DOT_BOTTOM_MM - BAR_DOT_TOP_MM)
        / BAR_DOT_HEIGHT_MM,
    ))
    style_axis(ax)
    ax.set_xlim(0, plot_width_mm)
    return fig, ax, tuple(centers)


def apply_publication_style():
    """Apply the immutable visual parameters shared by all panel templates."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "DejaVu Sans", "Liberation Sans",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": NUMERIC_TICK_SIZE,
        "ytick.labelsize": NUMERIC_TICK_SIZE,
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def create_panel(template_name="square_v1"):
    """Create one fixed-size panel using a registered template."""
    try:
        template = PANEL_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(PANEL_TEMPLATES)
        raise ValueError(
            f"Unknown panel template {template_name!r}; available: {available}."
        ) from exc

    apply_publication_style()
    fig = plt.figure(
        figsize=(
            template.width_mm / MM_PER_INCH,
            template.height_mm / MM_PER_INCH,
        ),
        facecolor="white",
    )
    ax = fig.add_axes(template.axes_rect)
    ax.set_box_aspect(template.axes_box_aspect)
    style_axis(ax)
    return fig, ax


def create_small_multiples(
    template_name,
    *,
    nrows,
    ncols,
    sharex=False,
    sharey=False,
    panel_width_mm=None,
    panel_height_mm=None,
):
    """Create an explicitly arranged grid of homogeneous small plots.

    ``nrows`` and ``ncols`` are intentionally required: the template defines
    visual geometry but never chooses the scientific panel arrangement.
    """
    try:
        template = SMALL_MULTIPLES_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(SMALL_MULTIPLES_TEMPLATES)
        raise ValueError(
            f"Unknown small-multiples template {template_name!r}; "
            f"available: {available}."
        ) from exc
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must both be positive integers.")

    panel_width_mm = (
        template.panel_width_mm if panel_width_mm is None else panel_width_mm
    )
    panel_height_mm = (
        template.panel_height_mm if panel_height_mm is None else panel_height_mm
    )
    if panel_width_mm <= 0 or panel_height_mm <= 0:
        raise ValueError("Panel width and height must both be positive.")

    width_mm = (
        template.left_mm
        + template.right_mm
        + ncols * panel_width_mm
        + (ncols - 1) * template.horizontal_gap_mm
    )
    height_mm = (
        template.bottom_mm
        + template.top_mm
        + nrows * panel_height_mm
        + (nrows - 1) * template.vertical_gap_mm
    )
    apply_publication_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        facecolor="white",
    )
    fig.subplots_adjust(
        left=template.left_mm / width_mm,
        right=1 - template.right_mm / width_mm,
        bottom=template.bottom_mm / height_mm,
        top=1 - template.top_mm / height_mm,
        wspace=template.horizontal_gap_mm / panel_width_mm,
        hspace=template.vertical_gap_mm / panel_height_mm,
    )
    for ax in axes.flat:
        style_axis(ax)
    return fig, axes


def style_axis(ax):
    """Apply the shared axis, spine, and tick treatment."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(
        axis="both",
        colors=EDGE_COLOR,
        direction="out",
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    ax.grid(False)


def scatter_size_for_bar(
    ax,
    bar_width,
    *,
    max_size=SCATTER_SIZE,
    min_size=SCATTER_SIZE_MIN,
    diameter_fraction=SCATTER_BAR_DIAMETER_FRACTION,
):
    """Return a scatter area scaled to the displayed width of a bar."""
    if bar_width <= 0:
        raise ValueError("bar_width must be positive.")
    if not 0 < diameter_fraction <= 1:
        raise ValueError("diameter_fraction must lie in (0, 1].")
    if min_size <= 0 or max_size < min_size:
        raise ValueError("Require 0 < min_size <= max_size.")

    # Bar artists update data limits lazily; draw once so the transform reflects
    # the final number and grouping of bars in the fixed template canvas.
    ax.figure.canvas.draw()
    x0_px = ax.transData.transform((0.0, 0.0))[0]
    x1_px = ax.transData.transform((bar_width, 0.0))[0]
    bar_width_points = abs(x1_px - x0_px) * 72.0 / ax.figure.dpi
    marker_diameter_points = bar_width_points * diameter_fraction
    return min(max(marker_diameter_points**2, min_size), max_size)


def save_panel(fig, output_stem, dpi=PNG_DPI):
    """Save fixed-canvas SVG and PNG outputs without automatic cropping."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(svg_path, facecolor="white")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return svg_path, png_path
