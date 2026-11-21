"""Shared bar layout and significance labels for experimental validation figures."""

from figure.figure_style import MM_PER_INCH, PANEL_TEMPLATES, create_panel

BAR_CENTER_MM = 8.5


BAR_WIDTH = 0.68


CONTROL_COLOR = "#8C8C8C"


POINT_COLOR = "#161616"


def significance_symbol(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def create_variable_wide_panel(n_bars):
    """Reuse wide_v1 margins and styling, scaling its plotting width by bar count."""
    template = PANEL_TEMPLATES["wide_v1"]
    template_width = template.width_mm
    height = template.height_mm
    left = template_width * template.axes_rect[0]
    bottom = height * template.axes_rect[1]
    right = template_width * (1 - template.axes_rect[0] - template.axes_rect[2])
    plot_height = height * template.axes_rect[3]
    plot_width = n_bars * BAR_CENTER_MM
    width = left + plot_width + right
    fig, ax = create_panel("wide_v1")
    fig.set_size_inches(width / MM_PER_INCH, height / MM_PER_INCH)
    ax.set_position((left / width, bottom / height, plot_width / width, plot_height / height))
    ax.set_box_aspect(plot_height / plot_width)
    return fig, ax
