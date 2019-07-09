"""Plotting style set up for luescher-nd plots
"""
import seaborn as sns


RC = {
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "font.sans-serif": "Computer Modern Sans serif",
    "font.monospace": "Computer Modern Typewriter",
    "text.usetex": False,
}

EXPORT_OPTIONS = {"bbox_inches": "tight"}
LEGEND_STYLE = {"frameon": False, "loc": "center left", "bbox_to_anchor": (1.0, 0.5)}
LINE_STYLE = {"ls": "-", "lw": 0.5}


def setup(n_colors: int = 5):
    """Sets up the font and colors
    """
    sns.set(
        context="paper", style="ticks", font_scale=1, font="Computer Modern Roman", rc=RC
    )
    sns.set_palette("cubehelix", n_colors=n_colors, desat=None, color_codes=False)


def finalize():
    """Finalizes the plot (before exporting)
    """
    sns.despine()
