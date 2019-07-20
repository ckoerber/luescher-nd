"""Plotting style set up for luescher-nd plots
"""
import re

import matplotlib
import seaborn as sns

EXPORT_OPTIONS = {"bbox_inches": "tight"}
LEGEND_STYLE = {"frameon": False, "loc": "center left", "bbox_to_anchor": (1.0, 0.5)}
LINE_STYLE = {"ls": "-", "lw": 1.0}
MARKERS = re.findall('``"([a-z]+)"``', matplotlib.markers.__doc__)
MARKER_STYLE = {"ms": 3}


def setup(n_colors: int = 5):
    """Sets up the font and colors
    """
    sns.set(
        context="paper", style="ticks", font_scale=1.0, rc={"mathtext.fontset": "cm"}
    )
    sns.set_palette("cubehelix", n_colors=n_colors, desat=None, color_codes=False)


def finalize():
    """Finalizes the plot (before exporting)
    """
    sns.despine()
