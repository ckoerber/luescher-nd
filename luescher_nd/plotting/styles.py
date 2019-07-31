"""Plotting style set up for luescher-nd plots
"""
from typing import Optional

import re

import matplotlib
import seaborn as sns

EXPORT_OPTIONS = {"bbox_inches": "tight"}
LEGEND_STYLE = {"frameon": False, "loc": "center left", "bbox_to_anchor": (1.0, 0.5)}
LINE_STYLE = {"ls": "-", "lw": 1.0}
MARKERS = re.findall('``"([a-z]+)"``', matplotlib.markers.__doc__)
MARKER_STYLE = {"ms": 3}
WIDTH = 6.0


def setup(n_colors: int = 5, pgf: bool = False, font_scale: float = 1.0):
    """Sets up the font and colors
    """
    sns.set_palette("cubehelix", n_colors=n_colors, desat=None, color_codes=False)
    if pgf:
        matplotlib.use("pgf")
        sns.set(
            context="paper",
            style="ticks",
            font_scale=font_scale,
            rc={
                "pgf.rcfonts": False,
                "axes.unicode_minus": False,
                "font.serif": [],
                "font.sans-serif": [],
            },
        )
    else:
        sns.set(
            context="paper",
            style="ticks",
            font_scale=font_scale,
            rc={"mathtext.fontset": "cm"},
        )


def finalize(fig: Optional[matplotlib.figure.Figure] = None, width: float = 1.0):
    """Finalizes the plot (before exporting)
    """
    sns.despine()
    if fig:
        ratio = fig.get_figheight() / fig.get_figwidth()
        fig.set_figwidth(WIDTH * width)
        fig.set_figheight(ratio * WIDTH * width)

        for ax in fig.axes:
            all_texts = [text for text in ax.texts]
            all_texts += [ax.title]
            all_texts += [t for t in ax.xaxis.get_ticklabels()]
            all_texts += [t for t in ax.yaxis.get_ticklabels()]
            for text in all_texts:
                if not "$" in text.get_text():
                    text.set_text(f"${text.get_text()}$")
