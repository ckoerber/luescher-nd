"""Plotting style set up for luescher-nd plots
"""
from typing import Optional
from typing import Dict
from typing import Any

import re

import matplotlib
import seaborn as sns

EXPORT_OPTIONS = {"bbox_inches": "tight"}
LEGEND_STYLE = {"frameon": False, "loc": "center left", "bbox_to_anchor": (1.0, 0.5)}
LINE_STYLE = {"ls": "-", "lw": 1.0}
MARKERS = re.findall('``"([a-z]+)"``', matplotlib.markers.__doc__)
MARKER_STYLE = {"ms": 3}
WIDTH = 6.0


def setup(
    n_colors: int = 5,
    pgf: bool = False,
    font_scale: float = 1.0,
    rc_kwargs: Optional[Dict[str, Any]] = None,
):
    """Sets up the font and colors
    """
    rc_kwargs = rc_kwargs or {}
    sns.set_palette("cubehelix", n_colors=n_colors, desat=None, color_codes=False)

    if pgf:
        matplotlib.use("pgf")
        rc = {
            "pgf.rcfonts": False,
            "axes.unicode_minus": False,
            "font.family": "serif",
            "font.serif": [],
            "font.sans-serif": [],
        }
        rc.update(rc_kwargs)
    else:
        rc = {"mathtext.fontset": "cm"}
        rc.update(rc_kwargs)

    sns.set(context="paper", style="ticks", font_scale=font_scale, rc=rc)


def mathify(text: "Text") -> "Text":
    """Wraps text instance in latex math envorinment if not already in env.
    """
    string = text.get_text()
    if string != "" and not "$" in string:
        text.set_text(f"${string}$")

    return text


def finalize(fig: Optional[matplotlib.figure.Figure] = None, width: float = 1.0):
    """Finalizes the plot (before exporting)
    """
    sns.despine()
    if fig:
        ratio = fig.get_figheight() / fig.get_figwidth()
        fig.set_figwidth(WIDTH * width)
        fig.set_figheight(ratio * WIDTH * width)
