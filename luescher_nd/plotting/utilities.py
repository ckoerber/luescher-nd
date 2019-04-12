"""Plotting utility functions for luescher_nd project
"""
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns


def plot_ere_grid(df: pd.DataFrame, x="x", y="y", **kwargs) -> sns.FacetGrid:
    """Plots effective range expansion grid for input DataFrame.
    """
    default_kwargs = {
        "col": "nstep",
        "hue": "n1d_max",
        "hue_kws": {"marker": ["s", "o", "v", "d"]},
        "palette": "cubehelix",
        "margin_titles": True,
    }

    default_kwargs.update(**kwargs)

    g = sns.FacetGrid(df, **default_kwargs)
    g.map(plt.plot, x, y, ms=1, lw=0.5, ls="--")

    for ax in np.array(g.axes).flatten():
        ax.axhline(0, c="black", lw=0.2)
        ax.axvline(0, c="black", lw=0.2)

    g.set_xlabels("$x$")
    g.set_ylabels(r"$p \cot (\delta_0(p))$")
    g.add_legend(title=r"$\frac{L}{\epsilon}$", fontsize=6)

    return g
