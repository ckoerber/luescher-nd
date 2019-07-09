#!/usr/bin/env python3
"""Creates grid plot of Effective range expansions for different parameters
"""
import os
from itertools import product

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt

from luescher_nd.database.utilities import read_table
from luescher_nd.database.utilities import DATA_FOLDER

from luescher_nd.plotting.styles import setup
from luescher_nd.plotting.styles import finalize
from luescher_nd.plotting.styles import LEGEND_STYLE
from luescher_nd.plotting.styles import EXPORT_OPTIONS
from luescher_nd.plotting.styles import LINE_STYLE
from luescher_nd.plotting.styles import MARKERS


FILE_OPTIONS = [
    {"file_name": "db-contact-fv-d-fitted.sqlite", "dispersion_zeta": True},
    {"file_name": "db-contact-fv-c-fitted.sqlite", "dispersion_zeta": False},
]


def export_grid_plot(file_name: str, dispersion_zeta: bool = True):
    """Generates a grid plot of ERE(x)

    With L on the rows, nstep on the cols and epsilon as hue/markers.

    Saves file to `{filename}.pdf`

    **Arguments**
        file_name: str
            The database to read from

        dispersion_zeta: bool = True
            If ERE should be computed using dispersion Lüscher or spherical Lüscher.
    """
    df = read_table(
        os.path.join(DATA_FOLDER, file_name),
        dispersion_zeta=dispersion_zeta,
        round_digits=1,
    ).sort_values("x")

    title = "Dispersion Lüscher" if dispersion_zeta else "Spherical Lüscher"
    title += " fitted contact interaction at unitarity"

    grid = sns.FacetGrid(
        data=df,
        col="nstep",
        hue="epsilon",
        row="L",
        sharex="row",
        sharey=True,
        legend_out=True,
        hue_kws={"marker": MARKERS},
        margin_titles=True,
        col_order=df.nstep.unique()[::-1],
    )
    grid.map(plt.plot, "x", "y", **LINE_STYLE)

    for ax in grid.axes.flatten():
        if dispersion_zeta:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 5)

    grid.add_legend(title=r"$\epsilon$ [fm]", frameon=False)
    grid.set_xlabels(r"$x = \frac{p^2 L^2}{4 \pi^2}$")
    grid.set_ylabels(r"$p \cot(\delta_0(p))$ [fm$^{-1}$]")
    grid.set_titles(
        row_template="${row_var} = {row_name}$ [fm]",
        col_template="$n_{{\mathrm{{step}}}} =$ {col_name}",
    )

    grid.fig.suptitle(title, y=1.05)

    grid.savefig(file_name.replace(".sqlite", ".pdf"), **EXPORT_OPTIONS)


def main():
    """Export all the file options to pdf
    """
    setup()

    for options in FILE_OPTIONS:
        export_grid_plot(**options)


if __name__ == "__main__":
    main()
