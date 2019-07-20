#!/usr/bin/env python3
"""Creates grid plot of Effective range expansions for different parameters
"""
import os

import seaborn as sns
import matplotlib.pylab as plt

from luescher_nd.database.utilities import read_table
from luescher_nd.database.utilities import DATA_FOLDER

from luescher_nd.plotting.styles import setup
from luescher_nd.plotting.styles import EXPORT_OPTIONS
from luescher_nd.plotting.styles import LINE_STYLE
from luescher_nd.plotting.styles import MARKERS


DATA_FOLDER = os.path.join(DATA_FOLDER, "three-d")

FILE_OPTIONS = [
    {
        "file_name": "db-contact-fv-c-fitted-parity-a-inv-lg.sqlite",
        "dispersion_zeta": False,
        "a_inv": -5.0,
    },
    {
        "file_name": "db-contact-fv-c-fitted-parity-lg.sqlite",
        "dispersion_zeta": False,
        "a_inv": 0,
    },
    {
        "file_name": "db-contact-fv-d-fitted-parity-lg.sqlite",
        "dispersion_zeta": True,
        "a_inv": 0,
    },
    {
        "file_name": "db-contact-fv-d-fitted-parity-a-inv-lg.sqlite",
        "dispersion_zeta": True,
        "a_inv": -5.0,
    },
]

ROUND_DIGITS = 3
FILTER_POLES = True
FILTER_BY_NSTATES = True
Y_BOUNDS = (-5, 5)


def export_grid_plot(file_name: str, dispersion_zeta: bool = True, a_inv: float = 0.0):
    """Generates a grid plot of ERE(x)

    With L on the rows, nstep on the cols and epsilon as hue/markers.

    Saves file to `{filename}.pdf`

    **Arguments**
        file_name: str
            The database to read from

        dispersion_zeta: bool = True
            If ERE should be computed using dispersion Lüscher or spherical Lüscher.

        a_inv: float = 0.0
            The expected offset of the ERE
    """
    df = (
        read_table(
            os.path.join(DATA_FOLDER, file_name),
            dispersion_zeta=dispersion_zeta,
            round_digits=ROUND_DIGITS,
            filter_poles=FILTER_POLES,
            filter_by_nstates=FILTER_BY_NSTATES,
        )
        .sort_values("x")
        .query(f"y > {a_inv + Y_BOUNDS[0]} and y < {a_inv + Y_BOUNDS[1]}")
    )

    title = "Dispersion Lüscher" if dispersion_zeta else "Spherical Lüscher"
    title += " fitted contact interaction"
    title += (
        " at unitarity"
        if abs(a_inv) < 1.0e-12
        else rf" at $-\frac{{1}}{{a_0}} = {a_inv:1.1f}$ [fm $^{{-1}}$]"
    )

    grid = sns.FacetGrid(
        data=df,
        col="nstep",
        hue="epsilon",
        row="L",
        sharex="row",
        sharey=True,
        legend_out=True,
        hue_kws={"marker": MARKERS, "ms": [1] * 10, "lw": [0.5] * 10, "ls": [":"] * 10},
        margin_titles=True,
        col_order=["1", "2", "4", "$\\infty$"],
    )
    grid.map(plt.plot, "x", "y")

    for ax in grid.axes.flatten():
        ax.axhline(a_inv, color="black", lw=0.5, zorder=-1)

    grid.add_legend(title=r"$\epsilon$ [fm]", frameon=False)
    grid.set_xlabels(r"$x = \left(\frac{p L}{2 \pi}\right)^2$")
    grid.set_ylabels(r"$p \cot(\delta_0(p))$ [fm$^{-1}$]")
    grid.set_titles(
        row_template=r"${row_var} = {row_name}$ [fm]",
        col_template=r"$n_{{\mathrm{{step}}}} =$ {col_name}",
    )

    grid.fig.suptitle(title, y=1.05)

    grid.savefig(file_name.replace(".sqlite", ".pdf"), **EXPORT_OPTIONS)


def main():
    """Export all the file options to pdf
    """
    setup()

    LINE_STYLE["ls"] = "--"

    for options in FILE_OPTIONS:
        export_grid_plot(**options)


if __name__ == "__main__":
    main()
