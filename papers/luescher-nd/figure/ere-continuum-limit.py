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
        "zeta": "spherical",
        "a_inv": -5.0,
    },
    {
        "file_name": "db-contact-fv-c-fitted-parity-lg.sqlite",
        "zeta": "spherical",
        "a_inv": 0,
    },
]

ROUND_DIGITS = 3
FILTER_POLES = True
FILTER_BY_NSTATES = True
Y_BOUNDS = (1.0e-3, 1.0e1)


def export_grid_plot(file_name: str, zeta: str = "spherical", a_inv: float = 0.0):
    """Generates a grid plot of ERE(x)

    With L on the rows, nstep on the cols and epsilon as hue/markers.

    Saves file to `{filename}.pdf`

    **Arguments**
        file_name: str
            The database to read from

        zeta: str = "spherical"
            If ERE should be computed using dispersion, cartesian or spherical Lüscher.

        a_inv: float = 0.0
            The expected offset of the ERE
    """
    df = (
        read_table(
            os.path.join(DATA_FOLDER, file_name),
            zeta=zeta,
            round_digits=ROUND_DIGITS,
            filter_poles=FILTER_POLES,
            filter_by_nstates=FILTER_BY_NSTATES,
        )
        .sort_values("x")
        .query("nstep == '4' or nstep == '$\\infty$' and L == 1")
        .sort_values("epsilon")
    )
    df["y"] -= a_inv
    df = df.query(f"y > {Y_BOUNDS[0]} and y < {Y_BOUNDS[1]}")

    title = f"{zeta.capitalize()} Lüscher"
    title += " fitted contact interaction"
    title += (
        " at unitarity"
        if abs(a_inv) < 1.0e-12
        else rf" at $-\frac{{1}}{{a_0}} = {a_inv:1.1f}$ [fm $^{{-1}}$]"
    )

    grid = sns.FacetGrid(
        data=df,
        col="nstep",
        hue="nlevel",
        row="L",
        legend_out=True,
        sharex=False,
        sharey=False,
        margin_titles=True,
        col_order=["4", "$\\infty$"],
    )
    grid.map(plt.plot, "epsilon", "y", marker="o", ms=1, ls=":", lw=0.5)

    grid.set_xlabels(r"$\epsilon$ [fm]")
    # grid.add_legend(title=r"$n_{\mathrm{level}}$", frameon=False)
    grid.set_ylabels(r"$p \cot(\delta_0(p))$ [fm$^{-1}$]")
    grid.set_titles(
        row_template=r"${row_var} = {row_name}$ [fm]",
        col_template=r"$n_{{\mathrm{{step}}}} =$ {col_name}",
    )

    grid.fig.suptitle(title, y=1.05)

    for ax in grid.axes.flatten():
        ax.set_yscale("log")
        ax.set_xscale("log")

    grid.savefig(
        "continuum-limit-" + file_name.replace(".sqlite", ".pdf"), **EXPORT_OPTIONS
    )


def main():
    """Export all the file options to pdf
    """
    setup()

    LINE_STYLE["ls"] = "--"

    for options in FILE_OPTIONS:
        export_grid_plot(**options)


if __name__ == "__main__":
    main()
