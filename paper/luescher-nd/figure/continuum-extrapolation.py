#!/usr/bin/env python3
"""Creates grid plot of effective range expansions using standard L\"{u}scher extracted by
extrapolating lattice spacing to to zero.
"""
import os
import re

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt

from luescher_nd.zeta.extern.pyzeta import zeta  # pylint: disable=E0611

from luescher_nd.database.utilities import read_table
from luescher_nd.database.utilities import get_continuum_extrapolation
from luescher_nd.database.utilities import DATA_FOLDER
from luescher_nd.database.utilities import get_degeneracy

from luescher_nd.plotting.styles import setup
from luescher_nd.plotting.styles import EXPORT_OPTIONS
from luescher_nd.plotting.styles import MARKERS
from luescher_nd.plotting.styles import finalize


FILES = [
    "contact-fitted_a-inv=+0.0_zeta=spherical_projector=a1g_n-eigs=200.sqlite",
    "contact-fitted_a-inv=-5.0_zeta=spherical_projector=a1g_n-eigs=200.sqlite",
]

FILTER = "nlevel < 20 and n1d > 5"
ROUND_DIGITS = 0
FILTER_POLES = False
FILTER_BY_NSTATES = False
Y_BOUNDS = (-3, 3)


def nstep_label(nstep) -> str:
    return "$" + (str(nstep) if nstep > 0 else r"\infty") + "$"


def get_spectrum(df: pd.DataFrame) -> "figure":  # pylint: disable=C0103
    """Exports effective spectrum to file.

    **Arguments**
        df: pd.DataFrame
            Must contain columns nstep, nlevel, L, epsilon and x

        file_name: str
            Name used for the export.
    """

    grid = sns.FacetGrid(
        data=df,
        col="nstep_label",
        hue="nlevel",
        row="L",
        sharex="row",
        sharey=True,
        legend_out=True,
        hue_kws={
            "marker": MARKERS * 5,
            "ms": [2] * 40,
            "lw": [0.5] * 40,
            "ls": ["--"] * 40,
        },
        margin_titles=True,
        col_order=[nstep_label(nstep) for nstep in [1, 2, 4, -1]],
    )
    grid.map(plt.plot, "epsilon", "x")

    grid.set_ylabels(r"$x = \frac{2 \mu E L^2}{4 \pi^2}$")
    grid.set_xlabels(r"$\epsilon [\mathrm{fm}]$")
    grid.set_titles(
        row_template=r"${row_var} = {row_name} [\mathrm{{fm}}]$",
        col_template=r"$n_s =$ {col_name}",
    )

    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    degs = [n2 for n2 in get_degeneracy(20) if n2 < 20]
    for ax in grid.axes.flatten():
        ax.set_yticks(degs)
        for d in set(degs):
            ax.axhline(d, ls="-", color="grey", lw=0.5, zorder=-10, alpha=0.5)

    return grid.fig


def get_ere(
    input_df: pd.DataFrame, a_inv: float, filter_degeneracy: bool = True
):  # pylint: disable=C0103
    """Exports effective range expansion to file.

    **Arguments**
        df: pd.DataFrame
            Must contain columns nstep, nlevel, L, epsilon=0 and x

        file_name: str
            Name used for the export.
    """
    df = input_df.copy()
    if filter_degeneracy:
        degenerate_nums = [n2 for n2, vec in get_degeneracy(20).items() if len(vec) > 1]
        df = df[~df.x.round(ROUND_DIGITS).isin(degenerate_nums)]

    df["y"] = zeta(df["x"]) / np.pi / df["L"]

    grid = sns.FacetGrid(
        data=df,
        col="L",
        hue="nstep_label",
        sharex=False,
        sharey=True,
        legend_out=True,
        hue_kws={
            "marker": MARKERS * 5,
            "ms": [1] * 40,
            "lw": [0.5] * 40,
            "ls": ["--"] * 40,
        },
        hue_order=[nstep_label(nstep) for nstep in [1, 2, 4, -1]],
        margin_titles=True,
    )
    grid.map(plt.plot, "x", "y")

    for ax in grid.axes.flatten():
        ax.axhline(a_inv, color="black", ls="-", lw=0.5)
        ax.set_ylim(a_inv + Y_BOUNDS[0], a_inv - Y_BOUNDS[0])

    grid.add_legend(title=r"$n_s$", frameon=False)
    grid.set_xlabels(r"$x = \frac{2 \mu E L^2}{4 \pi^2}$")
    grid.set_ylabels(r"$p \cot (\delta(p)) [\mathrm{fm}^{-1}]$")
    grid.set_titles(col_template=r"${col_var} = {col_name} [\mathrm{{fm}}]$")

    return grid.fig


def export_grid_plot(file_name: str):
    """Generates a grid plot of ERE for continuum extrapolated spectrum

    Saves spectrum to `continuum-spectrum-{filename}.pdf` and
    ere to Saves spectrum to `continuum-ere-{filename}.pdf`

    **Arguments**
        file_name: str
            The database to read from
    """
    df = read_table(
        os.path.join(DATA_FOLDER, file_name),
        zeta="spherical",
        round_digits=ROUND_DIGITS,
        filter_poles=FILTER_POLES,
        filter_by_nstates=FILTER_BY_NSTATES,
    ).query(FILTER)[["L", "epsilon", "nstep", "nlevel", "x"]]
    df["L"] = df["L"].round(14)

    df["nstep_label"] = df.nstep.apply(nstep_label)

    fit_df = get_continuum_extrapolation(df, include_statistics=False)
    tf = df.append(fit_df, sort=True).sort_values(["nlevel", "L", "nstep", "epsilon"])
    fit_df["nstep_label"] = fit_df.nstep.apply(nstep_label)

    a_inv = float(re.findall(r"a-inv=([\-\+\.0-9]+)", file_name)[0])

    setup(pgf=True, font_scale=0.5)
    fig = get_spectrum(tf)
    finalize(fig)
    fig.savefig(
        file_name.replace("contact-fitted", "continuum-spectrum")
        .replace(".sqlite", ".pgf")
        .replace("=", "_"),
        **EXPORT_OPTIONS,
    )

    setup(pgf=True, font_scale=1.0)
    fig = get_ere(fit_df, a_inv)
    finalize(fig)
    fig.savefig(
        file_name.replace("contact-fitted", "continuum-ere")
        .replace(".sqlite", ".pgf")
        .replace("=", "_"),
        **EXPORT_OPTIONS,
    )


def main():
    """Export all the file options to pdf
    """

    for file_name in FILES:
        export_grid_plot(file_name)


if __name__ == "__main__":
    main()
