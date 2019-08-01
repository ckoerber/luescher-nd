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
from luescher_nd.plotting.styles import finalize


# DATA_FOLDER = os.path.join(DATA_FOLDER, "three-d")

FILE_OPTIONS = [
    {
        "file_name": "contact-fitted_a-inv=-5.0_zeta=spherical_projector=a1g_n-eigs=200.sqlite",
        "zeta": "spherical",
        "a_inv": -5.0,
        "y_lim": (-15, 15),
    },
    {
        "file_name": "contact-fitted_a-inv=+0.0_zeta=spherical_projector=a1g_n-eigs=200.sqlite",
        "zeta": "spherical",
        "a_inv": 0,
        "y_lim": (-10, 20),
    },
    {
        "file_name": "contact-fitted_a-inv=-5.0_zeta=cartesian_projector=a1g_n-eigs=200.sqlite",
        "zeta": "cartesian",
        "a_inv": -5.0,
        "y_lim": (-10, 15),
    },
    {
        "file_name": "contact-fitted_a-inv=+0.0_zeta=cartesian_projector=a1g_n-eigs=200.sqlite",
        "zeta": "cartesian",
        "a_inv": 0,
        "y_lim": (-5, 20),
    },
    {
        "file_name": "contact-fitted_a-inv=-5.0_zeta=dispersion_projector=a1g_n-eigs=200.sqlite",
        "zeta": "dispersion",
        "a_inv": -5.0,
        "y_lim": None,
    },
    {
        "file_name": "contact-fitted_a-inv=+0.0_zeta=dispersion_projector=a1g_n-eigs=200.sqlite",
        "zeta": "dispersion",
        "a_inv": 0,
        "y_lim": None,
    },
]

ROUND_DIGITS = 1
FILTER_POLES = False
FILTER_BY_NSTATES = False
FILTER_DEGENERACY = True
Y_BOUNDS = (-500, 500)
FILTER = "n1d >= 10 and nstep != 1"


def nstep_label(nstep) -> str:
    return f"${nstep}$" if nstep > 0 else r"$\infty$"


def export_grid_plot(
    file_name: str, zeta: str = "spherical", a_inv: float = 0.0, y_lim=None
):
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
            filter_degeneracy=FILTER_DEGENERACY,
        )
        .sort_values("x")
        .query(FILTER)
    )
    df = df[df.epsilon * 1.0e4 % 2 == 0]

    title = f"{zeta.capitalize()} Lüscher"
    title += " fitted contact interaction"
    title += (
        " at unitarity"
        if abs(a_inv) < 1.0e-12
        else rf" at $-\frac{{1}}{{a_0}} = {a_inv:1.1f} \, [\mathrm{{fm}}^{{-1}}]$"
    )

    if y_lim:
        df.loc[df.y < y_lim[0], "y"] = y_lim[0] - 1
        df.loc[df.y > y_lim[1], "y"] = y_lim[1] + 1
    else:
        df["y"] = (df["y"] - a_inv).abs()

    df["nstep"] = df.nstep.apply(nstep_label)

    df["epsilon"] = df.epsilon.apply(lambda eps: f"{eps:0.3f}")

    grid = sns.FacetGrid(
        data=df,
        col="nstep",
        hue="epsilon",
        row="L",
        sharex=True,
        sharey=True,
        legend_out=True,
        hue_kws={"marker": MARKERS, "ms": [1] * 10, "lw": [0.5] * 10, "ls": [":"] * 10},
        margin_titles=True,
        col_order=[nstep_label(nstep) for nstep in [2, 4, -1]],
    )
    grid.map(plt.plot, "x", "y")

    for ax in grid.axes.flatten():
        ax.set_xlim(-1, 30)
        if y_lim is not None:
            ax.axhline(a_inv, color="black", lw=0.5, zorder=-1)
            ax.set_ylim(*y_lim)
        else:
            ax.set_yscale("log")
            ax.tick_params(axis="y", which="minor")

    grid.set_xlabels(r"$x = \frac{2 \mu E L^2}{4 \pi^2}$")
    if y_lim is not None:
        ylabels = r"$p \cot(\delta_0(p)) \,[\mathrm{fm}^{-1}]$"
    else:
        ylabels = (
            r"$\left| p \cot(\delta_0(p)) + \frac{1}{a_0} \right|\,[\mathrm{fm}^{-1}]$"
        )
    grid.set_ylabels()
    grid.set_titles(
        row_template=r"${row_var} = {row_name}\, [\mathrm{{fm}}]$",
        col_template=r"$n_{{\mathrm{{step}}}} =$ {col_name}",
    )

    # grid.fig.suptitle(title, y=1.05)

    plt.subplots_adjust(hspace=0.1)
    finalize(grid.fig)
    grid.add_legend(title=r"$\epsilon \,[\mathrm{fm}]$", frameon=False)

    grid.savefig(
        "ere-" + file_name.replace(".sqlite", ".pgf").replace("=", "_"), **EXPORT_OPTIONS
    )


def main():
    """Export all the file options to pdf
    """
    setup(pgf=True)

    LINE_STYLE["ls"] = "--"

    for options in FILE_OPTIONS:
        export_grid_plot(**options)


if __name__ == "__main__":
    main()
