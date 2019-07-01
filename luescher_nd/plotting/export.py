"""Contains plot functions for effective range expansion data
"""
import os

import pandas as pd
import numpy as np

from luescher_nd.zeta.extern import pyzeta as pyz
from luescher_nd.zeta.zeta3d import DispersionZeta3d

from luescher_nd.plotting.utilities import plot_ere_grid
from luescher_nd.plotting.utilities import raw_data_to_plot_frame
from luescher_nd.plotting.utilities import monotonous_bounds


DATA = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "data"
    )
)


def dispersion_luescher_grid(fout: str):  # pylint: disable=R0914
    """Plots the effective range expansion for dispersion Lüscher data.

    Creates a grid plot and saves figure to `fout`.
    """
    fin = os.path.join(DATA, "luescher-3d-dispersion-final.csv")
    df = raw_data_to_plot_frame(
        pd.read_csv(fin, dtype={"nstep": "Int64"}), dispersion_zeta=True
    )

    grid = plot_ere_grid(df)

    x = np.linspace(-4, int(df.x.max()) + 1, 50000)
    L = df.L.round(7).unique()[0]

    nstep_map = {r"$\infty$": None}
    nstep_map.update({f"${nstep}$": nstep for nstep in df.nstep.unique() if nstep > 0})
    for (nrow, ncol, nhue), _ in grid.facet_data():
        n1d_max = grid.hue_names[nhue]
        nstep = nstep_map[grid.col_names[ncol]]
        if n1d_max != 20:
            continue
        epsilon = L / n1d_max
        zeta = DispersionZeta3d(L, epsilon, nstep)(x)
        for imin, imax in monotonous_bounds(zeta):
            grid.axes[nrow, ncol].plot(
                x[imin:imax], zeta[imin:imax], ls=":", lw=0.5, c="black", zorder=-1
            )

    grid.fig.set_dpi(300)
    grid.fig.set_figheight(2)
    grid.fig.set_figwidth(6)

    grid.fig.savefig(fout, bbox_inches="tight")


def standard_luescher_grid(fout: str):
    """Plots the effective range expansion for standard Lüscher data.

    Creates a grid plot and saves figure to `fout`.
    """
    fin = os.path.join(DATA, "luescher-3d-standard-final.csv")
    df = raw_data_to_plot_frame(
        pd.read_csv(fin, dtype={"nstep": "Int64"}), dispersion_zeta=False
    )

    grid = plot_ere_grid(df)

    x = np.linspace(-4, int(df.x.max()) + 1, 1000)
    zeta = pyz.zeta(x)  # pylint: disable=I1101

    for ax in np.array(grid.axes).flatten():
        for imin, imax in monotonous_bounds(zeta):
            ax.plot(x[imin:imax], zeta[imin:imax], ls=":", lw=0.5, c="black", zorder=-1)

    grid.fig.set_dpi(300)
    grid.fig.set_figheight(2)
    grid.fig.set_figwidth(6)

    grid.fig.savefig(fout, bbox_inches="tight")
