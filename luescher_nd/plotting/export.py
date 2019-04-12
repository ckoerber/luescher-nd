"""
"""
import os

import pandas as pd
import numpy as np

from luescher_nd.zeta.extern import pyzeta as pyz
from luescher_nd.zeta.zeta3d import Zeta3D

from luescher_nd.plotting.utilities import plot_ere_grid


DATA = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data"))


def dispersion_luescher_grid(fout: str):
    """
    """
    fin = os.path.join(DATA, "dispersion-luescher-res.csv")
    df = pd.read_csv(fin)

    grid = plot_ere_grid(df)

    x = np.linspace(-4, int(df.x.max()) + 1)

    for ax in np.array(grid.axes).flatten():
        L, epsilon, nstep = None, None, None
        zeta = Zeta3D(L, epsilon, nstep)(x)
        ax.plot(x, zeta, ls=":", lw=0.2)

    grid.fig.set_dpi(300)
    grid.fig.set_figheight(6)
    grid.fig.set_figwidth(6)

    grid.fig.savefig(fout, bbox_inches="tight")


def standard_luescher_grid(fout: str):
    """
    """
    fin = os.path.join(DATA, "standard-luescher-res.csv")
    df = pd.read_csv(fin)

    grid = plot_ere_grid(df)

    x = np.linspace(-4, int(df.x.max()) + 1)
    zeta = pyz.zeta(x)

    for ax in np.array(grid.axes).flatten():
        ax.plot(x, zeta, ls=":", lw=0.2)

    grid.fig.set_dpi(300)
    grid.fig.set_figheight(6)
    grid.fig.set_figwidth(6)

    grid.fig.savefig(fout, bbox_inches="tight")
