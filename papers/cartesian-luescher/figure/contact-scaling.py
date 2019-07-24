#!/usr/bin/env python3
"""Script for displaying the scaling of the fitted contact interaction compared to
prediction
"""


import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt

from luescher_nd.database.utilities import read_table
from luescher_nd.database.utilities import DATA_FOLDER

from luescher_nd.plotting.styles import setup
from luescher_nd.plotting.styles import EXPORT_OPTIONS
from luescher_nd.plotting.styles import MARKERS

FILE_NAME = "contact-fitted_a-inv=+0.0_zeta=dispersion_projector=a1g_n-eigs=200.sqlite"
ROUND_DIGITS = 1
FILTER_POLES = False
FILTER_BY_NSTATES = False
FILTER_DEGENERACY = False
ZETA = None

COUNTER_TERMS = {
    1: 1.010_924_039_434_65,
    2: 0.876_111_009_542_394,
    4: 0.819_780_003_837_739,
    -1: 0.777_551_349_695_929,
    "spherical": 2 / np.pi,
}

LOG_EPS_RANGE = np.linspace(-6, -1)


def nstep_label(nstep) -> str:
    return f"${nstep}$" if nstep > 0 else r"$\infty$"


def prop_df() -> pd.DataFrame:
    """Prepares data fram for plotting contact scaling grid
    """
    df = read_table(
        os.path.join(DATA_FOLDER, FILE_NAME),
        zeta=ZETA,
        round_digits=ROUND_DIGITS,
        filter_poles=FILTER_POLES,
        filter_by_nstates=FILTER_BY_NSTATES,
        filter_degeneracy=FILTER_DEGENERACY,
    )
    m = df.mass.unique()[0]

    group = df.groupby(["L", "epsilon", "nstep"])[["contact_strength"]]

    assert group.std().values.sum() == 0.0

    tf = group.mean().reset_index()

    tf.contact_strength *= -1

    tf["analytic"] = tf.apply(
        lambda row: 4 * row["epsilon"] / m / COUNTER_TERMS[row["nstep"]], axis=1
    )
    tf["diff"] = 100 * ((tf["analytic"] - tf["contact_strength"]) / tf["analytic"]).abs()
    tf["log2_eps"] = np.log2(tf["epsilon"])  # pylint: disable=E1111

    return (
        (
            tf.drop(columns="analytic")
            .set_index(["L", "log2_eps", "nstep"])
            .drop(columns=["epsilon"])
            .stack()
            .reset_index()
            .rename(columns={"level_3": "type", 0: "value"})
        ),
        m,
    )


def plot(x_label, y_label, **kwargs):
    x_shift = {1: -1.5, 2: -0.5, 4: 0.5, -1: 1.5}

    df = kwargs.pop("data")
    m = kwargs.pop("m")
    dtype = df["type"].unique()[0]
    nstep = df["nstep"].unique()[0]

    if dtype == "contact_strength":
        plt.plot(df[x_label], df[y_label], ls="None", **kwargs)
        plt.plot(
            LOG_EPS_RANGE,
            4 * 2 ** LOG_EPS_RANGE / m / COUNTER_TERMS[nstep],
            lw=1,
            ls="-",
            color=kwargs["color"],
            zorder=-1,
        )

        if nstep == -1:
            plt.plot(
                LOG_EPS_RANGE,
                4 * 2 ** LOG_EPS_RANGE / m / COUNTER_TERMS["spherical"],
                lw=1,
                ls="--",
                color="black",
                label="Spherical",
                zorder=-1,
            )
    else:
        width = 0.05
        plt.bar(
            df[x_label] + x_shift[nstep] * width,
            df[y_label],
            width=width,
            linewidth=1,
            facecolor=kwargs["color"],
            alpha=0.9,
        )


def export_grid_plot(df: pd.DataFrame, m: float):
    """
    """
    grid = sns.FacetGrid(
        data=df,
        col="L",
        row="type",
        hue="nstep",
        sharey="row",
        sharex=True,
        hue_kws={"marker": MARKERS},
        margin_titles=True,
        gridspec_kws={"height_ratios": [1.5, 1]},
        hue_order=[1, 2, 4, -1],
        xlim=(-6, -1.5),
        aspect=1.5,
    )

    grid.map_dataframe(plot, "log2_eps", "value", m=m)

    labels = {f"{n}": nstep_label(n) for n in [1, 2, 4, -1]}
    legend_data = {
        val: grid._legend_data[key]  # pylint: disable=W0212
        for key, val in labels.items()
    }
    legend_data[r"$\mathrm{Spherical}$"] = plt.plot(
        np.nan, np.nan, ls="--", color="black"
    )[0]
    legend_data[r"$\mathrm{Dispersion}$"] = plt.plot(
        np.nan, np.nan, ls="-", color="black"
    )[0]

    grid.axes[0, 1].legend(
        legend_data.values(),
        legend_data.keys(),
        frameon=False,
        title=r"$n_\mathrm{step}$",
        bbox_to_anchor=(1.1, 0.3),
        loc="upper left",
    )

    for ax in grid.axes[0]:
        ax.set_ylim(1.0e-2, 4.0e-1)

    for ax in grid.axes.flatten():
        ax.set_yscale("log")

    grid.axes[0, 0].set_ylabel(r"$-c(\epsilon) \, [\mathrm{fm}^{-2}]$")
    grid.axes[1, 0].set_ylabel(r"$|\Delta c(\epsilon)| \, [\%]$")

    for ax in grid.axes[1]:
        ax.set_xticks(range(-6, -1))
        ax.set_xticklabels([f"$2^{{{tick}}}$" for tick in ax.get_xticks()])

    grid.set_titles(
        col_template=r"${col_var} = {col_name} \, [\mathrm{{fm}}]$",
        row_template=" " * 100,
    )

    grid.set_xlabels(r"$ \epsilon \, [\mathrm{fm}]$")

    grid.savefig(
        f"contact-scaling-{FILE_NAME}".replace(".sqlite", ".pdf").replace("=", "_"),
        **EXPORT_OPTIONS,
    )


def main():
    """Export all the file options to pdf
    """
    setup()
    df, m = prop_df()
    export_grid_plot(df, m)


if __name__ == "__main__":
    main()
