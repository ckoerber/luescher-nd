#!/usr/bin/env python3
"""Plots momentum dispersion relation
"""
import numpy as np
import matplotlib.pyplot as plt

from luescher_nd.plotting.styles import setup
from luescher_nd.plotting.styles import finalize
from luescher_nd.plotting.styles import LEGEND_STYLE
from luescher_nd.plotting.styles import EXPORT_OPTIONS
from luescher_nd.plotting.styles import LINE_STYLE


def nstep1(pe):
    return 2 - 2 * np.cos(pe)


def nstep2(pe):
    return 5 / 2 - 8 / 3 * np.cos(pe) + 1 / 6 * np.cos(2 * pe)


def nstep3(pe):
    return 49 / 18 - 3 * np.cos(pe) + 3 / 10 * np.cos(2 * pe) - 1 / 45 * np.cos(3 * pe)


def nstep4(pe):
    return (
        205 / 72
        - 16 / 5 * np.cos(pe)
        + 2 / 5 * np.cos(2 * pe)
        - 16 / 315 * np.cos(3 * pe)
        + 1 / 280 * np.cos(4 * pe)
    )


def nstepinf(pe):
    return pe ** 2


DISPERSION = {1: nstep1, 2: nstep2, 3: nstep3, 4: nstep4, -1: nstepinf}

LABEL = {
    -1: r"$n_{\mathrm{step}}=\infty$",
    1: r"$n_{\mathrm{step}}=1$",
    2: r"$n_{\mathrm{step}}=2$",
    3: r"$n_{\mathrm{step}}=3$",
    4: r"$n_{\mathrm{step}}=4$",
}


def main():
    setup(n_colors=len(LABEL))
    fig, ax = plt.subplots()

    fig.set_figwidth(fig.get_figwidth() / 2)
    fig.set_figheight(fig.get_figheight() / 2)

    # ax.set_title("Lattice Dispersion Relations")
    ax.set_xlabel(r"$p\epsilon$")
    ax.set_ylabel(r"$2 \mu E \epsilon^2$")
    ax.set_ylim(0, 10)
    ax.set_xlim(-np.pi, np.pi)

    xticks = np.pi * np.array([-1, -1 / 2, 0, 1 / 2, 1])
    xticklabels = np.array(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$", r"$+\pi$"]
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    pe = np.linspace(-np.pi, +np.pi, 1000)

    for d, disp in DISPERSION.items():
        ax.plot(pe, disp(pe), label=LABEL[d], **LINE_STYLE)

    ax.legend(**LEGEND_STYLE)

    finalize()

    fig.savefig("./dispersion.pdf", **EXPORT_OPTIONS)


if __name__ == "__main__":
    main()
