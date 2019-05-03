#!/usr/bin/env python
# pylint: disable=C0103
"""Script to compute discrete finite volume energy from effective range expansion
"""
from typing import Optional

from dataclasses import dataclass
from dataclasses import field

import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

from scipy.optimize import minimize_scalar

from luescher_nd.zeta.zeta3d import Zeta3D
from luescher_nd.zeta.extern.pyzeta import zeta  # pylint: disable=E0611

from luescher_nd.hamiltonians.longrange import MU, M0, GBAR0, HBARC, E0
from luescher_nd.hamiltonians.longrange import p_cot_delta

NSTEP = None

LRANGE = np.linspace(2, 40)


@dataclass(frozen=True)
class MinimizationKernel:
    """Class helps minimizing difference between zeta function and ERE.
    """

    l: float
    n1d: Optional[int] = None
    nstep: Optional[int] = NSTEP
    mu: float = MU
    m: float = M0
    gbar: float = GBAR0
    _zeta: float = field(init=False, repr=False)

    def __post_init__(self):
        """Sets zeta function
        """
        if self.n1d is not None:
            object.__setattr__(self, "_zeta", Zeta3D(self.l, self.epsilon, self.nstep))
        else:
            object.__setattr__(self, "_zeta", zeta)

    @property
    def epsilon(self) -> Optional[float]:
        """Returns lattice spacing"""
        return self.l / self.n1d if self.n1d else None

    def zeta(self, x: float) -> float:
        """Returns zeta function
        """
        return self._zeta(x)  # pylint: disable=E1102

    def p_cot_delta(self, x):
        """Returns analytic effective range expansion
        """
        p = np.sqrt(x + 0j) * 2 * np.pi / self.l
        return p_cot_delta(p, self.gbar, self.mu, self.m).real

    def __call__(self, x: float) -> float:
        """Returns absolute value of ERE minus S3 (dispersion/standard depending on n1d)
        """
        return np.abs(self.p_cot_delta(x) - self.zeta(x) / np.pi / self.l)


def main():
    """Prints finite volume energies from ERE.
    """
    fig, ax = plt.subplots(dpi=250)
    ax.axhline(E0 * HBARC)
    for epsilon in [1.0, 0.5, 0.4]:
        y = []
        for l in LRANGE:
            n1d = int(l / epsilon)
            kernel = MinimizationKernel(l, n1d=n1d, nstep=NSTEP)

            x = minimize_scalar(kernel, (-10, -1.0e-2)).x[0]
            elat = x * (2 * np.pi / kernel.l) ** 2 / 2 / kernel.mu

            y.append(elat * HBARC)

        ax.plot(LRANGE, y, ".", ms=1, label=rf"$\epsilon = {epsilon}$ [fm]")

    ax.set_ylabel("$E_0$ [MeV]")
    ax.set_xlabel("$L$ [fm]")
    ax.legend()
    sns.despine()
    plt.show(fig)


if __name__ == "__main__":
    main()
