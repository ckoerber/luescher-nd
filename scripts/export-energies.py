#!/usr/bin/env python
# pylint: disable=C0103
"""Export three-dimensional fits to unitary limit
"""
# Data management
import numpy as np
import pandas as pd

# Fitting
import scipy.optimize as opt
from scipy.interpolate import interp1d

import luescher_nd.utilities as ut
from luescher_nd.zeta.zeta3d import Zeta3D


HBARC = 197.326  # MeV / fm
M_NUCLEON = (938.27 + 939.57) / 2 / HBARC  # in fm^{-1}


class Minimizer:  # pylint: disable=R0903
    """Class to simplify fitting of energy levels
    """

    def __init__(self, solver, L, epsilon, mu):
        self.solver = solver
        self.epsilon = epsilon
        self.L = L
        self.mu = mu
        self.zeta3d = Zeta3D(L, epsilon)

    def __call__(self, c0):
        """Compute difference of first energy level and expected energy squared.
        """
        return self.cost_function(c0)

    def effective_range(self, x):
        """
        """
        p2 = x / (self.L / 2 / np.pi) ** 2
        Lambda = np.pi / self.epsilon
        r0 = 4 / np.pi / Lambda
        P = -np.pi ** 2 / 96
        return r0 / 2 * p2 - P * r0 ** 3 * p2 ** 2

    def cost_function(self, c0: float, n_energies=2, weight_at_zero=10.0) -> float:
        r"""Computes $p \\cot(delta(p))$ at $p = 0$ by solving lattice Schrödinger eqn.

        Interpolates phase shift expansion for interacting states and returns the result
        of $p \\cot(delta(p))$ at $p = 0$.
        """
        energies = self.solver.get_energies(c0, n_energies=n_energies)
        x = energies * self.mu * self.L ** 2 / 2 / np.pi ** 2
        if not x[0] < 0:
            raise ValueError("Smallest energy not a bound state")

        if x[1] < 0:
            raise ValueError("Excited state is bound state")

        y = self.zeta3d(x) / np.pi / self.L
        y_intp = interp1d(x, y, kind=1)

        return np.sum(np.array([y[0], y_intp(0) * weight_at_zero, y[1]]) ** 2)


def main(L: int = 1.0):  # pylint: disable=R0914
    """Compute energy levels for different derivative implementations
    """
    epsilons = [L / 10, L / 15]  # , L / 20, L / 50]

    mu = M_NUCLEON / 2

    data = []
    df = pd.DataFrame()

    for nstep in range(1, 5):
        for epsilon in epsilons:
            print(f"[+] nstep = {nstep}, epsilon = {epsilon}")

            n1d_max = int(L / epsilon)
            L_eff = n1d_max * epsilon

            solver = ut.Solver(
                n1d_max=n1d_max,
                ndim_max=3,
                lattice_spacing=epsilon,
                derivative_shifts=ut.get_laplace_coefficients(nstep),
            )

            minimizer = Minimizer(solver, L_eff, epsilon, mu)
            res = opt.minimize(minimizer, -1.0, bounds=((-4.0, -1.0e-4),), method="TNC")

            energies = solver.get_energies(res.x, n_energies=min((n1d_max - 2) ** 2, 60))

            for nlevel, energy in enumerate(energies):
                data.append(
                    {
                        "epsilon": epsilon,
                        "c": res.x[0],
                        "energy": energy,
                        "nlevel": nlevel,
                        "nstep": nstep,
                        "n1d_max": n1d_max,
                    }
                )

            tf = pd.DataFrame(data)
            df = df.append(tf)
            df.to_csv("luescher-3d-res.csv", index=False)


if __name__ == "__main__":
    main()
