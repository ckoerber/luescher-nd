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

    def __init__(self, solver, L, epsilon, mu, ymax=1.0e3):
        self.solver = solver
        self.epsilon = epsilon
        self.L = L
        self.mu = mu
        self.ymax = ymax
        self.zeta3d = Zeta3D(L, epsilon)

    def __call__(self, c0):
        """Compute difference of first energy level and expected energy squared.
        """
        return self.p_cot_delta_at_0(c0) ** 2

    def p_cot_delta_at_0(self, c0: float) -> float:
        r"""Computes $p \\cot(delta(p))$ at $p = 0$ by solving lattice Schrödinger eqn.

        Interpolates phase shift expansion for interacting states and returns the result
        of $p \\cot(delta(p))$ at $p = 0$.
        """
        print("------")
        print(c0)
        energies = self.solver.get_energies(c0, n_energies=60)
        all_x = energies * self.mu * self.L ** 2 / 2 / np.pi ** 2
        unique_x = []
        for x in all_x:
            if unique_x:
                if np.min(np.abs(unique_x - x)) > 1.0e-2:
                    unique_x.append(x)
            else:
                unique_x.append(x)

        unique_x = np.array(unique_x)
        all_y = self.zeta3d(unique_x) / np.pi / self.L

        print(unique_x)
        print(all_y)

        interacting_x = []
        interacting_y = []
        for x, y in zip(unique_x, all_y):
            if x < 0 or abs(y) < self.ymax:
                interacting_x.append(x)
                interacting_y.append(y)

        print(np.array(interacting_x))
        print(np.array(interacting_y))
        y_intp = interp1d(np.array(interacting_x), np.array(interacting_y), kind=1)
        return y_intp(0)


def main(L: int = 1.0):  # pylint: disable=R0914
    """Compute energy levels for different derivative implementations
    """
    epsilons = [L / 10, L / 15, L / 20, L / 50]

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
            res = opt.minimize(minimizer, -0.01, bounds=((-1.0, -1.0e-4),), method="TNC")

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
