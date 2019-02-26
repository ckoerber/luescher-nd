# pylint: disable=C0103
"""Export three-dimensional fits to unitary limit
"""
# Data management
import numpy as np
import pandas as pd

# Fitting
import scipy.optimize as opt

import luescher_nd.utilities as ut


HBARC = 197.326  # MeV / fm
M_NUCLEON = (938.27 + 939.57) / 2 / HBARC  # in fm^{-1}


class Minimizer:  # pylint: disable=R0903
    """Class to simplify fitting of energy levels
    """

    def __init__(self, e0, solver):
        self.e0 = e0
        self.solver = solver

    def func(self, c0):
        """Compute difference of first energy level and expected energy squared.
        """
        return (self.solver.get_energies(c0, n_energies=1) - self.e0) ** 2


def main(L: int = 1.0):  # pylint: disable=R0914
    """Compute energy levels for different derivative implementations
    """
    x0 = -0.096_124_05

    epsilons = [0.1]  # 0.08, 0.05, 0.02, 0.01]

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

            e0 = x0 * 2 * np.pi ** 2 / mu / L_eff ** 2

            minimizer = Minimizer(e0, solver)
            res = opt.minimize(minimizer.func, -1.0)

            energies = solver.get_energies(
                res.x, n_energies=min((n1d_max - 2) ** 2, 60)
            )

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
