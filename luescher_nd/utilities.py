# pylint: disable=E1101
"""Module for computing expectation values of kinetic hamiltonian in n dimensions.
"""
import logging

import itertools

import numpy as np
from scipy.special import gamma as gammafunc

from sympy import Matrix
from sympy.functions.combinatorial.factorials import factorial as analytic_factorial


def get_logger(level):
    """Creates a logger a given level
    """
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(name)s|%(asctime)s] %(message)s")
    ch.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level)
        logger.addHandler(ch)
    return logger


LOGGER = get_logger(logging.INFO)


def factorial(n: np.ndarray) -> np.ndarray:  # pylint: disable=C0103
    """Computes factorial of n
    """
    return gammafunc(n + 1)


def get_laplace_coefficients(n_step: int, analytic: bool = False):
    """Computes the derivative coefficient for lapace operator up to step range Nstep.

    The dispersion of the related momentum squared operator is equal to `p ** 2` up to
    order `p ** (2 * NStep)`.

    **Arguments**
        n_step: int
            The number of momentum steps in each direction.

        analytic: bool = False
            Solves analytically for coefficients and converts result to floats
            afterwards. Takes more time to compute coefficients.
    """
    v = np.zeros(n_step + 1, dtype=int)
    v[1] = 1

    if analytic:
        A = Matrix([[0] * v.size] * v.size)
        v = Matrix(v)
        for n, m in itertools.product(*[range(n_step + 1)] * 2):
            A[m, n] = 1 / analytic_factorial(2 * m) * (-1) ** m * n ** (2 * m)

        gamma = np.array(A.LUsolve(v)).flatten().astype(float)

    else:
        nn, mm = np.meshgrid(*[np.arange(n_step + 1)] * 2)
        A = 1 / factorial(2 * mm) * (-1) ** mm * nn ** (2 * mm)

        gamma = np.linalg.solve(A, v)

    coeffs = {}
    for n, coeff in enumerate(gamma):
        if n == 0:
            coeffs[n] = -coeff
        else:
            coeffs[+n] = -coeff / 2
            coeffs[-n] = -coeff / 2

    return coeffs
