"""Computation of 3D zeta function
"""
from itertools import product

import numpy as np


class Zeta3D:
    """
    """

    LB = 0.77755134969592873086316106575075368089

    def __init__(self, L, epsilon):
        self.L = L
        self.epsilon = epsilon
        self.N = int(L / epsilon)

        all_vecs = {}
        for nx, ny, nz in product(*[range(-self.N // 2 + 1, self.N // 2 + 1)] * 3):
            nr2 = nx ** 2 + ny ** 2 + nz ** 2
            all_vecs[nr2] = all_vecs.get(nr2, 0) + 1

        self.n2 = np.array(list(all_vecs.keys())).reshape(-1, 1)
        self.multiplicity = np.array(list(all_vecs.values())).reshape(-1, 1)

    def __call__(self, x: float):
        """
        """
        out = self.n2 - x
        out = np.where(np.abs(out) > 1.0e-12, out, np.NaN)
        out = self.multiplicity / out

        return np.sum(out, axis=0) - self.LB * 2 * np.pi ** 2 * self.N / 2
