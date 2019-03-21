# pylint: disable=C0103
"""Computation of 3D zeta function
"""
from typing import Optional

from dataclasses import dataclass
from dataclasses import field

from itertools import product

import numpy as np


@dataclass(frozen=True)
class Zeta3D:
    """Three dimensional Zeta function for discretized finite volume.
    """

    L: float
    epsilon: float
    nstep: Optional[int] = None

    _LB: float = field(
        repr=False, init=False, default=0.77755134969592873086316106575075368089
    )
    _normalization: float = field(repr=False, init=False, default=0.0)
    _n2: np.ndarray = field(repr=False, init=False, default=None)
    _multiplicity: np.ndarray = field(repr=False, init=False, default=None)

    @property
    def N(self) -> int:
        """Returns L/epsillon as an int
        """
        return int(self.L / self.epsilon)

    @property
    def lattice_modes(self) -> np.ndarray:
        """Returns all normalized momentum modes allowed on the lattice.

        The shape is [n1d**3, 3] where the second components access nx, ny, nz.
        """
        if self.nstep is None:
            lattice_modes = np.array(
                list(product(*[range(-self.N // 2 + 1, self.N // 2 + 1)] * 3))
            )
        else:
            lattice_modes = None

        return lattice_modes

    def __post_init__(self):
        """Initializes the elements of the sum for the zeta function denominator
        """
        all_vecs = {}
        for nx, ny, nz in self.lattice_modes:
            nr2 = nx ** 2 + ny ** 2 + nz ** 2
            all_vecs[nr2] = all_vecs.get(nr2, 0) + 1

        object.__setattr__(self, "_n2", np.array(list(all_vecs.keys())).reshape(-1, 1))
        object.__setattr__(
            self, "_multiplicity", np.array(list(all_vecs.values())).reshape(-1, 1)
        )
        object.__setattr__(
            self, "_normalization", self._LB * 2 * np.pi ** 2 * self.N / 2
        )

    def __call__(self, x: float):
        """Computes the result of the three-d Zeta function.
        """
        out = self._n2 - x
        out = np.where(np.abs(out) > 1.0e-12, out, np.NaN)
        out = self._multiplicity / out

        return np.sum(out, axis=0) - self._normalization
