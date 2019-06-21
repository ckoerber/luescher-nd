# pylint: disable=C0103
"""Computation of 2D zeta function
"""
from typing import Optional

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from luescher_nd.utilities import get_laplace_coefficients

CATALAN = 0.91596559417721901505460351493238411077414937428167


@dataclass(frozen=True)
class DispersionZeta2D:
    """Two dimensional dispersion Zeta function for discretized finite volume.
    """

    _ndim = 2

    L: float
    epsilon: float
    nstep: Optional[int] = None

    _normalization: float = field(repr=False, init=False, default=0.0)
    _n2: np.ndarray = field(repr=False, init=False, default=None)
    _multiplicity: np.ndarray = field(repr=False, init=False, default=None)

    @property
    def N(self) -> int:
        """Returns L/epsillon as an int
        """
        return int(self.L / self.epsilon)

    def _get_n2(self) -> np.ndarray:
        """Returns all normalized momentum modes allowed on the lattice.

        The shape is [n1d**3, 3] where the second components access nx, ny, nz.
        """
        if self.nstep is None:
            n2_disp1d = np.arange(-self.N // 2 + 1, self.N // 2 + 1) ** 2
        else:
            p1d = np.arange(self.N) * 2 * np.pi / self.L

            coeffs = get_laplace_coefficients(self.nstep)
            norm = self.L ** 2 / self.epsilon ** 2 / 4 / np.pi ** 2

            n2_disp1d = np.sum(
                [
                    -cn * np.cos(n * p1d * self.epsilon) * norm
                    for n, cn in coeffs.items()
                ],
                axis=0,
            )

        return np.sum(
            [el.flatten() for el in np.meshgrid(*[n2_disp1d] * self._ndim)], axis=0
        )

    def __post_init__(self):
        """Initializes the elements of the sum for the zeta function denominator
        """
        all_vecs = {}
        for n2 in self._get_n2():
            all_vecs[n2] = all_vecs.get(n2, 0) + 1

        norms = {}

        object.__setattr__(self, "_n2", np.array(list(all_vecs.keys())).reshape(-1, 1))
        object.__setattr__(
            self, "_multiplicity", np.array(list(all_vecs.values())).reshape(-1, 1)
        )
        object.__setattr__(
            self, "_normalization", np.pi ** 2 * self.N * norms[self.nstep]
        )

        raise NotImplementedError("Need to implement dispersion Lüscher counter terms.")

    def __call__(self, x: float):
        """Computes the result of the three-d Zeta function.
        """
        out = self._n2 - x
        out = np.where(np.abs(out) > 1.0e-12, out, np.NaN)
        out = self._multiplicity / out

        return np.sum(out, axis=0) - self._normalization


@dataclass(frozen=True)
class Zeta2D:
    """Two dimensional dispersion Zeta function for discretized finite volume.
    """

    _ndim = 2

    N: int
    spherical: bool = False

    _normalization: float = field(repr=False, init=False, default=0.0)
    _n2: np.ndarray = field(repr=False, init=False, default=None)
    _multiplicity: np.ndarray = field(repr=False, init=False, default=None)

    def _get_n2(self) -> np.ndarray:
        """Returns all normalized momentum modes allowed on the lattice.

        The shape is [n1d**3, 3] where the second components access nx, ny, nz.
        """
        n2_disp1d = np.arange(-self.N // 2 + 1, self.N // 2 + 1) ** 2

        return np.sum(
            [el.flatten() for el in np.meshgrid(*[n2_disp1d] * self._ndim)], axis=0
        )

    def __post_init__(self):
        """Initializes the elements of the sum for the zeta function denominator
        """
        all_vecs = {}
        for n2 in self._get_n2():
            all_vecs[n2] = all_vecs.get(n2, 0) + 1

        object.__setattr__(self, "_n2", np.array(list(all_vecs.keys())).reshape(-1, 1))
        object.__setattr__(
            self, "_multiplicity", np.array(list(all_vecs.values())).reshape(-1, 1)
        )
        object.__setattr__(
            self,
            "_normalization",
            2 * np.pi * np.log(self.N)
            if self.spherical
            else 2 * np.pi * np.log(self.N) - 4 * (CATALAN - np.pi / 8),
        )

    def __call__(self, x: float):
        """Computes the result of the three-d Zeta function.
        """
        out = self._n2 - x
        out = np.where(np.abs(out) > 1.0e-12, out, np.NaN)
        out = self._multiplicity / out

        return np.sum(out, axis=0) - self._normalization
