# pylint: disable=C0103
"""Computation of 3D zeta function
"""
from typing import Optional

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from luescher_nd.utilities import get_laplace_coefficients


@dataclass(frozen=True)
class Zeta3D:
    """Three dimensional Zeta function for discretized finite volume.
    """

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

        return np.sum([el.flatten() for el in np.meshgrid(*[n2_disp1d] * 3)], axis=0)

    def __post_init__(self):
        """Initializes the elements of the sum for the zeta function denominator
        """
        all_vecs = {}
        for n2 in self._get_n2():
            all_vecs[n2] = all_vecs.get(n2, 0) + 1

        norms = {
            1: 1.01085,
            2: 0.876033,
            3: 0.83731,
            4: 0.819702,
            5: 0.80982,
            6: 0.803554,
            7: 0.799252,
            8: 0.796127,
            9: 0.793761,
            10: 0.79191,
            11: 0.790426,
            12: 0.78921,
            13: 0.788196,
            14: 0.787339,
            15: 0.786606,
            16: 0.785971,
            17: 0.785416,
            18: 0.784928,
            19: 0.784495,
            20: 0.784108,
            21: 0.783761,
            22: 0.783447,
            23: 0.783162,
            24: 0.782903,
            25: 0.782666,
            26: 0.782448,
            27: 0.782248,
            28: 0.782062,
            29: 0.78189,
            30: 0.781731,
            31: 0.781582,
            32: 0.781443,
            33: 0.781313,
            34: 0.781191,
            35: 0.781076,
            36: 0.780969,
            37: 0.780867,
            38: 0.780771,
            39: 0.78068,
            40: 0.780594,
            41: 0.780512,
            42: 0.780435,
            43: 0.780361,
            44: 0.78029,
            45: 0.780223,
            46: 0.78016,
            47: 0.780098,
            48: 0.78004,
            49: 0.779984,
            50: 0.779931,
            None: 0.77755134969592873086316106575075368089,
        }

        object.__setattr__(self, "_n2", np.array(list(all_vecs.keys())).reshape(-1, 1))
        object.__setattr__(
            self, "_multiplicity", np.array(list(all_vecs.values())).reshape(-1, 1)
        )
        object.__setattr__(
            self, "_normalization", np.pi ** 2 * self.N * norms[self.nstep]
        )

    def __call__(self, x: float):
        """Computes the result of the three-d Zeta function.
        """
        out = self._n2 - x
        out = np.where(np.abs(out) > 1.0e-12, out, np.NaN)
        out = self._multiplicity / out

        return np.sum(out, axis=0) - self._normalization
