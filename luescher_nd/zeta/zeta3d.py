# pylint: disable=C0103
"""Computation of 3D zeta function
"""
from typing import Optional

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from luescher_nd.utilities import get_laplace_coefficients


@dataclass(frozen=True)
class DispersionZeta3d:
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
            1: 1.01092403943465,
            2: 0.876111009542394,
            3: 0.837387860316806,
            4: 0.819780003837739,
            5: 0.809897923441695,
            6: 0.803632303522803,
            7: 0.799330009223405,
            8: 0.796205142152675,
            9: 0.793838821006476,
            10: 0.791988326687533,
            11: 0.790503744465088,
            12: 0.789287691287492,
            13: 0.788274270126898,
            14: 0.787417357563582,
            15: 0.786683739985032,
            16: 0.786048913637068,
            17: 0.785494418656777,
            18: 0.785006092573360,
            19: 0.784572894336584,
            20: 0.784186093200921,
            21: 0.783838697216824,
            22: 0.783525042827789,
            23: 0.783240495095067,
            24: 0.782981225338827,
            25: 0.782744043889144,
            26: 0.782526272681728,
            27: 0.782325647072780,
            28: 0.782140239360991,
            29: 0.781968398629612,
            30: 0.781808702994389,
            31: 0.781659921378535,
            32: 0.781520982673569,
            33: 0.781390950676730,
            34: 0.781269003583586,
            35: 0.781154417100468,
            36: 0.781046550454205,
            37: 0.780944834736610,
            38: 0.780848763142425,
            39: 0.780757882752073,
            40: 0.780671787581956,
            41: 0.780590112680369,
            42: 0.780512529090365,
            43: 0.780438739534904,
            44: 0.780368474706527,
            45: 0.780301490065182,
            46: 0.780237563065005,
            47: 0.780176490744592,
            48: 0.780118087626490,
            49: 0.780062183880675,
            50: 0.780008623714170,
            None: 0.777551349695929,
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


@dataclass(frozen=True)
class Zeta3D:
    r"""Three dimensional Zeta function for discretized finite volume.

    This class implements
    $$
    S_3^{A}(x; N)
    =
    \\sum_{n_i \\in M^A(N)} \\frac{1}{\\vec{n}^2 - x}
    - 2 \\pi^2 N \\mathcal L_A
    $$
    where $N = \\Lambda L / (2 \\pi)$ is the cutoff of the zeta function, $A$ means
    either spherical or cartesian
    $$
    M^A(N) = \\begin{cases}
        \\left{
            (n_1, n_2) \\in \\mathbb Z^3
        \\middle\\vert
            -N \\leq n_i < N
        \\right}
        & A = \\text{cartesian} \\
        \\left{
            (n_1, n_2) \\in \\mathbb Z^3
        \\middle\\vert
            n_1^2 + n_2^2 < N
        \\right}
        & A = \\text{spherical}
    \\end{cases}
    $$
     and
    $\\mathcal L_\\text{spherical} = \\frac{2}{\\pi}$ but
    $\\mathcal L_\\text{cartesian} = 0.77755134963633393039$.
    """

    _ndim = 3
    _norm_const = {True: 2 / np.pi, False: 0.77755134963633393039}

    N: int
    spherical: bool = False

    _normalization: float = field(repr=False, init=False, default=0.0)
    _n2: np.ndarray = field(repr=False, init=False, default=None)
    _multiplicity: np.ndarray = field(repr=False, init=False, default=None)

    def _get_n2(self) -> np.ndarray:
        r"""Returns all normalized momentum modes allowed on the lattice.

        This is the list of all $n^2 = n_1^2 + n_2^2$ with
            * $\\Lambda \leq n_i < \\Lambda$ (cartesian)
            * $ $n^2 < \\Lambda$ (spherical)
        """
        n2_disp1d = np.arange(-self.N + 1, self.N + 1) ** 2

        n2 = np.sum(
            [el.flatten() for el in np.meshgrid(*[n2_disp1d] * self._ndim)], axis=0
        )
        if self.spherical:
            n2 = np.array([el for el in n2 if el < self.N ** 2])
        return n2

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
            self, "_normalization", 2 * np.pi * self.N * self._norm_const[self.spherical]
        )

    def __call__(self, x: float):
        """Computes the result of the three-d Zeta function.
        """
        out = self._n2 - x
        out = np.where(np.abs(out) > 1.0e-12, out, np.NaN)
        out = self._multiplicity / out

        return np.sum(out, axis=0) - self._normalization
