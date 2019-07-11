"""Implementation of kinetic Hamiltonians
"""
from typing import Optional
from typing import Dict
from typing import Any

from dataclasses import dataclass
from dataclasses import field

import logging
import os

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

from luescher_nd.utilities import get_logger
from luescher_nd.utilities import get_laplace_coefficients

from luescher_nd.database.connection import DatabaseSession

from luescher_nd.database.tables import create_database

LOGGER = get_logger(logging.INFO)

HBARC = 197.327


def get_kinetic_hamiltonian(  # pylint: disable=R0914, R0913
    n1d_max: int,
    lattice_spacing: float = 1.0,
    particle_mass: float = 4.758,
    ndim_max: int = 3,
    derivative_shifts: Optional[Dict[int, float]] = None,
) -> sp.csr_matrix:
    r"""Computes the kinetic Hamiltonian for a relative two-body system in a 3D box.

    The kinetic Hamiltonian is defined by
        `H0 = - Laplace / 2 /(particle_mass/2)`
    Where Laplace is the lattice laplace operator with twisted boundary conditions.

    The function uses scipy sparse matrices in column format.

    Arguments
    ---------
        n1d_max: int
            Spatial dimension of lattice in any direction.

        lattice_spacing: float
            The lattice spacing in fermi.

        particle_mass: float
            The mass of the particles to be described in inverse fermi.

        ndim_max: float
            The number of spatial dimensions.

        derivative_shifts: Dict[int, float] or None
            The implementation of the numerical laplace operator.
            E.g., for a one step derivative
                `d2f/dx2(n) = sum(
                    c_i * f(n + s_i) * exp(1j * twist_angle/n1d_max * s_i)
                )/lattice_spacing**2`
            where s_i and c_i are the keys and values of derivative_shifts.
            If None, the shifts are defaulted to {-1: 1.0, 0: -2.0, 1: 1.0}.

    Long description
    ----------------
    In the one-dimensional case, a simple discrete 1-step derivative for periodic
    boundary conditions is given by
    $$
        \frac{d^2 f(x = n \epsilon)}{dx^2}
        \rightarrow \frac{1}{\epsilon^2}
        \left[
            f(n\epsilon + \epsilon)
            - 2 f(n\epsilon)
            + f(n\epsilon - \epsilon)
        \right] \, ,
    $$
    where $\epsilon$ is the lattice spacing.

    For improved discrete derivative implementations (n-steps) see also
    [Finite difference coefficients]
    (https://en.wikipedia.org/wiki/Finite_difference_coefficient).

    The kinetic Hamiltonian in relative coordinates is computed by dividing the above
    expression by two times the reduced mass of the subsystem &mdash;
    in the case of equal mass particles, half the mass of individual particles.

    The below implementation of the kinetic Hamiltonian is general in the number of
    dimensions and the implementation of the derivative.
    This is achieved by creating a meta index for coordinates:
    ```python
    nr = n1 + n2 * n1d_max ** 1 + n3 * n1d_max ** 2 + ...
    ```
    where `ni` describe the index in each respective dimension.

    Taking the derivative in a respective direction adjusts the `ni` components.
    Because the grid is finite with boundary conditions, all `ni` are identified with
    their `n1d_max` modulus.

    For example, when shifitng the y-coordinate in a three dimensional space by one
    lattice spacing becomes
    ```
    nr_yshift = nx + ((ny + 1) % n1d_max) * n1d_max ** 1 + nz * n1d_max ** 2
    ```

    The below kinetic Hamiltonian iterates over all dimensions and shifts to create
    a `n_max x n_max` matrix with `n_max = n1d_max ** ndim_max`.
    """
    derivative_shifts = derivative_shifts or {-1: 1.0, 0: -2.0, 1: 1.0}

    data = sp.lil_matrix((n1d_max ** ndim_max, n1d_max ** ndim_max), dtype=float)

    h0_fact = 1 / lattice_spacing ** 2 / 2 / (particle_mass / 2)

    modified_shifts = {
        shift: coeff * h0_fact for shift, coeff in derivative_shifts.items()
    }

    for nr in range(n1d_max ** ndim_max):
        nr_subtracted = nr
        n1d_pow_ndim = 1

        for _ in range(ndim_max):
            nxi = (nr_subtracted % (n1d_pow_ndim * n1d_max)) // n1d_pow_ndim
            nr_subtracted -= nxi * n1d_pow_ndim

            for shift, coeff in modified_shifts.items():
                nr_shift = nr + n1d_pow_ndim * (
                    -nxi + (nxi + shift + n1d_max * 10) % n1d_max
                )
                data[(nr, nr_shift)] -= coeff

            n1d_pow_ndim *= n1d_max

    return data.tocsr()


@dataclass(frozen=True)
class MomentumKineticHamiltonian:
    """Kinetic Hamiltonian in momentum space

    **Attributes**
        n1d: int
            Number of lattice sites in one dimension

        epsilon: float = 1.0
            Lattice spacing in inverse fermi

        mass: float = 4.758
            Particle mass in inverse fermi. Hamiltonina uses reduced mass ``mu = m /2``

        ndim: int = 3
            Number of dimensions

        nstep: Optional[int] = 3
            Number of derivative steps. None corresponds to inifinte step derivative.

        filter_out: Optional[sp.csr_matrix] = None
            Projection operator which must commute with the Hamiltonian.
            If specified, applies ``H + cutoff * filter_out`` instead of ``H``.
            Thus, eigenstates with ``filter_out |psi> = |psi>`` are projected to larger
            energies.

        filter_cutoff: Optional[float] = None
            Size of the cutoff filter.
    """

    _table_class = None  # Must be an energy entry

    n1d: int
    epsilon: float = 1.0
    mass: float = 4.758
    ndim: int = 3
    nstep: Optional[int] = 3
    filter_out: Optional[sp.csr_matrix] = field(default=None, repr=False)
    filter_cutoff: Optional[float] = field(default=None, repr=False)

    _disp_over_m: np.ndarray = field(init=False, repr=False)
    _op: LinearOperator = field(default=None, repr=False)

    @property
    def L(self):  # pylint: disable=C0103
        """Lattice spacing time nodes in one direction
        """
        return self.epsilon * self.n1d

    @property
    def p2(self):  # pylint: disable=C0103
        """Returns momentum dispersion
        """
        return self._disp_over_m * 2 * (self.mass / 2)

    def __post_init__(self):
        """Initializes the dispersion relation the matvec kernel.
        """
        if self.nstep is None:
            p1d = (
                np.array(
                    list(range(0, (self.n1d + 1) // 2))
                    + list(range(self.n1d // 2, 0, -1))
                )
                * 2
                * np.pi
                / self.L
            )
            disp1d = p1d ** 2
        else:
            coeffs = get_laplace_coefficients(self.nstep)
            p1d = np.arange(self.n1d) * 2 * np.pi / self.L
            disp1d = np.sum(
                [
                    -cn * np.cos(n * p1d * self.epsilon) / self.epsilon ** 2
                    for n, cn in coeffs.items()
                ],
                axis=0,
            )

        disp = np.sum(np.array(np.meshgrid(*[disp1d] * self.ndim)), axis=0).flatten()

        object.__setattr__(self, "_disp_over_m", disp / 2 / (self.mass / 2))

    @property
    def op(self) -> LinearOperator:  # pylint: disable=C0103
        """The matvec kernel of the Hamiltonian
        """
        if not self._op:
            object.__setattr__(
                self,
                "_op",
                LinearOperator(matvec=self._apply, shape=[self.n1d ** self.ndim] * 2),
            )
        return self._op

    def _apply(self, vec):
        """Wraps projector application with operator application

        Computes ``(H + cutoff P) |psi>``. Thus states with ``P|psi> = |psi>`` are
        shifted to larger energies while states with ``P|psi> = 0`` remain.
        Note this only works if ``[H, P] = 0``.
        """
        out = self.apply(vec)
        if self.filter_out is not None and self.filter_cutoff is not None:
            out += self.filter_cutoff * self.filter_out @ vec
        return out

    def apply(self, vec):
        """Applies hamiltonian to vector
        """
        return self._disp_over_m * vec

    def export_eigs(  # pylint: disable=R0914
        self,
        database: str,
        overwrite: bool = False,
        eigsh_kwargs: Optional[Dict[str, Any]] = None,
        export_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Computes eigenvalues of hamiltonian and writes to database.

        First checks if values are already present. If true, does nothing.
        However, it does not check how many levels are present.
        So if you want to compute more levels, set ``overwrite`` to ``True``.

        **Arguments**
            database: str
                Database name for export / import.

            overwrite: bool = False
                Overwrite existing entries.

            eigsh_kwargs: Optional[Dict[str, Any]] = None
                Arguments fed to ``scipy.sparse.linalg.eigsh``.
                Default are ``return_eigenvectors=False`` and ``which="SA"``.
                These cannot be changed.

            export_kwargs: Optional[Dict[str, Any]] = None
                Arguments fed to ``table.get_or_create``.
                This does not overwrite properties of the hamiltonian.
        """
        if self._table_class is None:
            raise KeyError(f"Unkown table entry for {self}. Cannot export to database.")

        eigsh_kwargs = eigsh_kwargs or {}

        eigsh_kwargs["return_eigenvectors"] = False
        eigsh_kwargs["which"] = "SA"
        if "k" not in eigsh_kwargs:
            eigsh_kwargs["k"] = 6

        if not os.path.exists(database):
            LOGGER.info("Creating database `%s`", database)
            create_database(database)

        export_kwargs = export_kwargs or {}
        export_kwargs.update(
            {
                key: getattr(self, key)
                for key in self._table_class.keys()
                if not key in ["E", "nlevel"]
            }
        )

        with DatabaseSession(database, commit=False) as sess:
            matches = sess.query(self._table_class).filter_by(**export_kwargs).all()
            if matches and not overwrite:
                LOGGER.debug("Found %d entries for `%s`. Skip.", len(matches), self)
                return

        LOGGER.info("Exporting eigenvalues of `%s`", self)

        if self.filter_out is not None and self.filter_cutoff is not None:
            eigsh_kwargs["return_eigenvectors"] = True
            eigs, vecs = eigsh(self.op, **eigsh_kwargs)
            vecs = vecs.T
        else:
            eigs = np.sort(eigsh(self.op, **eigsh_kwargs))
            vecs = None

        n_created = 0
        with DatabaseSession(database) as sess:
            data = export_kwargs.copy()
            for nlevel, eig in enumerate(eigs):

                if vecs is not None:
                    f_res = vecs[nlevel] @ (self.filter_out @ vecs[nlevel])
                    LOGGER.debug("%d -> %1.3f", nlevel, 2 * f_res - 1)
                    if abs(f_res) > 1.0e-12:
                        # only consider states which have zero projection
                        LOGGER.debug("Ignoring energy value E%d", nlevel)
                        continue

                data.update({"E": eig, "nlevel": nlevel})
                entry, created = self._table_class.get_or_create(session=sess, **data)
                if created:
                    LOGGER.debug("Created `%s`", entry)
                    n_created += 1

        LOGGER.info("\tExported %d entries", n_created)

    @classmethod
    def exists_in_db(cls, database: str, **export_kwargs) -> bool:
        """Filters the data base table if entries are present.

        **Arguments**
            database: str
                Address of the database.

            export_kwargs: Dict[str, Any]
                Columns to filter (equal).
        """
        with DatabaseSession(database, commit=False) as sess:
            exists = (
                sess.query(cls._table_class).filter_by(**export_kwargs).first()
                is not None
            )

        return exists
