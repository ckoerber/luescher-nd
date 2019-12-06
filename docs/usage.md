# How to user luescher-nd


## Export energies

The script which was used to generate the data is located `export-energies.py` at
```bash
cd luescher-nd/scripts/short-range
```

This script reads in a configuration file, e.g., `configuration.yaml`, fits the contact interaction, computes the resulting spectrum and writes energies to a database file.
It is adaptive in the sense that it only writes data which does not exists (with the exception that it only queries for `n1d=n1d, epsilon=epsilon, nstep=nstep` -- if the spectrum is not complete, it will still skip this computation.)

The script is run by
```bash
python export-energies.py configuration.yaml
```

An example configuration file is
```yaml
basis:             # Parameters for the basis
    epsilon:       # Lattice spacing in inverse [fm]
        - 0.250
        - 0.200
    L:             # Box length in inverse [fm]
        - 1.0
    nstep:         # Implementation of the derivative
        - 1
        - null     # n_step = infty derivative
eigenvalues:
    solver:        # Parameters passed to Lanczos solver
        k: 200     # number of eigenvalues
    projector:     # Parameters passed to eigenstate projector
        type: a1g         # Type of projector, can be parity or a1g
        cutoff: 3.0E+02   # By how much eigenvalues are shifted
physics:
    a_inv: 0.0     # Inverse scattering length in [fm]
zeta:              # Parameters for the zeta function
    kind: spherical  # kind of zeta function can be spherical, cartesian or dispersion
```


## Read in energies

See also the example notebook in `notebooks/example.ipynb` on how to read in and plot the data.

We also provide the scripts used for generating the figures in the paper at `paper/luescher-nd/figure` (see the `Makefile` for dependency trees).
