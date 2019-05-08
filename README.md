# Lüscher n-dim test

Testing lattice Hamiltonian implementation for testing Lüscher's formula in 1, 2, 3 spatial dimension.

## ToDo

* Repeat contact computation for larger L (check Lüscher corrections)
* Repeat contact computation for non-zero scattering length
* Figure out which way to benchmark lattice parameters
* Make nice plot about commuting limits from FV discrete energy levels to IV continuum scattering (groups)
* Think about projectors

## Install
This is a `python` (preferably a new version of Python 3) project which uses `jupyter` notebooks.

The solver module wraps a C++ implementation of a sparse martix solver.
This depends on the libraries [`eigen`](https://github.com/eigenteam/eigen-git-mirror)  and [`spectra`](https://github.com/yixuan/spectra).

To tell the compiler where to find those libraries run
```bash
python setup.py configure --eigen <eigen> --spectra <spectra>
```

After configuration, the modules can be installed via pip:
```bash
pip install -e .
```

For the exact module versions see `requirements-freeze.txt`.



## Tests
Run
```bash
python setup.py test
```

## Run
Open the notebook in the `jupyter` dashboard and browse the notebooks directory.

## Authors
* [Christopher Körber](mailto:christopher@ckoerber.com)

## License
See [`LICENSE`](LICENSE)
