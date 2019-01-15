"""Install script for zeta function computations
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

EXTENSIONS = [
    Extension(
        "pyzeta", ["pyzeta.pyx"], language="c++", extra_compile_args=["--std=c++11"]
    )
]
setup(
    name="pyzeta",
    ext_modules=cythonize(EXTENSIONS),
    setup_requires=["cython", "numpy"],
    install_requires=["numpy"],
)
