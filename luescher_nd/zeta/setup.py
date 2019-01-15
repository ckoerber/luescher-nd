"""Install script for zeta function computations
"""
from setuptools import Extension
from setuptools import setup

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
