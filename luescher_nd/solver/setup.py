"""Install script for zeta function computations
"""
from setuptools import Extension
from setuptools import setup

from Cython.Build import cythonize

EXTENSIONS = [
    Extension(
        "solver",
        ["solver.py.pyx"],
        language="c++",
        extra_compile_args=["--std=c++11"],
        include_dirs=[
            "/usr/local/include/eigen3",
            "/Users/christopherkorber/local/spectra/include",
        ],
    )
]

setup(
    name="solver",
    ext_modules=cythonize(EXTENSIONS),
    setup_requires=["cython", "numpy"],
    install_requires=["numpy"],
)
