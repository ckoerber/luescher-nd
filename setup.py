#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the luescher-nd
"""
from os import path

from setuptools import setup
from setuptools import Extension
from setuptools import find_packages

from Cython.Build import cythonize

__version__ = 0.1
__author__ = "Christopher Körber"

THISDIR = path.abspath(path.dirname(__file__))
with open(path.join(THISDIR, "README.md"), encoding="utf-8", mode="r") as f:
    README = f.read()

with open(path.join(THISDIR, "requirements.txt"), mode="r") as f:
    REQUIREMENTS = [el.strip() for el in f.read().split(",")]


EXTENSIONS = [
    Extension(
        "pyzeta",
        [path.join(THISDIR, "luescher_nd", "zeta", "pyzeta.pyx")],
        language="c++",
        extra_compile_args=["--std=c++11"],
        build_lib=path.join("luescher_nd", "zeta"),
    )
]

setup(
    name="luescher_nd",
    version=str(__version__),
    description="Lattice Hamiltonian in finte Volume for two particles",
    long_description=README,
    long_description_content_type="text/markdown",
    author=__author__,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    test_suite="tests",
    ext_modules=cythonize(EXTENSIONS),
    setup_requires=["cython", "numpy"],
)
