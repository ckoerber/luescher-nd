#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the luescher-nd
"""
from os import path

from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
from setuptools import Command

import yaml

from Cython.Build import cythonize

__version__ = 0.1
__author__ = "Christopher Körber"

THISDIR = path.abspath(path.dirname(__file__))
with open(path.join(THISDIR, "README.md"), encoding="utf-8", mode="r") as f:
    README = f.read()

with open(path.join(THISDIR, "requirements.txt"), mode="r") as f:
    REQUIREMENTS = [el.strip() for el in f.read().split(",")]


CONFIGURE_FILE = "configure.yaml"


class Configure(Command):
    """Custom install command"""

    description = "Sets up path for eigen and spectra."

    eigen = path.abspath(path.join("/", "usr", "local", "include", "eigen3"))
    spectra = path.abspath(path.join("/", "usr", "local", "include", "spectra"))

    user_options = [
        ("eigen=", "e", "Eigen include directory"),
        ("spectra=", "s", "Spectra include directory"),
    ]

    def initialize_options(self):
        """Default directories to look for install.
        """
        data = None
        if path.exists(CONFIGURE_FILE):
            with open(CONFIGURE_FILE, "r") as inp:
                data = yaml.load(inp)

        if data:
            self.eigen = data["eig_dir"] if "eig_dir" in data else self.eigen
            self.spectra = data["spec_dir"] if "spec_dir" in data else self.spectra

    def finalize_options(self):
        """Checks if directories exist
        """

        if not path.exists(self.eigen):
            raise ValueError("Could not locate eigen dir")

        if not path.exists(self.spectra):
            raise ValueError("Could not locate spectra dir")

    def run(self):
        """Exports configure options to file
        """
        data = {"eigen": self.eigen, "spectra": self.spectra}
        with open(CONFIGURE_FILE, "w") as out:
            yaml.dump(data, out, default_flow_style=False)


INC_DIRS = [Configure.eigen, Configure.spectra]
if path.exists(CONFIGURE_FILE):
    with open(CONFIGURE_FILE, "r") as INP:
        DATA = yaml.load(INP)
    if DATA:
        if "eigen" in DATA and "spectra" in DATA:
            INC_DIRS = DATA.values()

EXTENSIONS = [
    Extension(
        path.join("luescher_nd", "zeta", "pyzeta").replace("/", "."),
        [path.join(THISDIR, "luescher_nd", "zeta", "pyzeta.pyx")],
        language="c++",
        extra_compile_args=["--std=c++11"],
    ),
    Extension(
        path.join("luescher_nd", "zeta", "extern", "pyzeta").replace("/", "."),
        [path.join(THISDIR, "luescher_nd", "zeta", "extern", "pyzeta.pyx")],
        language="c++",
        extra_compile_args=["--std=c++11"],
    ),
]


setup(
    name="luescher_nd",
    version=str(__version__),
    description="Lattice Hamiltonian in finte Volume for two particles",
    long_description=README,
    long_description_content_type="text/markdown",
    author=__author__,
    packages=find_packages(exclude=["tests", "solvers", "benchmarks"]),
    install_requires=REQUIREMENTS,
    test_suite="tests",
    cmdclass={"configure": Configure},
    ext_modules=cythonize(EXTENSIONS),
    setup_requires=["cython", "numpy"],
)
