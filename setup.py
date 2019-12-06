#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup file for the luescher-nd
"""
from os import path

import sys

from urllib.request import urlretrieve

from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
from setuptools.command.install import install as InstallCommand
from setuptools.command.develop import develop as DevelopCommand
from setuptools.command.egg_info import egg_info as EggInfoCommand

from Cython.Build import cythonize

__version__ = "1.0.0"
__author__ = "Christopher Körber"

if sys.version_info[0] < 3:
    raise Exception("luescher_nd requires python version 3.6 or later")
else:
    if sys.version_info[1] < 6:
        raise Exception("luescher_nd requires python version 3.6 or later")

THISDIR = path.abspath(path.dirname(__file__))
with open(path.join(THISDIR, "README.md"), encoding="utf-8", mode="r") as f:
    README = f.read()

with open(path.join(THISDIR, "requirements.txt"), mode="r") as f:
    REQUIREMENTS = [el.strip() for el in f.read().split(",")]


ZETA_EXTERNAL = path.join(THISDIR, "luescher_nd", "zeta", "extern", "zeta.cc")


def download_extern():
    """This function downloads `zeta.cc` and `zeta.h` from the repository

    https://github.com/cjmorningstar10/TwoHadronsInBox

    The files are needed for the three-dimensional zeta function
    (`from luescher_nd.zeta.extern.pyzeta import zeta`).

    If the files are not present, the library is still installed but the above import
    will fail.
    You either have to provide your own implementation or use this repo partially.
    """
    repository = "https://raw.githubusercontent.com/cjmorningstar10/TwoHadronsInBox/"
    branch = "master"
    file = "source/zeta.cc"
    extensions = [".cc", ".h"]

    info = """
    This module depends on the files `zeta.cc` and `zeta.h`
    from the repository

    https://github.com/cjmorningstar10/TwoHadronsInBox

    to computed the three-dimensional zeta function
    (e.g., `from luescher_nd.zeta.extern.pyzeta import zeta`).

    This script will place them in the right directory and compile them.
    If the files are not present, the install will fail (you have to set
    `CYTHONIZED_EXT=None` in this file and not import the above function.)
    """
    print(info)

    for extension in extensions:
        url = f"{repository}/{branch}/{file}".replace(".cc", extension)
        file_path = ZETA_EXTERNAL.replace(".cc", extension)
        urlretrieve(url, file_path)


class CustomInstallCommand(InstallCommand):
    def run(self):
        if not path.exists(ZETA_EXTERNAL) or not path.exists(
            ZETA_EXTERNAL.replace(".cc", ".h")
        ):
            download_extern()
        InstallCommand.run(self)


class CustomDevelopCommand(DevelopCommand):
    def run(self):
        if not path.exists(ZETA_EXTERNAL) or not path.exists(
            ZETA_EXTERNAL.replace(".cc", ".h")
        ):
            download_extern()
        DevelopCommand.run(self)


class CustomEggInfoCommand(EggInfoCommand):
    def run(self):
        if not path.exists(ZETA_EXTERNAL) or not path.exists(
            ZETA_EXTERNAL.replace(".cc", ".h")
        ):
            download_extern()
        EggInfoCommand.run(self)


EXTENSIONS = [
    Extension(
        path.join("luescher_nd", "zeta", "extern", "pyzeta").replace("/", "."),
        [path.join(THISDIR, "luescher_nd", "zeta", "extern", "pyzeta.pyx")],
        language="c++",
        extra_compile_args=["--std=c++11"],
    ),
    Extension(
        path.join("luescher_nd", "zeta", "cpp", "pyzeta").replace("/", "."),
        [
            path.join(THISDIR, "luescher_nd", "zeta", "cpp", "pyzeta.pyx"),
            path.join(THISDIR, "luescher_nd", "zeta", "cpp", "zeta.cc"),
        ],
        language="c++",
        extra_compile_args=["--std=c++11"],
    ),
]
CYTHONIZED_EXT = cythonize(EXTENSIONS)

setup(
    name="luescher_nd",
    version=str(__version__),
    description="Lattice Hamiltonian in finte Volume for two particles",
    long_description=README,
    long_description_content_type="text/markdown",
    author=__author__,
    packages=find_packages(exclude=["tests", "benchmarks"]),
    install_requires=REQUIREMENTS,
    ext_modules=CYTHONIZED_EXT,
    setup_requires=["cython", "numpy", "pytest-runner"],
    tests_require=["pytest"],
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
)
