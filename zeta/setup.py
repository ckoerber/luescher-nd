from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    name="pyzeta",
    ext_modules=cythonize(
        Extension(
            "pyzeta", sources=["zeta_wrapper.pyx", "zeta.cc"], language="c++"
        )  # additional source file(s)
    ),
)
