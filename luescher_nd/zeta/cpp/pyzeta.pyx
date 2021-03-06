# cython: language_level=3
from collections.abc import Iterable

from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np


cdef extern from "zeta_wrapper.h":
    cdef cppclass SphericalZeta_CC:
        SphericalZeta_CC(unsigned int, unsigned int, bool) except +
        vector[double] operator()(const vector[double])

    cdef cppclass CartesianZeta_CC:
        CartesianZeta_CC(unsigned int, unsigned int, bool) except +
        vector[double] operator()(const vector[double])

    cdef cppclass DispersionZeta_CC:
        DispersionZeta_CC(
            unsigned int,
            unsigned int,
            double,
            unsigned int,
            bool
        ) except +
        vector[double] operator()(const vector[double])

cdef class SphericalZeta:
    cdef SphericalZeta_CC *c_ptr

    def __cinit__(self, unsigned int D, unsigned int N, bool improved=True):
        self.c_ptr = new SphericalZeta_CC(D, N, improved)

    def __call__(self, x):
        x = np.array([x]) if not isinstance(x, Iterable) else x
        return np.array(self.c_ptr[0](x))

cdef class CartesianZeta:
    cdef CartesianZeta_CC *c_ptr

    def __cinit__(self, unsigned int D, unsigned int N, bool improved=True):
        self.c_ptr = new CartesianZeta_CC(D, N, improved)

    def __call__(self, x):
        x = np.array([x]) if not isinstance(x, Iterable) else x
        return np.array(self.c_ptr[0](x))

cdef class DispersionZeta:
    cdef DispersionZeta_CC *c_ptr

    def __cinit__(
        self,
        unsigned int D,
        unsigned int N,
        double L,
        unsigned int nstep=1,
        bool improved=True
    ):
        raise NotImplementedError("Dispersion Zeta not yet implemented.")
        self.c_ptr = new DispersionZeta_CC(D, N, L, nstep, improved)

    def __call__(self, x):
        x = np.array([x]) if not isinstance(x, Iterable) else x
        return np.array(self.c_ptr[0](x))
