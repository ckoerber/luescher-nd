# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np


cdef extern from "zeta_flat.h":
    cdef cppclass SphericalZeta:
        SphericalZeta(unsigned int, unsigned int, bool) except +
        double operator()(const double)

    cdef cppclass CartesianZeta:
        CartesianZeta(unsigned int, unsigned int, bool) except +
        double operator()(const double)

    cdef cppclass DispersionZeta:
        DispersionZeta(
            unsigned int,
            unsigned int,
            double,
            unsigned int,
            bool
        ) except +
        double operator()(const double)

cdef class PySphericalZeta:
    cdef SphericalZeta *c_ptr

    def __cinit__(self, unsigned int D, unsigned int N, bool improved=True):
        self.c_ptr = new SphericalZeta(D, N, improved)

    def __call__(self, const double x):
        return self.c_ptr[0](x)

cdef class PyCartesianZeta:
    cdef CartesianZeta *c_ptr

    def __cinit__(self, unsigned int D, unsigned int N, bool improved=True):
        self.c_ptr = new CartesianZeta(D, N, improved)

    def __call__(self, const double x):
        return self.c_ptr[0](x)

cdef class PyDispersionZeta:
    cdef DispersionZeta *c_ptr

    def __cinit__(
        self,
        unsigned int D,
        unsigned int N,
        double L,
        unsigned int nstep=1,
        bool improved=True
    ):
        self.c_ptr = new DispersionZeta(D, N, L, nstep, improved)

    def __call__(self, const double x):
        return self.c_ptr[0](x)
