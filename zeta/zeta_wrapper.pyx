# cython: language_level=3

from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np


cdef extern from "zeta.h":
    cdef cppclass spherical:
        spherical(unsigned int, unsigned int, bool) except +
        bool filter(vector[unsigned int])
        int degeneracy(vector[unsigned int])
        double counterterm(double)


cdef class SphericalDomain:
    cdef spherical *c_spherical_ptr

    def __cinit__(self, unsigned int D, unsigned int N, bool improved=True):
        self.c_spherical_ptr = new spherical(D, N, improved)

    def filter(self, vector[unsigned int] v):
        return self.c_spherical_ptr.filter(v)

    def degeneracy(self, vector[unsigned int] v):
        return self.c_spherical_ptr.degeneracy(v)

    def counterterm(self, double x):
        return self.c_spherical_ptr.counterterm(x)
