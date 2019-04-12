# cython: language_level=3
from libcpp.vector cimport vector
import numpy as np

cdef extern from "zeta.cc":
    double zetaRGL_re(int l, int m, const vector[double] s, double gam, double usq)

def zeta(const vector[double] x):
    """Zeta function for m=0, l=0 channel with zero CMS momentum
    """
    x = x if hasattr(x, "__iter__") else [x] 
    return np.array([zetaRGL_re(0, 0, [0.0, 0.0, 0.0], 1.0, xi) for xi in x])
