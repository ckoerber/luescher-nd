from libcpp.vector cimport vector

cdef vector[long] ivec
cdef vector[double] dvec

cdef extern from "zeta.cpp":
    vector[double] zeta2(const vector[double] x, const long Lambda)

def py_zeta2(const vector[double] x, const long Lambda):
    """
    """
    return zeta2(x, Lambda)
