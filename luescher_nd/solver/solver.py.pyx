# cython: language_level=3
from libcpp.vector cimport vector
import numpy as np

cdef extern from "solver.cpp":
    void convert_print_mat(const int n, const vector[long] rows, const vector[long] cols, const vector[double] coeffs)

def convert_print(const int n, const vector[long] rows, const vector[long] cols, const vector[double] coeffs):
    """
    """
    return convert_print_mat(n, rows, cols, coeffs)
