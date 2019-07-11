#!/usr/bin/env python3

import numpy as np
from luescher_nd.operators import get_A1g_operator as A1g
import scipy.sparse as sp

import matplotlib.pyplot as plt

ndim = 3
n1d=6
op = A1g(n1d, ndim)

for i in range(n1d**ndim):
    for j in range(n1d**ndim):
        if np.abs(op[i,j] - op[j,i]) > 1e-7:
            print(f"Not hermitian!  Problem with op[{i},{j}]≠op[{j},{i}]")
            exit()
print("Hermitian!")

dense = op.todense()

f = plt.figure()
a = f.add_subplot(111)

print(f"det = {np.linalg.det(dense)}")

a.imshow(dense)

plt.show()

#print(op)
# print(sp.det(op))
