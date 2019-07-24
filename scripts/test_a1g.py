#!/usr/bin/env python3

import numpy as np
from luescher_nd.operators import get_projector_to_a1g as A1g
from luescher_nd.operators import get_a1g_reducer as reducer
from luescher_nd.operators import a1g_list as states
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt

ndim=3
n1d=4

s = states(n1d, ndim, 9)
for k in iter(sorted(s.keys())):
    print(f"{k} {s[k][0][1]}")
exit()

ndim = 3
n1d=6
op = A1g(n1d, ndim)
vals = eigsh(op, k=n1d**ndim / 2)
print(vals)
dense = op.todense()
r = reducer(n1d, ndim)

f = plt.figure()
a = f.add_subplot(211)
a.set_title(f"Nx = {n1d}, D={ndim}")
a.imshow(dense)

b = f.add_subplot(212)
b.imshow(r)
b.set_title(f"Shape = {r.shape}")

plt.show()

#print(op)
# print(sp.det(op))
