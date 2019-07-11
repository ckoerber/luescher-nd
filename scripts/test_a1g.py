#!/usr/bin/env python3

import numpy as np
from luescher_nd.operators import get_A1g_operator as A1g
from luescher_nd.operators import get_A1g_reducer as reducer
import scipy.sparse as sp

import matplotlib.pyplot as plt

ndim = 3
n1d=6
op = A1g(n1d, ndim)
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
