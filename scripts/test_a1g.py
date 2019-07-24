#!/usr/bin/env python3

import numpy as np
from luescher_nd.operators import get_projector_to_a1g as A1g
from luescher_nd.operators import get_a1g_reducer as reducer
import luescher_nd.operators.a1g as a1g
import luescher_nd.lattice as lattice
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt

import itertools, functools


ndim=3
n1d=6

nsq_max = None
for n1d in [4,6]:#20,50]:
    print(f"\n\n\n\nn1d = {n1d}")
    s = lattice.all_nsq_primitives(n1d, ndim)
    print("States:")
    for k in iter(sorted(s.keys())):
        result=""
        for vector in s[k]:
            result+=f"{vector} "
        print(f"    {k} {result}")
    print("Degeneracies:")
    d = a1g.nsq_degeneracy(n1d, ndim)
    for k in iter(sorted(d.keys())):
        print(f"    {k} {d[k]}")

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
