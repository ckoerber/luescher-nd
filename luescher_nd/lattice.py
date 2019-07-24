
import numpy as np

import itertools as i

def n(n1d):
    return np.array(
        list(np.arange(0, (n1d + 1) // 2)) + list(-np.arange(n1d // 2, 0, -1))
    )

def sites(n1d, ndim):
    return n1d**ndim

def momenta(n1d, ndim):
    return [tuple(p) for p in np.transpose([el.flatten() for el in np.meshgrid(*[n(n1d)] * ndim)])]

def momentum_lookup(n1d, ndim):
    lookup = dict()
    for i, p in enumerate(momenta(n1d, ndim)):
        lookup[tuple(p)] = i

    return lookup

def nsq_primitives(n1d, ndim, nsq):
    ns = n(n1d)
    squares = set([n**2 for n in ns])
    primitives = set()

    # A real product is overkill, but it's fine for now.
    for n2s in i.product(squares, repeat=ndim):
        if np.sum(n2s) == nsq:
            vector = np.sqrt(n2s).astype(int)
            primitives.add(tuple(sorted(vector)))

    return primitives

def all_nsq_primitives(n1d, ndim, nsq=None):
    ns = n(n1d)
    squares = set([n**2 for n in ns])
    primitives = dict()

    if nsq is None:
        max_nsq = ndim*max(squares)
    else:
        max_nsq = nsq

    for n2s in i.product(squares, repeat=ndim):
        norm2 = np.sum(n2s)
        if norm2 > max_nsq:
            continue
        if norm2 not in primitives:
            primitives[norm2] = set()
        vector = tuple(sorted(np.sqrt(n2s).astype(int)))
        primitives[norm2].add(vector)

    return primitives
