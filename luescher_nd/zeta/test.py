"""Tests python zeta function computation
"""
import numpy as np
import time

from pyzeta import py_zeta2


def zeta2(x: np.ndarray, l: int):
    """
    """
    res = 0

    for n1 in range(-l, l + 1):
        for n2 in range(-l, l + 1):
            res += 1 / (n1 ** 2 + n2 ** 2 - x)

    return np.array(res) - np.log(l) * np.pi * 2


def test_pyzeta2():
    """
    """
    x = np.linspace(0, 10, 21) + 0.1
    lambdas = [1, 2, 5, 10, 100]

    for l in lambdas:
        expected = zeta2(x, l)
        computed = py_zeta2(x, l)
        np.testing.assert_allclose(expected, computed)


def measure_time(func, *args, nruns=10):
    """
    """
    timeings = []
    for _ in range(nruns):
        t1 = time.time_ns()
        func(*args)
        t2 = time.time_ns()
        timeings.append((t2 - t1) / 1.e9)

    return np.mean(timeings), np.std(timeings, ddof=1)


def test_timing():
    """
    """
    x = np.linspace(0, 10, 21) + 0.1
    l = 100

    d1, m1 = measure_time(zeta2, x, l)
    print(f"Python timeings: {d1:1.2e} +/- {m1:1.2e}")

    d2, m2 = measure_time(py_zeta2, x, l)
    print(f"C++ timeings: {d2:1.2e} +/- {m2:1.2e}")

    assert d2 < d1 + np.sqrt(m1 ** 2 + m2 ** 2)
