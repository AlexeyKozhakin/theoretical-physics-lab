"""Field propagator (note stability/03): gamma algebra, free limit, Landau-level normalization."""

import math

import numpy as np
from scipy.integrate import quad

from magcdw.stab.propagator import G0, G1, G2, G5, SPIN, free_propagator, propagator


def test_gamma_algebra():
    g = [G0, G1, G2]
    metric = np.diag([1, -1, -1])
    for a in range(3):
        for b in range(3):
            assert np.allclose(g[a] @ g[b] + g[b] @ g[a], 2 * metric[a, b] * np.eye(4))
        assert np.allclose(G5 @ g[a] + g[a] @ G5, 0)
    assert np.allclose(G5 @ G5, np.eye(4))
    assert np.allclose(SPIN, np.diag([1, -1, 1, -1]))
    assert np.allclose(SPIN @ G5, G5 @ SPIN)


def test_free_limit_is_linear_in_B():
    for k0, k1, k2, M, mu in ((0.5j, 0.4, 0.3, 0.7, 0.0), (0.7j, 0.4, -0.3, 0.7, 0.3)):
        F = free_propagator(k0, k1, k2, M, mu)
        errs = [np.max(np.abs(propagator(k0, k1, k2, M, B, mu) - F)) for B in (0.04, 0.02, 0.01)]
        assert errs[0] < 0.05
        assert 1.7 < errs[0] / errs[1] < 2.4 and 1.7 < errs[1] / errs[2] < 2.4


def test_landau_level_normalization():
    """int d^2k/(2pi)^2 tr S~ = -2 M d_B sum_n s_n/(k0~^2 - eps_n^2): UV-finite difference in k0."""
    B, M, ka, kb = 1.5, 0.3, 0.8j, 2.0j

    def tr_int(k0):
        return quad(lambda k: k / (2 * math.pi) * np.trace(
            propagator(k0, k, 0.0, M, B, n_max=int(8 * k * k / B + 600))).real, 0, 40, limit=800)[0]
    n = np.arange(200000)
    s = np.where(n == 0, 1, 2)
    f = lambda k0: -2 * M * B / (2 * math.pi) * np.sum(s / (k0 * k0 - M * M - 2 * n * B))
    numeric, analytic = tr_int(ka) - tr_int(kb), (f(ka) - f(kb)).real
    assert abs(numeric - analytic) < 3e-3 * abs(analytic)
