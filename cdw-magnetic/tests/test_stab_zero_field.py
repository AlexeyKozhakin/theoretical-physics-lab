"""B = 0 second variation (note stability/02): closed forms against direct 2D integrals."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from magcdw.common import occupation
from magcdw.stab.zero_field import gap_at_zero, modes, r_at_zero, r_function, r_vacuum


def r_direct(q, M, mu, T):
    f = lambda E: 1 - float(occupation(E, mu, T))

    def inner(p):
        E = math.sqrt(M * M + p * p)

        def angle(th):
            Ep = math.sqrt(M * M + p * p + q * q + 2 * p * q * math.cos(th))
            d = Ep * Ep - E * E
            return 0.0 if abs(d) < 1e-12 else (f(E) / E - f(Ep) / Ep) / d
        return p * quad(angle, 0, math.pi, limit=200)[0] * 2 / (2 * math.pi) ** 2
    return quad(inner, 0, np.inf, limit=400, epsabs=1e-12)[0]


@pytest.mark.parametrize("q,M,mu,T", [(0.7, 1.0, 0.5, 0.05), (1.5, 0.6, 1.1, 0.05),
                                      (0.5, 0.6, 1.1, 0.05), (2.5, 0.4, 0.9, 0.02)])
def test_closed_form_against_direct_integral(q, M, mu, T):
    assert abs(r_function(q, M, mu, T) - r_direct(q, M, mu, T)) < 1e-10


def test_vacuum_closed_form():
    assert abs(r_vacuum(0.7, 1.0) - r_direct(0.7, 1.0, 0.0, 0.0)) < 1e-12


def test_zero_temperature_degeneracy_below_2pF():
    M, mu = 0.6, 1.1
    pF = math.sqrt(mu * mu - M * M)
    for q in (0.1, 0.5, 1.0, 1.9 * pF):
        assert abs(r_function(q, M, mu, 0.0)) < 1e-13
    assert r_function(2.2 * pF, M, mu, 0.0) > 0


def test_limits_at_small_q():
    for M, mu, T in ((0.6, 0.3, 0.05), (1.0, 0.0, 0.01)):
        assert abs(r_function(1e-5, M, mu, T) - r_at_zero(M, mu, T)) < 1e-8
    # radial mode at q -> 0 equals Omega''(M_0)/2 in the vacuum (g = -pi, M_0 = 1)
    _, radial = modes(1e-6, 1.0, 0.0, 0.3, 0.0)
    assert abs(radial - 0.5 * (-1 / math.pi + 2 / math.pi)) < 1e-9


def test_goldstone_zero_and_rotation_invariance():
    g1, r1 = modes(0.8, 1.0, 0.0, 0.2, 0.01)
    g2, r2 = modes(0.8, 0.6, 0.8, 0.2, 0.01)
    assert abs(g1 - g2) < 1e-14 and abs(r1 - r2) < 1e-14
    assert abs(modes(1e-7, 1.0, 0.0, 0.2, 0.01)[0]) < 1e-12


def test_symmetric_phase_constant():
    assert abs(gap_at_zero(1.05, 0.0) - (-1 / math.pi + 1.05 / math.pi)) < 1e-15
