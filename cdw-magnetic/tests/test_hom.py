"""Homogeneous problem: formulas against independent references (mpmath, old code, FD)."""

import math

import mpmath as mp
import numpy as np
import pytest

from magcdw.common import hurwitz_zeta
from magcdw.hom import formulas as F
from magcdw.hom.minimize import find_minimum

RNG = np.random.default_rng(7)


@pytest.mark.parametrize("s", [-0.5, 0.5, 1.5])
def test_hurwitz_zeta_against_mpmath(s):
    xs = np.r_[1e-8, 1e-3, 0.3, 1.0, 7.0, 1e3]
    ref = np.array([float(mp.zeta(s, x)) for x in xs])
    assert np.max(np.abs(hurwitz_zeta(s, xs) / ref - 1)) < 1e-12


def test_vacuum_against_mpmath():
    for B, M in RNG.uniform([0.01, 0], [7, 4], (50, 2)):
        with mp.workdps(30):
            x = mp.mpf(M) ** 2 / (2 * mp.mpf(B))
            ref = (M ** 2 / (2 * -math.pi) + M * B / (2 * math.pi)
                   - float((2 * mp.mpf(B)) ** 1.5 * mp.zeta(-0.5, x)) / (2 * math.pi))
        assert abs(F.vacuum(B, M) - ref) < 1e-12


def test_zero_field_limit_of_vacuum():
    for M in (0.1, 0.7, 1.5):
        assert abs(F.vacuum(1e-6, M) - (M ** 2 / (2 * -math.pi) + M ** 3 / (3 * math.pi))) < 1e-6


def test_potential_matches_reference_code():
    import homogeneous_potential as H
    for B, M, mu in RNG.uniform([0.05, 0, 0], [7, 3, 2.5], (60, 3)):
        for T in (0.0, 0.01):
            assert abs(F.potential(B, M, mu, T) - H.potential(B, M, mu, T)[0]) < 1e-11


@pytest.mark.parametrize("T", [0.001, 0.01, 0.1])
def test_gap_is_derivative(T):
    for B, M, mu in RNG.uniform([0.05, 0.02, 0], [7, 3, 2.5], (40, 3)):
        h = 1e-5
        fd = (F.potential(B, M + h, mu, T) - F.potential(B, M - h, mu, T)) / (2 * h * M)
        assert abs(F.gap(B, M, mu, T) - fd) < 1e-6 * max(1.0, abs(fd))


@pytest.mark.parametrize("T", [0.0, 0.001, 0.01])
def test_bounds_enclose_gap(T):
    for B, mu in RNG.uniform([0.05, 0], [7, 2.5], (30, 2)):
        for p in (1e-6, 0.01, 0.3, 1.1):
            q = p * 1.7
            lo, up = F.gap_bounds(B, p, q, mu, T)
            values = F.gap(B, np.linspace(p, q, 200), mu, T)
            assert lo <= values.min() + 1e-12 and values.max() <= up + 1e-12


def test_minimum_never_worse_than_grid_search():
    import homogeneous_potential as H
    for B, mu in RNG.uniform([0.05, 0], [7, 2], (15, 2)):
        for T in (0.0, 0.01):
            r = find_minimum(B, mu, T)
            o = H.minimize_mass(B, mu, T, grid_points=401, mass_max=6)
            assert r.omega_rel <= o.potential - F.potential(B, 0, mu, T) + 1e-10


def test_vacuum_mass_and_phases():
    # mu = 0, weak field: M* close to m_* = 1 and grows with B (magnetic catalysis)
    m1, m2 = find_minimum(0.05, 0.0, 0.0).mass, find_minimum(2.0, 0.0, 0.0).mass
    assert 1.0 < m1 < 1.05 and m2 > m1
    assert find_minimum(2.0, 0.0, 0.0).phase == "V"
    assert find_minimum(0.5, 1.8, 0.0).phase == "T"
