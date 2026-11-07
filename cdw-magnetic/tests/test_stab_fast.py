"""Fast stability kernels: vacuum table + matter sums (note stability/05)."""

import math

import numpy as np

from magcdw.stab import fast
from magcdw.stab.landau import kernels, vacuum_kernels
from magcdw.stab.zero_field import modes as zero_field_modes


def test_table_interpolation():
    rng = np.random.default_rng(11)
    for _ in range(6):
        X, Y = rng.uniform(0.07, 9.5), rng.uniform(0.03, 5.8)
        p, d = vacuum_kernels(X, Y)
        pi, di = fast.vacuum(X, Y, 1.0)
        assert abs(pi - p) < 1e-4 * abs(p) and abs(di - d) < 1e-3 * abs(d)


def test_scaling_with_field():
    # Phi_vac(q, M, B) = sqrt(B) f(q/sqrt(B), M/sqrt(B)): direct computation at B = 0.36
    q, M, B = 0.6, 0.48, 0.36
    s = math.sqrt(B)
    p1, d1 = fast.vacuum(q, M, B)
    p2, d2 = vacuum_kernels(q / s, M / s)
    assert abs(p1 - s * p2) < 1e-4 * abs(p1) and abs(d1 - s * d2) < 1e-3 * abs(d1)


def test_fast_equals_full_landau_sum():
    for q, M, B, mu, T in ((0.8, 0.8, 0.5, 0.3, 0.05), (1.2, 0.3, 0.3, 1.0, 0.02)):
        f, s = fast.kernels(q, M, B, mu, T), kernels(q, M, B, mu, T)
        assert abs(f[0] - s[0]) < 2e-5 and abs(f[1] - s[1]) < 2e-4


def test_symmetric_kernel_weak_field_limit():
    # M = 0, small field: approaches the zero-field value G(0+)/2 + (q^2/2) R(q)
    q, mu, T = 0.8, 1.2, 0.05
    target = zero_field_modes(q, 0.0, 0.0, mu, T)[0]
    value = fast.symmetric_kernel(q, 0.05, mu, T)
    assert abs(value - target) < 0.05 * abs(target) + 2e-3
