"""Second variation in the field, Landau-level form (note stability/04)."""

import math
import sys
from pathlib import Path

import numpy as np

from magcdw.hom import formulas as F
from magcdw.hom.minimize import find_minimum
from magcdw.stab.landau import displacement, kernels, polarization
from magcdw.stab.zero_field import r_function


def test_displacement_matches_exact_formula_and_is_unitary():
    from relative_potential import displacement as exact
    for a in (0.1, 1.3, 3.0, -0.8):
        assert np.max(np.abs(displacement(a, 150) - exact(a, 150))) < 1e-12
    D = displacement(0.7, 2000)
    rows = np.sum(D[:1800] ** 2, axis=1)
    assert np.all(np.abs(rows - 1) < 1e-9)


def test_uniform_limit_equals_gap_function_derivative():
    """Pi_eps(0) - Pi_omega(0) = M G'(M)/2 for any M (no gap equation needed)."""
    for M, B, mu, T in ((0.8, 0.5, 0.3, 0.05), (0.5, 1.0, 0.9, 0.05)):
        vals = []
        for N in (50, 100, 200):
            pe, pw = polarization(0.0, M, B, mu, T, n_cut=N, smooth=True)
            vals.append(pe - pw)
        A = np.array([[1.0, N ** -0.5, 1.0 / N] for N in (50, 100, 200)])
        limit = np.linalg.solve(A, np.array(vals))[0]
        h = 1e-5
        target = 0.5 * M * (F.gap(B, M + h, mu, T) - F.gap(B, M - h, mu, T)) / (2 * h)
        assert abs(limit - target) < 2e-4 * max(1.0, abs(target))


def test_goldstone_kernel_tends_to_zero_field():
    M, mu, T, q = 0.8, 0.3, 0.05, 0.8
    target = 0.5 * q * q * r_function(q, M, mu, T)
    phi = [kernels(q, M, B, mu, T)[0] for B in (0.4, 0.2)]
    # deviation shrinks with B and is already small
    assert abs(phi[1] - target) < abs(phi[0] - target) + 1e-4
    assert abs(phi[1] - target) < 0.02 * target


def test_stiffness_matches_spiral_potential_of_branch_a():
    """4 M_0^2 lim Phi_pi/(q^2/2) = d^2 Omega(M_0, b)/db^2 at b = 0 (independent Rabi computation)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from fast_engine import homogeneous_vacuum, low_levels, thermal_weight, vacuum_difference
    B, mu, T = 0.5, 0.3, 0.05
    M0 = find_minimum(B, mu, T).mass

    def omega(b):
        lv = low_levels(B, M0, b, mu + 40 * T)
        return (homogeneous_vacuum(B, M0) + vacuum_difference(B, M0, b, 512)
                - B / (2 * math.pi) * np.sum(thermal_weight(lv, mu, T)))
    h = 0.02
    d2 = 2 * (omega(h) - omega(0.0)) / h ** 2
    qs = np.array([0.1, 0.2, 0.3])
    K = [kernels(q, M0, B, mu, T)[0] / (q * q / 2) for q in qs]
    K0 = np.polyfit(qs ** 2, K, 2)[-1]
    assert abs(4 * M0 * M0 * K0 - d2) < 2e-3 * d2
