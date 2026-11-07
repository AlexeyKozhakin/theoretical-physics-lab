"""Homogeneous condensate in the field: Omega(M; mu, B, T) and the gap function G(M).

One multiplet, normalization of [CDW] (vacuum mass m_* = pi/|g| = 1 at g = -pi),
B = |eB| > 0, per unit area and fermion copy:

    Omega(M) = M^2/(2g) + M B/(2 pi) - (2B)^{3/2}/(2 pi) zeta(-1/2, M^2/(2B))
               - d_B sum_n s_n H(eps_n),   eps_n = sqrt(M^2 + 2 n B),
    d_B = B/(2 pi),  s_0 = 1,  s_n = 2 (n >= 1),  H — common.thermal_weight.

Gap function G = Omega'(M)/M = G_vac + G_th:

    G_vac = 1/g + B/(2 pi M) - sqrt(2B)/(2 pi) zeta(1/2, M^2/(2B))   (strictly increasing)
    G_th  = d_B sum_n s_n w(eps_n)/eps_n,  w = n_F(eps-mu) + n_F(eps+mu)  (non-increasing)

The 1/M terms of G_vac and of the n = 0 term of G_th cancel against each other when
the lowest level is filled; `gap` uses the combined form, `gap_parts` the split one.
"""

import math

import numpy as np

from ..common import COUPLING, hurwitz_zeta, occupation, thermal_weight

RIEMANN_ZETA_M12 = -0.20788622497735457      # zeta(-1/2)


def _x(field, mass):
    return np.asarray(mass, float) ** 2 / (2 * field)


def vacuum(field, mass, coupling=COUPLING):
    """Renormalized vacuum part (no thermal term); B = 0 gives M^2/2g + M^3/3pi."""
    M = np.abs(np.asarray(mass, float))
    if field == 0:
        return M ** 2 / (2 * coupling) + M ** 3 / (3 * math.pi)
    x = _x(field, M)
    # zeta(-1/2, x) = x^{1/2} + zeta(-1/2, x + 1): regular at x = 0.
    z = np.sqrt(x) + hurwitz_zeta(-0.5, x + 1)
    return M ** 2 / (2 * coupling) + M * field / (2 * math.pi) - (2 * field) ** 1.5 * z / (2 * math.pi)


def level_cut(field, chemical, temperature, margin=40.0):
    """Index n_max so that levels above it contribute < exp(-margin) relatively."""
    top = abs(chemical) + margin * temperature
    return int(math.ceil(top ** 2 / (2 * field))) + 1


def levels(field, mass, n_max):
    """eps_n, n = 0..n_max, and multiplicities s_n; mass may be an array (levels on the last axis)."""
    n = np.arange(n_max + 1)
    eps = np.sqrt(np.asarray(mass, float)[..., None] ** 2 + 2 * field * n)
    s = np.r_[1.0, np.full(n_max, 2.0)]
    return eps, s


def potential(field, mass, chemical, temperature, coupling=COUPLING):
    """Omega(M; mu, B, T)."""
    if field <= 0:
        raise ValueError("use B > 0 (the B = 0 limit is in limits.py)")
    eps, s = levels(field, mass, level_cut(field, chemical, temperature))
    thermal = np.sum(s * thermal_weight(eps, chemical, temperature), axis=-1)
    return vacuum(field, mass, coupling) - field / (2 * math.pi) * thermal


def gap_parts(field, mass, chemical, temperature, coupling=COUPLING):
    """(G_vac, G_th) for M > 0: G_vac strictly increasing, G_th non-increasing in M."""
    M = np.asarray(mass, float)
    if np.any(M <= 0):
        raise ValueError("gap_parts needs M > 0; use gap() at M = 0")
    x = _x(field, M)
    g_vac = (1 / coupling + field / (2 * math.pi * M)
             - math.sqrt(2 * field) / (2 * math.pi) * hurwitz_zeta(0.5, x))
    eps, s = levels(field, M, level_cut(field, chemical, temperature))
    g_th = field / (2 * math.pi) * np.sum(s * occupation(eps, chemical, temperature) / eps, axis=-1)
    return g_vac, g_th


def gap_bounds(field, p, q, chemical, temperature, coupling=COUPLING):
    """Lower and upper bounds of G on [p, q], 0 < p < q, from the regular split

        G = A(M) - d_B u(M)/M + D(M),
        A = 1/g - sqrt(2B)/(2pi) zeta(1/2, M^2/2B + 1)       (increasing, regular at 0)
        u = 1 - w(M)  in [0, 1]                              (non-decreasing)
        D = d_B sum_{n>=1} 2 w(eps_n)/eps_n                  (non-increasing).
    """
    d = field / (2 * math.pi)
    M = np.array([p, q], float)
    A = 1 / coupling - math.sqrt(2 * field) / (2 * math.pi) * hurwitz_zeta(0.5, _x(field, M) + 1)
    eps, _ = levels(field, M, level_cut(field, chemical, temperature))
    w = occupation(eps, chemical, temperature)
    u = 1 - w[:, 0]
    D = 2 * d * np.sum(w[:, 1:] / eps[:, 1:], axis=-1)
    lower = A[0] - d * u[1] / p + D[1]
    upper = A[1] - d * u[0] / q + D[0]
    return lower, upper


def gap(field, mass, chemical, temperature, coupling=COUPLING):
    """G(M) = Omega'(M)/M with the n = 0 singularity combined; finite at M -> 0 when w(0) = 1.

    G = 1/g - sqrt(2B)/(2pi) zeta(1/2, x+1) + d_B (w(M) - 1)/M + d_B sum_{n>=1} 2 w(eps_n)/eps_n.
    At M = 0 the limit d_B w'(0) is used for T > 0; at T = 0 it is 0 if mu > 0 and -inf if mu = 0.
    """
    M = np.atleast_1d(np.abs(np.asarray(mass, float)))
    x = _x(field, M)
    d = field / (2 * math.pi)
    n_max = level_cut(field, chemical, temperature)
    eps, s = levels(field, M, n_max)
    w = occupation(eps, chemical, temperature)
    upper = 2 * d * np.sum(w[..., 1:] / eps[..., 1:], axis=-1)
    w0 = w[..., 0]
    lll = np.empty_like(M)
    small = M == 0
    lll[~small] = d * (w0[~small] - 1) / M[~small]
    if np.any(small):
        if temperature > 0:
            mu, T = abs(chemical), temperature
            # w'(0) = -(1/T) [f(1-f)](mu/T) * 2 / 2 with f = n_F; even in mu.
            f = 1 / (1 + math.exp(min(mu / T, 700.0)))
            lll[small] = -d * 2 * f * (1 - f) / T
        else:
            lll[small] = 0.0 if chemical != 0 else -np.inf
    value = 1 / coupling - math.sqrt(2 * field) / (2 * math.pi) * hurwitz_zeta(0.5, x + 1) + lll + upper
    return value if np.ndim(mass) else float(value[0])


def thresholds(field, chemical):
    """Masses M_n = sqrt(mu^2 - 2nB) where level n crosses mu (T = 0 kinks of Omega)."""
    n = np.arange(int(chemical ** 2 / (2 * field)) + 1)
    return np.sqrt(np.maximum(chemical ** 2 - 2 * field * n, 0.0))
