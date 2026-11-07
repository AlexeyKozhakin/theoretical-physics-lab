"""Fermion propagator in the constant field (note stability/03), our conventions.

Operator (signs of the stability section, pi_0 = 0 by a U_A(1) rotation):

    D = gamma^mu P_mu + mu gamma^0 + M,   P_mu = i d_mu - e A_mu,   A_2 = B x_1,

so [pi_x, pi_y] = -i eB with pi^j = -P_j; we take eB > 0. S(x, y) = D^{-1} =
exp(i Phi(x, y)) S~(x - y) with the Schwinger phase Phi(x, y) + Phi(y, x) = 0, and

    S~(k) = exp(-k_perp^2/eB) sum_n (-1)^n D_n(k) / (k~0^2 - M^2 - 2 n eB),   k~0 = k0 + mu,

    D_n = (k~0 gamma^0 - M) [ (1 + i g1g2) L_n(x) - (1 - i g1g2) L_{n-1}(x) ]
          + 4 (k1 gamma^1 + k2 gamma^2) L^1_{n-1}(x),          x = 2 k_perp^2 / eB,

(k1, k2 contravariant, i g1g2 = i gamma^1 gamma^2 = diag(sigma3, sigma3)). This is
Gusynin-Miransky-Shovkovy hep-ph/9509320, Eqs. (25)-(26), with m -> -M (mass sign of our D),
S_ours = -i S_GMS (their propagator carries a factor i), sign(eB)_GMS = -1 (opposite
commutator); the transverse sign is that of GMS and is confirmed by the free limit B -> 0
(generating functions at z = -1: sum (-1)^n e^{-x/2} L_n = 1/2, sum (-1)^n e^{-x/2} L^1_n = 1/4).
"""

import math

import numpy as np

# gamma matrices, (A.1), (A.5), (A6)
_s1 = np.array([[0, 1], [1, 0]], complex)
_s2 = np.array([[0, -1j], [1j, 0]], complex)
_s3 = np.array([[1, 0], [0, -1]], complex)
_Z = np.zeros((2, 2), complex)
_I2 = np.eye(2, dtype=complex)
G0 = np.block([[_s3, _Z], [_Z, -_s3]])
G1 = np.block([[1j * _s1, _Z], [_Z, -1j * _s1]])
G2 = np.block([[1j * _s2, _Z], [_Z, -1j * _s2]])
G5 = 1j * np.block([[_Z, -_I2], [_I2, _Z]])
I4 = np.eye(4, dtype=complex)
SPIN = 1j * G1 @ G2                      # = diag(sigma3, sigma3)


def laguerre_weighted(x, n_max):
    """e^{-x/2} L_n(x) and e^{-x/2} L^1_n(x), n = 0..n_max (stable three-term recurrence)."""
    L0 = np.zeros(n_max + 1)
    L1 = np.zeros(n_max + 1)
    w = math.exp(-x / 2)
    for alpha, out in ((0, L0), (1, L1)):
        out[0] = w
        if n_max >= 1:
            out[1] = w * (1 + alpha - x)
        for n in range(1, n_max):
            out[n + 1] = ((2 * n + 1 + alpha - x) * out[n] - (n + alpha) * out[n - 1]) / (n + 1)
    return L0, L1


def propagator(k0, k1, k2, mass, field, chemical=0.0, n_max=None):
    """S~(k) as a 4x4 matrix; k0 may be complex (Matsubara: k0 = i omega)."""
    kt0 = k0 + chemical
    kp2 = k1 * k1 + k2 * k2
    x = 2 * kp2 / field
    if n_max is None:
        n_max = int(4 * x + 400)
    L0, L1 = laguerre_weighted(x, n_max)
    Lm1 = np.r_[0.0, L0[:-1]]             # L_{n-1}, with L_{-1} = 0
    L1m1 = np.r_[0.0, L1[:-1]]            # L^1_{n-1}
    n = np.arange(n_max + 1)
    sign = (-1.0) ** n
    sign[-1] *= 0.5                       # alternating series: average of the last two partial sums
    den = kt0 * kt0 - mass * mass - 2 * n * field
    a = np.sum(sign * L0 / den)           # coefficient of (1 + i g1g2)
    b = np.sum(sign * Lm1 / den)          # coefficient of (1 - i g1g2)
    c = np.sum(sign * L1m1 / den)         # transverse
    # the extra e^{-k_perp^2/eB} = e^{-x/2} is already inside laguerre_weighted
    long = kt0 * G0 - mass * I4
    return long @ ((I4 + SPIN) * a - (I4 - SPIN) * b) + 4 * (k1 * G1 + k2 * G2) * c


def free_propagator(k0, k1, k2, mass, chemical=0.0):
    kt0 = k0 + chemical
    kslash = kt0 * G0 - k1 * G1 - k2 * G2
    return (kslash - mass * I4) / (kt0 * kt0 - k1 * k1 - k2 * k2 - mass * mass)
