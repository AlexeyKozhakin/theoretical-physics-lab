"""Static polarization of the homogeneous background in the field, Landau-level form (note stability/04).

One-particle Hamiltonian (signs of the stability section, gamma^0 D = i d_0 + mu - H, pi_0 = 0):

    H = alpha . pi - beta M,  alpha^j = gamma^0 gamma^j,  beta = gamma^0,  [pi_x, pi_y] = -i eB.

With a = (pi_x - i pi_y)/sqrt(2 eB) the two 2x2 blocks (psi_1, psi_2) are

    h_b = [[-m_b, i c a], [-i c a^dag, m_b]],   c = sqrt(2 eB),  m_1 = +M,  m_2 = -M,

eigenstates (u |n-1>, v |n>) with energies s eps_n, eps_n = sqrt(M^2 + 2 n eB), s = +-1,

    u = i sqrt((eps - s m_b)/(2 eps)),   v = s sqrt((eps + s m_b)/(2 eps)),

and for n = 0 only (0, |0>) with energy m_b (block 1: +M, block 2: -M).

Perturbations: dH = -beta eps(x) - i beta gamma^5 omega(x);
  -beta            = diag(-sigma3, +sigma3): couples a block to itself;
  -i beta gamma^5  = [[0, -sigma3], [-sigma3, 0]]: couples the two blocks.
Both act as diag(-1, +1) on (upper, lower) components, so for states a (level n), c (level m)

    <a| e^{i q.eta} V |c> = - u_a^* u_c D_{n-1,m-1}(xi) + v_a^* v_c D_{n,m}(xi),   xi = q^2/(2 eB),

D the displacement matrix (Laguerre). eps-omega mixing vanishes identically (same-block vs
cross-block). Per unit area and fermion copy

    Pi_ch(q) = (d_B/2) sum_{a,c in ch} |<a|..|c>|^2 (n_F(E_a - mu) - n_F(E_c - mu)) / (E_a - E_c),

d_B = eB/(2 pi), the degenerate limit E_a = E_c giving n_F'(E). Pi(q) - Pi(0) is UV finite;
the second-variation kernels are (gap equation 1/(4G) = -Pi_omega(0) for M != 0)

    Phi_pi(q)    = Pi_omega(q) - Pi_omega(0),
    Phi_sigma(q) = Omega''(M)/2 + Pi_eps(q) - Pi_eps(0).
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit



def displacement(alpha, size):
    """<n|D(alpha)|m>, n, m < size, real alpha, via normalized Laguerre functions

        l_n^(k)(x) = sqrt(n!/(n+k)!) x^{k/2} e^{-x/2} L_n^(k)(x),   x = alpha^2,
        l_{n+1} = [(2n+1+k-x) l_n - sqrt(n (n+k)) l_{n-1}] / sqrt((n+1)(n+k+1)),

    a recurrence in n with O(1) coefficients, vectorized over the diagonals k = m - n.
    D_{n,n+k} = (-sign alpha)^k l_n^(k), and D_{n+k,n} = (sign alpha)^k l_n^(k).
    """
    if alpha == 0:
        return np.eye(size)
    x = alpha * alpha
    k = np.arange(size, dtype=float)
    ell = np.zeros((size, size))                      # ell[n, k]
    log0 = 0.5 * k * math.log(x) - x / 2 - 0.5 * np.array([math.lgamma(v + 1) for v in k])
    ell[0] = np.exp(log0)
    if size > 1:
        ell[1] = (1 + k - x) * ell[0] / np.sqrt(k + 1)
    for n in range(1, size - 1):
        ell[n + 1] = ((2 * n + 1 + k - x) * ell[n] - np.sqrt(n * (n + k)) * ell[n - 1]) / np.sqrt((n + 1) * (n + k + 1))
    D = np.zeros((size, size))
    sgn = 1.0 if alpha > 0 else -1.0
    for kk in range(size):
        n = np.arange(size - kk)
        vals = ell[n, kk]
        D[n, n + kk] = (-sgn) ** kk * vals
        D[n + kk, n] = sgn ** kk * vals
    return D


def _fermi(E, mu, T):
    return expit(-(E - mu) / T)


def _fermi_prime(E, mu, T):
    f = _fermi(E, mu, T)
    return -f * (1 - f) / T


def levels(mass, field, n_cut):
    """States per block: list of (n, s, energy, r, v) with u = i r."""
    out = {1: [], 2: []}
    for b, m in ((1, mass), (2, -mass)):
        out[b].append((0, 0, m, 0.0, 1.0))
        for n in range(1, n_cut + 1):
            eps = math.sqrt(mass * mass + 2 * n * field)
            for s in (1, -1):
                out[b].append((n, s, s * eps, math.sqrt((eps - s * m) / (2 * eps)),
                               s * math.sqrt((eps + s * m) / (2 * eps))))
    return out


def _channel(states_a, states_c, Dm, mu, T, n_low, smooth=False):
    """Sum over pairs whose centre level (n_a + n_c)/2 is <= n_low (analogue of a cutoff on
    |p + q/2| at B = 0); the same rule is used at q = 0, so Pi(q) - Pi(0) has no boundary term."""
    n_a = np.array([s[0] for s in states_a]); E_a = np.array([s[2] for s in states_a])
    r_a = np.array([s[3] for s in states_a]); v_a = np.array([s[4] for s in states_a])
    n_c = np.array([s[0] for s in states_c]); E_c = np.array([s[2] for s in states_c])
    r_c = np.array([s[3] for s in states_c]); v_c = np.array([s[4] for s in states_c])
    # D_{n-1, m-1}: for n = 0 or m = 0 the upper component is absent (r = 0), any index works
    Dup = Dm[np.maximum(n_a - 1, 0)[:, None], np.maximum(n_c - 1, 0)[None, :]]
    Dlo = Dm[n_a[:, None], n_c[None, :]]
    # u_a^* u_c = (-i r_a)(i r_c) = r_a r_c
    amp = -(r_a[:, None] * r_c[None, :]) * Dup + (v_a[:, None] * v_c[None, :]) * Dlo
    dE = E_a[:, None] - E_c[None, :]
    fa, fc = _fermi(E_a, mu, T), _fermi(E_c, mu, T)
    same = np.abs(dE) < 1e-12
    ratio = np.where(same, _fermi_prime(E_a, mu, T)[:, None] * np.ones_like(dE),
                     (fa[:, None] - fc[None, :]) / np.where(same, 1.0, dE))
    F = amp * amp * ratio
    centre = 0.5 * (n_a[:, None] + n_c[None, :])
    if smooth:
        # heat-kernel-like regulator exp(-centre/N) ~ exp(-E^2/Lambda^2): smooth in N
        return float(np.sum(F * np.exp(-centre / n_low)))
    # pair-centre cutoff (n_a + n_c)/2 <= n_low: the same rule at q and at q = 0
    return float(F[centre <= n_low].sum())


def polarization(q, mass, field, chemical, temperature, n_cut=200, smooth=False):
    """(Pi_eps(q), Pi_omega(q)) with Landau levels n <= n_cut (UV divergent; use differences)."""
    xi = q * q / (2 * field)
    top = 14 * n_cut if smooth else n_cut           # exp(-14) ~ 1e-6 weight at the top
    window = int(4 * math.sqrt(xi * (top + 1)) + 4 * xi + 40) if xi > 0 else 0
    n_all = top + window
    Dm = displacement(math.sqrt(xi), n_all + 1) if xi > 0 else np.eye(n_all + 1)
    st = levels(mass, field, n_all)
    d = field / (2 * math.pi)
    pe = sum(_channel(st[b], st[b], Dm, chemical, temperature, n_cut, smooth) for b in (1, 2))
    pw = sum(_channel(st[b], st[c], Dm, chemical, temperature, n_cut, smooth) for b, c in ((1, 2), (2, 1)))
    return 0.5 * d * pe, 0.5 * d * pw


def kernels(q, mass, field, chemical, temperature, cutoffs=(50, 100, 200)):
    """UV-finite kernels at wave vector q > 0, extrapolated in the regulator N.

    Returns (Phi_pi, R): Phi_pi = Pi_omega(q) - Pi_omega(0) (Goldstone kernel, gap equation used)
    and R = [Pi_eps(q) - Pi_omega(q)] / (2 M^2), so that for a background (sigma, pi_0)

        Phi(q) = Phi_pi(q) * 1 + 2 R(q) * v v^T,   v = (sigma, pi_0),

    eigenvalues Phi_pi (Goldstone) and Phi_pi + 2 M^2 R (radial). Smooth regulator exp(-centre/N);
    the N -> infinity limit is taken by fitting a + b N^{-1/2} + c N^{-1} through three cutoffs.
    """
    rows = []
    for N in cutoffs:
        e0, w0 = polarization(0.0, mass, field, chemical, temperature, n_cut=N, smooth=True)
        eq, wq = polarization(q, mass, field, chemical, temperature, n_cut=N, smooth=True)
        rows.append((wq - w0, (eq - wq) / (2 * mass * mass)))
    A = np.array([[1.0, N ** -0.5, 1.0 / N] for N in cutoffs])
    vals = np.array(rows)
    coef = np.linalg.solve(A, vals)
    return float(coef[0, 0]), float(coef[0, 1])
