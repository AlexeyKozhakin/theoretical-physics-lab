"""Shared numerics: thermal weights, Fermi functions, Hurwitz zeta (vectorized)."""

import math

import numpy as np
from scipy.special import expit

COUPLING = -math.pi            # g; m_* = pi / |g| = 1


def thermal_weight(energy, chemical, temperature):
    """H(E) = T ln(1+e^{-(E-mu)/T}) + T ln(1+e^{-(E+mu)/T}); T = 0: (|mu| - E)_+."""
    E = np.asarray(energy, dtype=float)
    if temperature == 0:
        return np.maximum(abs(chemical) - E, 0.0)
    return temperature * (np.logaddexp(0, (chemical - E) / temperature)
                          + np.logaddexp(0, (-chemical - E) / temperature))


def occupation(energy, chemical, temperature):
    """w(E) = -dH/dE = n_F(E - mu) + n_F(E + mu); T = 0: theta(|mu| - E) (1/2 at E = |mu|)."""
    E = np.asarray(energy, dtype=float)
    if temperature == 0:
        mu = abs(chemical)
        return np.where(E < mu, 1.0, np.where(E == mu, 0.5, 0.0))
    return expit((chemical - E) / temperature) + expit((-chemical - E) / temperature)


# Bernoulli numbers B_2, B_4, ..., B_20 for the Euler-Maclaurin tail.
_B2K = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510,
        43867 / 798, -174611 / 330]


def hurwitz_zeta(s, x, terms=12):
    """zeta(s, x) = sum_k (x + k)^{-s}, x > 0, any real s != 1 (analytic continuation).

    Euler-Maclaurin: direct sum of `terms` terms, then the integral, the half term and
    10 Bernoulli corrections at x + terms >= 12. Relative accuracy ~1e-15.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("hurwitz_zeta requires x > 0")
    k = np.arange(terms)
    direct = np.sum((x[..., None] + k) ** (-s), axis=-1)
    N = x + terms
    tail = N ** (1 - s) / (s - 1) + 0.5 * N ** (-s)
    rising = s                                   # s (s+1) ... (s+2j-2)
    power = N ** (-s - 1)
    fact = 1.0
    for j, b in enumerate(_B2K, start=1):
        fact *= (2 * j - 1) * (2 * j) if j > 1 else 2
        tail = tail + b / fact * rising * power
        rising *= (s + 2 * j - 1) * (s + 2 * j)
        power = power / (N * N)
    return direct + tail
