"""Fast stability kernels in the field (note stability/05).

    Pi = Pi_vac (T = mu = 0) + Pi_mat (delta n = n_F(E - mu) - theta(-E)).

* Vacuum: depends on (q, M, B) only, and by dimensional analysis
      Phi_pi_vac(q, M, B) = sqrt(B) f(q/sqrt(B), M/sqrt(B)),  the same for DeltaPi_vac;
  f is tabulated once (code/stab_vacuum_table.py) and interpolated by bicubic splines of
  y * Phi_pi/x^2 and y * DeltaPi (finite at x -> 0 and at y -> 0, where both grow like 1/y).
  Outside the table: direct computation.
* Matter: finite sums over the occupied levels (landau.matter_polarization), milliseconds.

Kernels: Phi_pi(q) = Pi_omega(q) - Pi_omega(0) (Goldstone), R(q) = [Pi_eps(q) - Pi_omega(q)]/(2 M^2);
radial = Phi_pi + 2 M^2 R. For M = 0 use symmetric_kernel().
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..hom import formulas as hom
from .landau import matter_polarization, vacuum_kernels

TABLE = Path(__file__).resolve().parent / "data" / "vacuum_table.npz"


@lru_cache(maxsize=1)
def _table():
    data = np.load(TABLE)
    x, y = data["x"], data["y"]
    # Both kernels grow like 1/y at y = M/sqrt(B) -> 0 (lowest Landau level): interpolate y * f.
    yK = y[None, :] * data["phi_pi"] / (x[:, None] ** 2)
    yD = y[None, :] * data["delta_pi"]
    return (x, y, RectBivariateSpline(x, y, yK, kx=3, ky=3, s=0),
            RectBivariateSpline(x, y, yD, kx=3, ky=3, s=0))


def vacuum(q, mass, field):
    """(Phi_pi_vac, DeltaPi_vac) at (q, M, B)."""
    s = math.sqrt(field)
    X, Y = q / s, mass / s
    x, y, spK, spD = _table()
    if X <= x[-1] and y[0] <= Y <= y[-1]:
        Xc = max(X, x[0])                       # K is flat below the first node (O(x^2) change)
        return (s * float(spK(Xc, Y)[0, 0]) * X * X / Y, s * float(spD(Xc, Y)[0, 0]) / Y)
    p, d = vacuum_kernels(X, Y)
    return s * p, s * d


def kernels(q, mass, field, chemical, temperature):
    """(Phi_pi, R) for a background with M = mass != 0 that solves the gap equation."""
    pv, dv = vacuum(q, mass, field)
    e0, w0 = matter_polarization(0.0, mass, field, chemical, temperature)
    eq, wq = matter_polarization(q, mass, field, chemical, temperature)
    return pv + (wq - w0), (dv + eq - wq) / (2 * mass * mass)


def modes(q, mass, field, chemical, temperature):
    """(Goldstone, radial) eigen-kernels for M != 0."""
    phi, R = kernels(q, mass, field, chemical, temperature)
    return phi, phi + 2 * mass * mass * R


def symmetric_kernel(q, field, chemical, temperature, regulator_mass=1e-3):
    """M = 0: Phi = G(0+)/2 + [Pi(q) - Pi(0)] at M -> 0 (both channels equal)."""
    g0 = float(hom.gap(field, 0.0, chemical, temperature))
    m = regulator_mass
    pv, _ = vacuum(q, m, field)
    _, w0 = matter_polarization(0.0, m, field, chemical, temperature)
    _, wq = matter_polarization(q, m, field, chemical, temperature)
    return 0.5 * g0 + pv + (wq - w0)
