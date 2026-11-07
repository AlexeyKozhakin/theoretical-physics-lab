"""Second variation at B = 0 for arbitrary mu and T (note stability/02): the reference for B != 0.

Signs of the stability section (D = i d-slash + mu gamma^0 + sigma + i gamma^5 pi_0). Static
fluctuations X = (eps, omega); per unit area and fermion copy

    Delta Omega = int d^2q/(2pi)^2  X(q)^dagger  Phi(q)  X(q),     Delta Omega = -S^(2)/(N int dx^0),

    M_0 != 0 (gap equation):  Phi = Gamma(q) R(q),
        Gamma = [[2 sigma^2 + q^2/2, 2 sigma pi_0], [2 sigma pi_0, 2 pi_0^2 + q^2/2]],
        eigenvalues lambda_1 = q^2/2 (Goldstone), lambda_2 = 2 M_0^2 + q^2/2 (radial);
    M_0 = 0 (no gap equation):  Phi = [G(0+)/2 + (q^2/2) R(q)|_{M -> 0}] * identity.

    R(q) = int d^2p/(2pi)^2 [f(E)/E - f(E')/E'] / (E'^2 - E^2),
    E = sqrt(M^2 + p^2),  E' = sqrt(M^2 + (p + q)^2),  f = 1 - n_F(E - mu) - n_F(E + mu).

Closed forms (p_F = sqrt(mu^2 - M^2)):
    R_vac = arctan(q / 2M) / (2 pi q)                                       (f = 1)
    R_mat = -(1/(pi q)) int_M^{sqrt(M^2 + q^2/4)} dE w(E) / sqrt(q^2 + 4M^2 - 4E^2),  w = 1 - f
    T = 0:  R = 0 for q <= 2 p_F;   R = [arctan(q/2M) - arcsin(2mu/Q) + arcsin(2M/Q)] / (2 pi q),
            Q = sqrt(q^2 + 4 M^2), for q > 2 p_F (or mu <= M: R = R_vac).
"""

import math

import numpy as np
from scipy.integrate import quad

from ..common import COUPLING, occupation, thermal_weight


def r_vacuum(q, mass):
    """R with f = 1: int d^2p/(2pi)^2 1/(E E' (E + E')) = arctan(q/2M)/(2 pi q)."""
    q, M = float(q), float(mass)
    if q == 0:
        return math.inf if M == 0 else 1 / (4 * math.pi * M)
    if M == 0:
        return 1 / (4 * q)
    return math.atan(q / (2 * M)) / (2 * math.pi * q)


def r_matter(q, mass, chemical, temperature):
    """Matter part (f -> f - 1 = -w); one-dimensional integral, closed form at T = 0."""
    q, M, mu = float(q), float(mass), abs(float(chemical))
    if q == 0:
        raise ValueError("use q > 0; the q -> 0 limit is r_at_zero()")
    top = math.sqrt(M * M + q * q / 4)
    Q2 = q * q + 4 * M * M
    if temperature == 0:
        if mu <= M:
            return 0.0
        upper = min(mu, top)
        if M == 0:
            lower_angle = 0.0
        else:
            lower_angle = math.asin(2 * M / math.sqrt(Q2))
        return -(math.asin(min(1.0, 2 * upper / math.sqrt(Q2))) - lower_angle) / (2 * math.pi * q)
    # T > 0: substitute E = (Q/2) sin(theta) to remove the endpoint singularity
    a = math.asin(min(1.0, 2 * M / math.sqrt(Q2))) if M > 0 else 0.0

    def integrand(theta):
        E = 0.5 * math.sqrt(Q2) * math.sin(theta)
        return float(occupation(E, mu, temperature))

    pts = []
    for e in (mu,):
        if M < e < top:
            pts.append(math.asin(2 * e / math.sqrt(Q2)))
    value, _ = quad(integrand, a, math.pi / 2, points=pts or None, limit=200, epsabs=1e-13, epsrel=1e-11)
    return -value / (2 * math.pi * q)


def r_at_zero(mass, chemical, temperature):
    """R(q -> 0) = -int d^2p/(2pi)^2 d(f/E)/d(E^2) = f(M)/(4 pi M)."""
    M = float(mass)
    f = 1 - float(occupation(M, chemical, temperature))
    return f / (4 * math.pi * M)


def r_function(q, mass, chemical, temperature):
    """R(q) = R_vac + R_mat for q > 0."""
    return r_vacuum(q, mass) + r_matter(q, mass, chemical, temperature)


def gap_at_zero(chemical, temperature, coupling=COUPLING):
    """G(0+) = Omega'(M)/M at M -> 0 for B = 0: 1/g + H(0)/pi."""
    return 1 / coupling + float(thermal_weight(0.0, chemical, temperature)) / math.pi


def modes(q, sigma, pi0, chemical, temperature, coupling=COUPLING):
    """(Phi_Goldstone, Phi_radial) at wave vector q for the homogeneous background (sigma, pi0).

    M_0 != 0: (q^2/2) R and (2 M_0^2 + q^2/2) R (gap equation assumed to hold);
    M_0 = 0: both equal G(0+)/2 + (q^2/2) R|_{M -> 0}.
    """
    M = math.hypot(sigma, pi0)
    if M == 0:
        value = 0.5 * gap_at_zero(chemical, temperature, coupling) + 0.5 * q * q * r_function(q, 0.0, chemical, temperature)
        return value, value
    R = r_function(q, M, chemical, temperature)
    return 0.5 * q * q * R, (2 * M * M + 0.5 * q * q) * R
