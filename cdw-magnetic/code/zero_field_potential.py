"""Independent B=0 CDW potential: analytic vacuum and transverse integral.

Finite temperature is a convolution of the T=0 medium term. Quadrature
errors are numerical estimates; the high-chemical-potential tail is bounded.
No finite-B Landau truncation is used here.
"""

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import expit


def vacuum(mass, wave, coupling=-math.pi):
    M, b = abs(float(mass)), abs(float(wave))
    if coupling == 0:
        raise ValueError("Coupling must be nonzero")
    if M == 0:
        return 0.0
    if b >= M:
        return M ** 2 / (2 * coupling) + M ** 2 * b / 4
    root = math.sqrt(M ** 2 - b ** 2)
    return (M ** 2 / (2 * coupling)
            + ((2 * M ** 2 + b ** 2) * root + 3 * M ** 2 * b * math.asin(b / M)) / (6 * math.pi))


def transverse_medium(gap, chemical):
    """Integral_0^infinity dy (nu-sqrt(y^2+a^2))_+."""
    a, nu = abs(float(gap)), abs(float(chemical))
    if a >= nu:
        return 0.0
    if a == 0:
        return nu ** 2 / 2
    root = math.sqrt((nu - a) * (nu + a))
    return (nu * root - a ** 2 * math.acosh(nu / a)) / 2


def medium_zero_temperature(mass, wave, chemical, tolerance=1e-11):
    M, b, nu = abs(float(mass)), abs(float(wave)), abs(float(chemical))
    if nu == 0:
        return 0.0, 0.0
    intervals = []
    if nu > M + b:
        intervals.append((1, 0.0, math.sqrt((nu - b) ** 2 - M ** 2)))
    if nu + b > M:
        low = max(M, b - nu)
        intervals.append((-1, math.sqrt(max(0.0, low ** 2 - M ** 2)),
                          math.sqrt((nu + b) ** 2 - M ** 2)))
    result, estimate = 0.0, 0.0
    for sign, lower, upper in intervals:
        crossing = math.sqrt(max(0.0, b ** 2 - M ** 2))
        points = [crossing] if sign == -1 and lower < crossing < upper else None
        value, error = quad(lambda x: transverse_medium(math.hypot(x, M) + sign * b, nu),
                            lower, upper, points=points, epsabs=tolerance, epsrel=tolerance, limit=180)
        result += value
        estimate += error
    return -result / math.pi ** 2, estimate / math.pi ** 2


def potential(mass, wave, chemical, temperature, coupling=-math.pi, tolerance=2e-10):
    M, b, mu = abs(float(mass)), abs(float(wave)), abs(float(chemical))
    if temperature < 0 or tolerance <= 0:
        raise ValueError("Require T>=0 and positive tolerance")
    base = vacuum(M, b, coupling)
    if temperature == 0:
        medium, error = medium_zero_temperature(M, b, mu, tolerance)
        return base + medium, error
    T = temperature
    # The medium integral is convolved, while the vacuum remains exact.
    upper = mu + 36 * T
    while True:
        polynomial = (upper ** 3 + 3 * upper ** 2 * T + 6 * upper * T ** 2 + 6 * T ** 3
                      + 2 * b * (upper ** 2 + 2 * upper * T + 2 * T ** 2)
                      + b ** 2 * (upper + T))
        tail_bound = math.exp((mu - upper) / T) * (1 + math.exp(-2 * mu / T)) * polynomial / (2 * math.pi)
        if tail_bound < tolerance / 4:
            break
        upper += 8 * T
    maximum_inner_error = 0.0

    def integrand(nu):
        nonlocal maximum_inner_error
        medium, error = medium_zero_temperature(M, b, nu, tolerance / 10)
        maximum_inner_error = max(maximum_inner_error, error)
        e1, e2 = expit(-abs((nu - mu) / T)), expit(-(nu + mu) / T)
        kernel = (e1 * (1 - e1) + e2 * (1 - e2)) / T
        return kernel * medium

    knots = [mu + k * T for k in (-12, -6, 0, 6, 12)] + [abs(M - b), M + b]
    knots = sorted(set(x for x in knots if 0 < x < upper))
    value, error = quad(integrand, 0, upper, points=knots, epsabs=tolerance / 4,
                        epsrel=tolerance, limit=200)
    return base + value, error + maximum_inner_error + tail_bound


if __name__ == "__main__":
    from homogeneous_potential import potential as homogeneous

    # Independent limits, including thermal particle/antiparticle weights.
    for M, mu, T in ((0, 1.05, 0.02), (0.7, 0.8, 0), (0.4, 0.8, 0.2), (1.1, 0, 0.1)):
        got, error = potential(M, 0, mu, T)
        expected, other_error = homogeneous(0, M, mu, T)
        assert abs(got - expected) < error + other_error + 2e-9
    for b in (0.2, 0.8, 1.4):
        got, error = potential(0, b, 1.05, 0.02)
        expected = homogeneous(0, 0, 1.05, 0.02)[0]
        assert abs(got - expected) < error + 2e-9
    for M, b, nu in ((0.1, 0.8, 1.05), (0.7, 0.4, 1.4), (0.3, 1.2, 2.0)):
        value, error = potential(M, b, nu, 0)
        expected = M ** 2 / (-2 * math.pi) + nu * M ** 2 / (2 * math.pi) - nu ** 3 / (6 * math.pi)
        assert abs(value - expected) < error + 2e-9
    print("PASS B=0: homogeneous limits, M=0 independence in b, and filled-branch T=0 identity.")
    M, b = 0.11199624009403308, 0.7995368100527627
    value, error = potential(M, b, 1.05, 0.02)
    reference = homogeneous(0, 0, 1.05, 0.02)[0]
    print(f"B=0 control at M={M}, b={b}: gap={value-reference:.12g}, estimated error={error:.3g}")
