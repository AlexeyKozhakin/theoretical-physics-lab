"""Independent polar-momentum checks of the B=0 vacuum and thermal part."""

import json
import math
from pathlib import Path

from scipy.integrate import quad

from homogeneous_potential import thermal_weight
from zero_field_potential import potential, vacuum


def energies(p, angle, M, b):
    longitudinal = math.hypot(p * math.cos(angle), M)
    transverse = p * math.sin(angle)
    return math.hypot(transverse, longitudinal + b), math.hypot(transverse, longitudinal - b)


def radial_vacuum(M, b, cutoff):
    def angular(angle):
        def integrand(p):
            full = sum(energies(p, angle, M, b))
            free = sum(energies(p, angle, 0, b))
            return p * (full - free) - M ** 2
        points = sorted(set(x for x in (b, math.sqrt(max(0, b ** 2 - M ** 2))) if 0 < x < cutoff))
        return quad(integrand, 0, cutoff, points=points, epsabs=2e-9, epsrel=2e-9, limit=160)[0]
    integral, estimate = quad(angular, 0, math.pi / 2, epsabs=2e-8, epsrel=2e-8, limit=160)
    leading_tail = (M ** 4 + 3 * M ** 2 * b ** 2) / (8 * math.pi * cutoff)
    return M ** 2 / (-2 * math.pi) - integral / math.pi ** 2 + leading_tail, estimate / math.pi ** 2


def polar_thermal(M, b, mu, T):
    radius = mu + b + 36 * T
    max_inner_error = 0.0
    def angular(angle):
        nonlocal max_inner_error
        def integrand(p):
            plus, minus = energies(p, angle, M, b)
            return p * float(thermal_weight(plus, mu, T) + thermal_weight(minus, mu, T))
        points = sorted(set(x for x in (b, abs(mu-b), mu+b) if 0 < x < radius))
        value, error = quad(integrand, 0, radius, points=points, epsabs=2e-9, epsrel=2e-9, limit=200)
        max_inner_error = max(max_inner_error, error)
        return value
    integral, estimate = quad(angular, 0, math.pi / 2, epsabs=2e-9, epsrel=2e-9, limit=160)
    tail = T ** 2 * (radius + T) / math.pi * (1 + math.exp(-2*mu/T)) * math.exp((mu+b-radius)/T)
    return -integral / math.pi ** 2, (estimate + math.pi/2*max_inner_error) / math.pi ** 2 + tail


if __name__ == "__main__":
    vacuum_checks = []
    for M, b in ((0.7, 0.4), (0.3, 0.8), (0.5, 0.5)):
        expected = vacuum(M, b)
        rows = []
        for cutoff in (20, 40):
            value, error = radial_vacuum(M, b, cutoff)
            rows.append({"cutoff": cutoff, "value": value, "quadrature_error_estimate": error,
                         "difference_from_analytic": value - expected})
        assert abs(rows[-1]["difference_from_analytic"]) < 2e-7
        assert abs(rows[-1]["difference_from_analytic"]) <= abs(rows[0]["difference_from_analytic"]) / 4 + 2e-9
        vacuum_checks.append({"M": M, "b": b, "analytic": expected, "radial_checks": rows})
    thermal_checks = []
    for M, b, mu, T in ((0.11199624, 0.79953681, 1.05, 0.02), (0.7, 0.4, 0.8, 0.2),
                         (0.3, 0.8, 0, 0.1)):
        value, error = potential(M, b, mu, T)
        medium, other_error = polar_thermal(M, b, mu, T)
        direct = vacuum(M, b) + medium
        assert abs(value - direct) < error + other_error + 2e-9
        thermal_checks.append({"M": M, "b": b, "mu": mu, "T": T, "convolution": value,
                               "polar_direct": direct, "difference": value-direct,
                               "error_estimates_sum": error + other_error})
    output = Path("cdw-magnetic/code/zero-field-potential-checks.json")
    output.write_text(json.dumps({"vacuum": vacuum_checks, "thermal": thermal_checks}, indent=2) + "\n")
    print("PASS B=0: analytic vacuum vs independently subtracted radial integral; thermal convolution vs direct polar integral.")
    print(f"Wrote {output}")
