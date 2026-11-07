"""Homogeneous NJL3 reference, per area and copy; B means |eB|.

The LL-tail bound is analytic. SciPy quadrature/minimization estimates and
floating roundoff are separate from that bound. A grid-refined 1D minimum
is a numerical search on an explicitly reported finite mass interval.
"""

from dataclasses import dataclass
import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def thermal_weight(energy, chemical, temperature):
    E = np.asarray(energy, dtype=float)
    if temperature == 0:
        return np.maximum(abs(chemical) - E, 0)
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    return temperature * (np.logaddexp(0, (chemical - E) / temperature)
                          + np.logaddexp(0, (-chemical - E) / temperature))


@dataclass(frozen=True)
class LadderSum:
    lower: float
    upper: float
    last_landau_index: int


def ladder_thermal_sum(field, offset, chemical, temperature, start_rank=0, tolerance=1e-12):
    """Sum H(sqrt(max(offset+2Bn,0))) over homogeneous ranks j>=start.

    Ranks are j=0 -> n=0; j=2n-1,2n -> n>=1. Clamping negative
    squared energies is used only for spectral lower comparison bounds.
    """
    if field <= 0 or temperature < 0 or start_rank < 0 or tolerance <= 0:
        raise ValueError("require B>0, T>=0, start_rank>=0 and tolerance>0")
    chemical = abs(float(chemical))
    start_rank = int(start_rank)
    first = (start_rank + 1) // 2
    if temperature == 0:
        last = max(first - 1, math.floor((chemical ** 2 - offset) / (2 * field)))
        tail = 0.0
    else:
        exponent_margin = 32.0
        while True:
            target = chemical + exponent_margin * temperature
            last = max(first, math.ceil((target ** 2 - offset) / (2 * field)))
            last_energy = math.sqrt(max(offset + 2 * field * last, 0))
            tail = (2 * temperature ** 2 / field * (last_energy + temperature)
                    * (1 + math.exp(-2 * chemical / temperature))
                    * math.exp((chemical - last_energy) / temperature))
            if tail <= tolerance:
                break
            exponent_margin += 8
    if last < first:
        return LadderSum(0.0, 0.0, last)
    n = np.arange(first, last + 1)
    energies = np.sqrt(np.maximum(offset + 2 * field * n, 0))
    multiplicities = np.full(len(n), 2, dtype=int)
    if first == 0:
        multiplicities[0] = 1 if start_rank == 0 else 0
    positive = n >= 1
    multiplicities[positive] = ((2 * n[positive] - 1 >= start_rank).astype(int)
                                + (2 * n[positive] >= start_rank).astype(int))
    value = float(np.sum(multiplicities * thermal_weight(energies, chemical, temperature)))
    return LadderSum(value, value + tail, last)


def vacuum(field, mass, coupling=-math.pi):
    if field < 0 or coupling == 0:
        raise ValueError("require B>=0 and nonzero coupling")
    M = abs(float(mass))
    if field == 0:
        return M ** 2 / (2 * coupling) + M ** 3 / (3 * math.pi)
    with mp.workdps(32):
        B, m = mp.mpf(str(field)), mp.mpf(str(M))
        z = mp.zeta(-mp.mpf("0.5"), m ** 2 / (2 * B))
        return float(m ** 2 / (2 * coupling) + m * B / (2 * mp.pi)
                     - (2 * B) ** mp.mpf("1.5") * z / (2 * mp.pi))


def potential(field, mass, chemical, temperature, coupling=-math.pi, tolerance=1e-11):
    """Return (Omega, thermal-tail/quadrature estimated error)."""
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    M, mu = abs(float(mass)), abs(float(chemical))
    base = vacuum(field, M, coupling)
    if field > 0:
        ladder = ladder_thermal_sum(field, M ** 2, mu, temperature, tolerance=tolerance)
        degeneracy = field / (2 * math.pi)
        return base - degeneracy * ladder.lower, degeneracy * (ladder.upper - ladder.lower)
    if temperature == 0:
        gain = max(mu - M, 0) ** 2 * (mu + 2 * M) / (6 * math.pi)
        return base - gain, 0.0
    value, error = quad(lambda E: E * float(thermal_weight(E, mu, temperature)),
                        M, max(M, mu), epsabs=tolerance / 2, epsrel=tolerance)
    tail, tail_error = quad(lambda E: E * float(thermal_weight(E, mu, temperature)),
                            max(M, mu), np.inf, epsabs=tolerance / 2, epsrel=tolerance)
    return base - (value + tail) / math.pi, (error + tail_error) / math.pi


@dataclass(frozen=True)
class HomogeneousMinimum:
    mass: float
    potential: float
    mass_interval: tuple
    candidates: tuple
    grid_points: int
    upper_boundary_selected: bool


def minimize_mass(field, chemical, temperature, coupling=-math.pi, grid_points=81, mass_max=None):
    if grid_points < 5:
        raise ValueError("grid_points must be at least 5")
    scale = math.pi / abs(coupling)
    if mass_max is None:
        mass_max = 2 * (scale + math.sqrt(field) + abs(chemical) + temperature)
    grid = np.linspace(0, mass_max, grid_points)
    # Include filling thresholds even at T>0, where they seed narrow features.
    if field > 0:
        indices = np.arange(math.floor(chemical ** 2 / (2 * field)) + 1)
        thresholds = np.sqrt(np.maximum(chemical ** 2 - 2 * field * indices, 0))
        grid = np.unique(np.r_[grid, thresholds[thresholds <= mass_max]])

    def objective(M):
        return potential(field, M, chemical, temperature, coupling)[0]

    values = np.array([objective(mass) for mass in grid])
    candidates = [(0.0, float(values[0])), (float(mass_max), float(values[-1]))]
    # A small broken minimum can lie inside the first grid cell while both
    # grid endpoints lie above the symmetric state.
    near_zero = minimize_scalar(objective, bounds=(0, grid[1]), method="bounded",
                                options={"xatol": 2e-10})
    candidates.append((float(near_zero.x), float(near_zero.fun)))
    for index in range(1, len(grid) - 1):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            optimum = minimize_scalar(objective, bounds=(grid[index - 1], grid[index + 1]),
                                      method="bounded", options={"xatol": 2e-10})
            candidates.append((float(optimum.x), float(optimum.fun)))
            candidates.append((float(grid[index]), float(values[index])))
    best_mass, best_value = min(candidates, key=lambda pair: pair[1])
    return HomogeneousMinimum(best_mass, best_value, (0.0, float(mass_max)), tuple(candidates),
                              len(grid), best_mass >= mass_max - 1e-6)
