"""Certified search for all extrema of the homogeneous Omega(M) and its global minimum.

On [p, q] (0 < p < q) the regular split G = A - d_B u/M + D of formulas.gap_bounds
(A increasing, u = 1 - w non-decreasing in [0, 1], D non-increasing) gives

    A(p) - d_B u(q)/p + D(q) <= G(M) <= A(q) - d_B u(p)/q + D(p),

so intervals where these bounds have one sign contain no root and are discarded;
the rest are bisected (the same scheme as njl3.b0 in the zero-field work). At T = 0
G_th jumps down where a level crosses mu: such a jump can only produce a kink
maximum of Omega, never a minimum (Omega' = M G drops). Minima are sign changes
- -> + of the continuous G and are refined by brentq.

Where the bounds do not decide the sign and |G| < FLAT at the ends and the middle of an
interval (Omega flat to roundoff, as at mu ~ m_* in a weak field), the interval is not
bisected further; its ends enter as 'flat' candidates. Bisection stops at cells of
relative width 1e-5; a cell is then classified by the signs of G at its ends (a sign
change is refined by brentq). Two uncertified steps follow from this: flat stretches,
and pairs of roots closer than 1e-5 that would cancel within one cell.

Phase labels (by the occupation of Landau levels at the minimum, as V / I_j / T in
the (mu, L) work): T — M* = 0; V — M* > 0 and no level below mu; I_j — M* > 0 and
levels n = 0..j below mu.
"""

from dataclasses import dataclass, field as dc_field
import math

import numpy as np
from scipy.optimize import brentq

from ..common import COUPLING
from . import formulas as F

SMALL = 1e-7          # below this mass the stable form F.gap is used directly
DEGENERATE = 1e-12    # Omega(M) - Omega(0) above -DEGENERATE: M = 0 is kept
FLAT = 1e-9           # |G| below this at the ends and middle of an undecided interval: flat


@dataclass
class Minimum:
    field: float
    chemical: float
    temperature: float
    mass: float                       # M* of the global minimum
    omega_rel: float                  # Omega(M*) - Omega(0)
    phase: str                        # 'V', 'I<j>' or 'T'
    extrema: list = dc_field(default_factory=list)   # (M, 'min' | 'max' | 'touch' | 'flat'), M > 0
    intervals: int = 0                # bisection intervals examined


def mass_bound(field, chemical, temperature, coupling=COUPLING):
    """M_max with G > 0 on [M_max, inf): G_vac(M_max) > 0 and G_th >= 0."""
    M = 2 * (math.pi / abs(coupling) + math.sqrt(field) + abs(chemical) + temperature)
    while F.gap_parts(field, M, chemical, temperature, coupling)[0] <= 0:
        M *= 2
    return M


def extrema(field, chemical, temperature, coupling=COUPLING, width=1e-5):
    """All sign changes of G on (0, M_max]: list of (M, kind), plus the number of intervals."""
    M_max = mass_bound(field, chemical, temperature, coupling)
    found = []
    examined = 0
    # Stable form near zero: sign of G on (0, SMALL] from its two ends.
    g0 = F.gap(field, 0.0, chemical, temperature, coupling)
    gs = F.gap(field, SMALL, chemical, temperature, coupling)
    if g0 < 0 < gs:
        found.append((brentq(lambda m: F.gap(field, m, chemical, temperature, coupling),
                             1e-300, SMALL, xtol=1e-16), "min"))
    elif g0 > 0 > gs:
        found.append((SMALL / 2, "max"))
    stack = [(SMALL, M_max)]
    pending = []
    flat = []
    while stack:
        p, q = stack.pop()
        examined += 1
        lower, upper = F.gap_bounds(field, p, q, chemical, temperature, coupling)
        if lower > 0 or upper < 0:
            continue
        # Degenerate flat stretch (mu ~ m_* at weak field, where thermal smearing removes the
        # Landau oscillations): G vanishes to roundoff and no sign can be certified. Omega
        # changes by < FLAT M^2 there; the ends enter as candidates.
        values = F.gap(field, np.array([p, 0.5 * (p + q), q]), chemical, temperature, coupling)
        if np.all(np.abs(values) < FLAT):
            flat.append((p, q))
            continue
        if q - p > width * max(1.0, q):
            mid = 0.5 * (p + q)
            stack += [(mid, q), (p, mid)]
        else:
            pending.append((p, q))
    # Merge adjacent undecided cells and classify by the signs at their ends.
    pending.sort()
    merged = []
    for p, q in pending:
        if merged and abs(merged[-1][1] - p) <= 1e-15 * max(1.0, p):
            merged[-1] = (merged[-1][0], q)
        else:
            merged.append((p, q))
    for p, q in merged:
        gp = F.gap(field, p, chemical, temperature, coupling)
        gq = F.gap(field, q, chemical, temperature, coupling)
        if gp < 0 < gq:
            found.append((brentq(lambda m: F.gap(field, m, chemical, temperature, coupling),
                                 p, q, xtol=1e-15), "min"))
        elif gp > 0 > gq:
            found.append((0.5 * (p + q), "max"))
        elif gp == 0 or gq == 0 or (gp < 0 and gq < 0) or (gp > 0 and gq > 0):
            # tangency within the cell: report as a touching point (degenerate extremum)
            m = 0.5 * (p + q)
            if abs(F.gap(field, m, chemical, temperature, coupling)) < 1e-9:
                found.append((m, "touch"))
    flat.sort()
    merged_flat = []
    for p, q in flat:
        if merged_flat and abs(merged_flat[-1][1] - p) <= 1e-15 * max(1.0, p):
            merged_flat[-1] = (merged_flat[-1][0], q)
        else:
            merged_flat.append((p, q))
    for p, q in merged_flat:
        found += [(p, "flat"), (q, "flat")]
    found.sort()
    return found, examined


def phase_label(field, chemical, mass):
    if mass == 0:
        return "T"
    filled = int(np.sum(np.sqrt(mass ** 2 + 2 * field * np.arange(
        int(chemical ** 2 / (2 * field)) + 2)) < chemical))
    return "V" if filled == 0 else f"I{filled - 1}"


def find_minimum(field, chemical, temperature, coupling=COUPLING):
    found, examined = extrema(field, chemical, temperature, coupling)
    omega0 = float(F.potential(field, 0.0, chemical, temperature, coupling))
    best_mass, best = 0.0, 0.0
    for m, kind in found:
        if kind in ("min", "touch", "flat"):
            value = float(F.potential(field, m, chemical, temperature, coupling)) - omega0
            # A broken minimum must beat M = 0 by more than roundoff (degenerate flat cases).
            if value < best and value < -DEGENERATE:
                best_mass, best = m, value
    return Minimum(field, chemical, temperature, best_mass, best,
                   phase_label(field, chemical, best_mass), found, examined)
