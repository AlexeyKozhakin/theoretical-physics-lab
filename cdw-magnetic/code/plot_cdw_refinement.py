"""Plot a local search landscape and independent fixed-point basis bounds."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cdw-magnetic/code/cdw-candidate-refinement.json")
    parser.add_argument("--output", default="cdw-magnetic/figures/cdw-candidate-refinement.png")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text())
    surface = report["local_surface"]
    masses, waves = np.array(surface["mass_axis"]), np.array(surface["wave_axis"])
    values = np.array(surface["corrected_gap"]) * 1e5
    point = report["best_local_point"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    levels = np.r_[np.linspace(-2.5, 0, 11), np.linspace(2, 24, 12)]
    picture = left.contourf(masses, waves, values, levels=levels, cmap="RdBu_r", extend="both")
    zero = left.contour(masses, waves, values, levels=[0], colors="#343434", linewidths=1)
    left.clabel(zero, fmt={0: "0"}, fontsize=9)
    left.plot(point["mass"], point["wave"], "*", color="#131313", markersize=12)
    left.set_xlabel(r"$M/m_*$")
    left.set_ylabel(r"$b/m_*$")
    left.set_title("Окрестность локального минимума\nПриближённый потенциал с поправкой хвоста", fontsize=11)
    fig.colorbar(picture, ax=left, label=r"$10^5\,\delta_{\rm corrected}/m_*^3$", shrink=0.85)

    rows = report["validation"]
    N = np.array([row["N"] for row in rows])
    lower = np.array([row["gap_lower"] for row in rows]) * 1e5
    upper = np.array([row["gap_upper"] for row in rows]) * 1e5
    center = (lower + upper) / 2
    right.errorbar(N, center, yerr=(upper - lower) / 2, fmt="o", color="#2d6b92",
                   capsize=7, label="Границы с учётом хвостов базиса")
    right.plot(N, [row["corrected_gap"] * 1e5 for row in rows], "s--", color="#bb3038",
               markersize=5, label="Приближённое значение для поиска")
    right.axhline(0, color="#444444", linewidth=1)
    right.set_xscale("log", base=2)
    right.set_xticks(N, labels=[str(n) for n in N])
    right.set_ylim(min(lower) - 0.25, 0.25)
    right.set_xlabel("N состояний в каждой башне")
    right.set_ylabel(r"$10^5[\Omega(M,b)-\Omega_{\rm hom,min}]/m_*^3$")
    right.set_title("Независимая проверка выбранного фона\nВерхняя граница остаётся ниже нуля", fontsize=11)
    right.grid(alpha=0.2)
    right.legend(fontsize=8, loc="upper right")
    fig.suptitle(rf"$|eB|={report['B']},\ \mu={report['mu']},\ T={report['T']}$; "
                 rf"$M\approx{point['mass']:.5f},\ b\approx{point['wave']:.5f}$" +
                 "\nЛокальный поиск в одном сечении; фазовые границы пока не определены", fontsize=12)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    print(f"Wrote {destination}")
