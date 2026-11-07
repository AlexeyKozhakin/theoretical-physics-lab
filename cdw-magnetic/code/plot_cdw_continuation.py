"""Plot confirmed fixed points on two cuts, without global phase labels."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    report = json.loads(Path("cdw-magnetic/code/cdw-branch-continuation.json").read_text())
    if not report["completed"]:
        raise RuntimeError("Continuation checkpoint is incomplete")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, name, coordinate, label, title in (
        (axes[0, 0], "mu_cut", "mu", r"$\mu/m_*$", r"$|eB|=0.2$: подтверждённые точки"),
        (axes[0, 1], "field_cut", "B", r"$|eB|/m_*^2$", r"$\mu=1.05$: подтверждённые точки")):
        cases = [x for x in report[name] if x["status"] == "confirmed_improving_point"]
        x = np.array([c[coordinate] for c in cases])
        low = np.array([c["validation"][-1]["gap_lower"] for c in cases])*1e5
        high = np.array([c["validation"][-1]["gap_upper"] for c in cases])*1e5
        ax.errorbar(x, (low+high)/2, yerr=(high-low)/2, fmt="o", color="#276a94",
                    capsize=4, label="Полные оценки в выбранных фонах")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$10^5[\Omega(M,b)-\Omega_{\rm hom,min}]/m_*^3$")
        ax.set_xlim(min(c[coordinate] for c in report[name])-.005,
                    max(c[coordinate] for c in report[name])+.005)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for n in (2, 3, 4):
        value = 1.05**2/(2*n)
        axes[0, 1].axvline(value, color="#aaa", linestyle=":", linewidth=1)
    axes[0, 1].text(.04, .08, "Пунктир: пороги Ландау при M=0", fontsize=8,
                     transform=axes[0, 1].transAxes)

    ax = axes[1, 0]
    cases = report["mu_cut"]
    improving = [c for c in cases if c["status"] == "confirmed_improving_point"]
    ax.plot([c["mu"] for c in improving], [c["selected_point"]["mass"] for c in improving],
            "o-", color="#276a94", label="M локального кандидата")
    ax.plot([c["mu"] for c in improving], [c["selected_point"]["wave"] for c in improving],
            "s-", color="#c07c22", label="b локального кандидата")
    ax.plot([c["mu"] for c in cases], [c["homogeneous_reference"]["mass"] for c in cases],
            ".--", color="#777777", label="M однородного опорного минимума")
    ax.set_title(r"Положение найденных минимумов при $|eB|=0.2$", fontsize=11)
    ax.set_xlabel(r"$\mu/m_*$")
    ax.set_ylabel(r"$M/m_*,\ b/m_*$")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    zero = report["zero_field_control"]["fixed_backgrounds"][0]
    original = json.loads(Path("cdw-magnetic/code/cdw-candidate-refinement.json").read_text())
    check = original["validation"][-1]
    lo, hi = check["gap_lower"]*1e5, check["gap_upper"]*1e5
    ax.bar([0], [zero["gap"]*1e5], color="#c07c22", width=.5)
    ax.errorbar([1], [(lo+hi)/2], yerr=[(hi-lo)/2], fmt="o", color="#276a94", capsize=8)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks([0, 1], ["B = 0", "|eB| = 0.2"])
    ax.set_ylabel(r"$10^5[\Omega(M,b)-\Omega_{\rm hom,min}]/m_*^3$")
    ax.set_title("Один и тот же неоднородный фон\nM ≈ 0.1120, b ≈ 0.79954, μ = 1.05", fontsize=11)
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("Два сечения локального поиска при T = 0.02\n"
                 "Показаны подтверждённые кандидаты; пустые участки не исключают CDW", fontsize=13)
    destination = Path("cdw-magnetic/figures/cdw-branch-continuation.png")
    fig.savefig(destination, dpi=180)
    print(f"Wrote {destination}")
