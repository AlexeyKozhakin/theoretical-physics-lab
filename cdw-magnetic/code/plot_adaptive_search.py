"""Plot sampled candidate statuses, explicitly not a phase diagram."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cdw-magnetic/code/adaptive-search-pilot.json")
    parser.add_argument("--output", default="cdw-magnetic/figures/adaptive-search-pilot.png")
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text())
    if not report.get("completed"):
        raise RuntimeError("Pilot checkpoint is incomplete")
    styles = {
        "sample_above_homogeneous_reference": ("#3577a8", "o", 14),
        "improving_candidate": ("#bc3038", "^", 45),
        "unresolved": ("#d98b1e", "s", 36),
        "homogeneous_reference_family": ("#8b9399", "x", 24),
        "symmetric_representative": ("#30343b", "+", 45),
        "numerical_failure": ("#d98b1e", "s", 36),
        "insufficient_basis": ("#d98b1e", "s", 36),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, case in zip(axes.flat, report["cases"]):
        for status, (color, marker, size) in styles.items():
            points = [point for point in case["points"] if point["status"] == status]
            if points:
                ax.scatter([p["mass"] for p in points], [p["wave"] for p in points],
                           color=color, marker=marker, s=size, alpha=0.9, linewidths=0.9)
        ax.set_title(rf"$|eB|={case['B']},\ \mu={case['mu']}$" +
                     f"\n{len(case['points'])} проверенных точек", fontsize=11)
        ax.set_xlim(-0.035, case["domain"]["mass"][1] + 0.035)
        ax.set_ylim(-0.035, case["domain"]["wave"][1] + 0.035)
        ax.grid(alpha=0.15)
        if case["improving_candidates"]:
            ax.annotate("Кандидаты с отрицательной\nверхней границей", xy=(0.1, 0.8),
                        xytext=(0.65, 1.45), fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": "#bc3038"}, color="#9b232a")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$M/m_*$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$b/m_*$")
    legend = [Line2D([], [], marker="o", linestyle="", color="#3577a8", markersize=5,
                     label="Проверенная точка выше однородного минимума"),
              Line2D([], [], marker="^", linestyle="", color="#bc3038", markersize=7,
                     label="Кандидат ниже однородного минимума"),
              Line2D([], [], marker="s", linestyle="", color="#d98b1e", markersize=6,
                     label="Знак не определён с заданной точностью"),
              Line2D([], [], marker="x", linestyle="", color="#8b9399", markersize=6,
                     label="Однородные фоны"),
              Line2D([], [], marker="+", linestyle="", color="#30343b", markersize=7,
                     label=r"$M=0$: модуляция физически несущественна")]
    fig.suptitle(r"Адаптивные проверочные сечения: $T/m_*=0.02$, $g<0$" +
                 "\nЭто конечный поиск кандидатов, не фазовая диаграмма", fontsize=13, y=0.98)
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.015))
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    print(f"Wrote {destination}")
