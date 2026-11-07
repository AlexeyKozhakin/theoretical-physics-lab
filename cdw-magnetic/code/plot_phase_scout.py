"""Render the finite-search map with a clear distinction in validation level."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np


def edges(axis):
    x = np.array(axis)
    return np.r_[x[0]-(x[1]-x[0])/2,(x[1:]+x[:-1])/2,x[-1]+(x[-1]-x[-2])/2]


if __name__=="__main__":
    report = json.loads(Path("cdw-magnetic/code/phase-scout.json").read_text())
    if not report["completed"]:
        raise RuntimeError("Map checkpoint is incomplete")
    B, mu = report["field_axis"],report["chemical_axis"]
    shape = (len(B),len(mu))
    status = np.full(shape,3)
    gain, mass, wave = (np.full(shape,np.nan) for _ in range(3))
    codes = {"symmetric_provisional":0,"homogeneous_broken_provisional":1,
             "confirmed_cdw":2,"unresolved":3}
    confirmed_coordinates = []
    for row in report["rows"]:
        j = B.index(row["B"])
        for point in row["points"]:
            i = mu.index(point["mu"])
            status[j,i] = codes[point["status"]]
            if point["status"]=="confirmed_cdw":
                gain[j,i] = -point["validation"][-1]["upper"]
                mass[j,i] = point["selected_point"]["mass"]
                wave[j,i] = point["selected_point"]["wave"]
                confirmed_coordinates.append((point["mu"],point["B"]))
    xe,ye = edges(mu),edges(B)
    colors = ["#f1dfae","#90b5cc","#c94556","#777777"]
    fig,axes = plt.subplots(1,2,figsize=(13,6.5))
    axes[0].pcolormesh(xe,ye,status,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4),4),
                       shading="flat",rasterized=True)
    handles = [Patch(facecolor=colors[2],label="CDW: выигрыш проверен с хвостами базиса"),
               Patch(facecolor=colors[1],label="M > 0, b = 0: лидер быстрого поиска"),
               Patch(facecolor=colors[0],label="M = 0: лидер быстрого поиска"),
               Patch(facecolor=colors[3],label="Поиск или знак не разрешён")]
    axes[0].set_title("Предварительная карта\nПроверка CDW строже остальных меток",fontsize=11)
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.01),ncol=2,fontsize=9,
               frameon=False)
    positive = gain[np.isfinite(gain)]
    if len(positive):
        picture = axes[1].pcolormesh(xe,ye,np.ma.masked_invalid(gain),cmap="viridis",
                                    norm=LogNorm(vmin=positive.min(),vmax=positive.max()),shading="flat")
        fig.colorbar(picture,ax=axes[1],label=r"$-\delta_{\rm upper}/m_*^3>0$",shrink=.85)
    axes[1].set_title("Нижняя оценка выигрыша проверенных фонов\nБелое поле: выигрыш здесь не подтверждён",fontsize=11)
    for ax in axes:
        ax.set_xlabel(r"$\mu/m_*$")
        ax.set_ylabel(r"$|eB|/m_*^2$")
        ax.set_xlim(min(mu),max(mu))
        ax.set_ylim(min(B),max(B))
        if confirmed_coordinates:
            x,y=np.array(confirmed_coordinates).T
            ax.scatter(x,y,s=3,color="#202020",alpha=.7)
    counts = report["counts"]
    fig.suptitle(r"$T/m_*=0.02,\ g=-\pi$: быстрый обзор $20\times25$ точек"+
                 f"\n{counts.get('confirmed_cdw',0)} проверенных CDW-точек; "
                 "границы между ячейками не являются установленными границами фаз",fontsize=12)
    fig.tight_layout(rect=(0,.13,1,.9))
    destination = Path("cdw-magnetic/figures/phase-scout.png")
    fig.savefig(destination,dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,values,label in zip(axes,(mass,wave),(r"$M/m_*$",r"$b/m_*$")):
        picture=ax.pcolormesh(xe,ye,np.ma.masked_invalid(values),cmap="viridis",shading="flat")
        fig.colorbar(picture,ax=ax,label=label)
        ax.set_xlabel(r"$\mu/m_*$")
        ax.set_ylabel(r"$|eB|/m_*^2$")
        ax.set_xlim(min(mu),max(mu));ax.set_ylim(min(B),max(B))
    fig.suptitle("Координаты выбранных CDW-фонов\nЛокальный поиск, глобальность координат не установлена",fontsize=12)
    fig.savefig("cdw-magnetic/figures/phase-scout-condensates.png",dpi=180)
    print(f"Wrote {destination} and phase-scout-condensates.png")
