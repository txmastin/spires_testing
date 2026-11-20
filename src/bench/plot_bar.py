#!/usr/bin/env python
"""Single-size bar chart: training wall time per framework at fixed N.

Form: magnitude comparison across ~11 named categories -> horizontal bars,
sorted by value, direct-labelled. Colour encodes reservoir class (rate-coded
ESN vs spiking LSM), which is the one distinction that changes how a number
should be read; it never encodes rank. Whiskers span min-max across repeats.
Palette slots 1-2 of the validated categorical order.
"""
import json, sys, pathlib, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"

TASK_LABEL = {"narma10": "NARMA-10", "mackey_glass": "Mackey-Glass (tau=17)"}

def main(path, n, out, task="narma10"):
    rs = [json.loads(l) for l in open(path)]
    rs = [r for r in rs if r["status"] == "ok" and r["n"] == n
          and r.get("task", "narma10") == task]
    if not rs: raise SystemExit(f"no ok records for n={n} task={task}")
    by = collections.defaultdict(list)
    for r in rs: by[r["framework"]].append(r)

    rows = []
    for k, v in by.items():
        tot = [x["t_harvest"] + x["t_fit"] for x in v]
        rows.append(dict(name=k, med=min(tot), lo=min(tot), hi=max(tot),
                         group=v[0]["group"], reps=len(v),
                         nrmse=st.median(x["nrmse"] for x in v),
                         harvest=min(x["t_harvest"] for x in v)))
    rows.sort(key=lambda r: r["med"])

    names = [r["name"] for r in rows]
    med = np.array([r["med"] for r in rows])
    lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    cols = [BLUE if r["group"] == "esn" else ORANGE for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ypos = np.arange(len(rows))

    ax.barh(ypos, med, height=0.66, color=cols, zorder=3)
    ax.errorbar(med, ypos, xerr=[med - lo, hi - med], fmt="none",
                ecolor=INK, elinewidth=1.2, capsize=3, zorder=4, alpha=0.65)

    pad = med.max() * 0.012
    for i, r in enumerate(rows):
        ax.text(r["hi"] + pad, i, f"{r['med']:.3f}s", va="center", ha="left",
                fontsize=9.5, color=INK, zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Training wall time  (harvest + readout fit, seconds)",
                  fontsize=10, color=MUTED)
    ax.set_xlim(0, med.max() * 1.18)
    ax.set_title(f"Reservoir training time at N = {n}  |  "
                 f"{TASK_LABEL.get(task, task)} (T = 4000)",
                 fontsize=12.5, color=INK, pad=12, loc="left")

    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[Patch(facecolor=BLUE, label="rate-coded ESN"),
                       Patch(facecolor=ORANGE, label="spiking LSM")],
              loc="center right", frameon=False, fontsize=9.5,
              bbox_to_anchor=(1.0, 0.62))

    fig.text(0.01, 0.015,
             f"median of {rows[0]['reps']} repeats; whiskers min-max (single-outlier sensitive).  "
             "Intel i7-10510U, 4 cores, CPU only.  "
             "Frameworks run idiomatically (dt not matched across libraries).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, facecolor="white")
    print("wrote", out)

    print(f"\n== {TASK_LABEL.get(task, task)}, N={n} ==")
    print(f"{'framework':26s} {'total':>8s} {'harvest':>8s} {'nrmse':>8s} {'class':>6s}")
    for r in rows:
        print(f"{r['name']:26s} {r['med']:8.3f} {r['harvest']:8.3f} "
              f"{r['nrmse']:8.4f} {'ESN' if r['group']=='esn' else 'LSM':>6s}")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else "narma10")
