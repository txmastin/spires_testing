#!/usr/bin/env python
"""Scaling figure: training time vs reservoir size N, per framework.

Log-log, because the interesting content is the slope -- how each framework's
cost grows with N -- and the frameworks span two orders of magnitude at any
fixed N. One panel per task. Lines are direct-labelled at their right end so
identity never depends on colour alone; colour still encodes reservoir class.
"""
import json, sys, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"
TASK_LABEL = {"narma10": "NARMA-10", "mackey_glass": "Mackey-Glass (tau=17)"}

def main(path, out, tasks):
    rs = [json.loads(l) for l in open(path)]
    rs = [r for r in rs if r["status"] == "ok"]
    tasks = [t for t in tasks if any(r.get("task", "narma10") == t for r in rs)]

    fig, axes = plt.subplots(1, len(tasks), figsize=(7.2 * len(tasks), 5.4),
                             squeeze=False)
    fig.patch.set_facecolor("white")

    for ax, task in zip(axes[0], tasks):
        sub = [r for r in rs if r.get("task", "narma10") == task]
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        grp = {}
        for r in sub:
            by[r["framework"]][r["n"]].append(r["t_harvest"] + r["t_fit"])
            grp[r["framework"]] = r["group"]

        # Mean with a standard-deviation band over repeats. The band is the
        # point of the figure as much as the line: it shows which frameworks
        # deliver a predictable training time and which do not.
        order = sorted(by, key=lambda k: np.mean(by[k][max(by[k])]))
        ends = []
        for name in order:
            ns = sorted(by[name])
            ys = np.array([np.mean(by[name][n]) for n in ns])
            sd = np.array([np.std(by[name][n], ddof=1) if len(by[name][n]) > 1 else 0.0
                           for n in ns])
            c = BLUE if grp[name] == "esn" else ORANGE
            ax.fill_between(ns, np.maximum(ys - sd, 1e-12), ys + sd,
                            color=c, alpha=0.16, lw=0, zorder=2)
            ax.plot(ns, ys, "-o", color=c, linewidth=2, markersize=5,
                    alpha=0.9, zorder=3)
            ends.append([name, ns[-1], ys[-1], c])

        # De-collide the right-hand direct labels. Several frameworks land
        # within a few percent of each other at the largest N, which overprints
        # their names into an unreadable stack. Push labels apart in log space
        # to a minimum separation, then draw a leader line back to the true
        # endpoint so the label still reads as belonging to its series.
        ends.sort(key=lambda e: e[2])
        span = np.log10(max(e[2] for e in ends)) - np.log10(min(e[2] for e in ends))
        gap = max(span, 1e-9) * 0.055
        placed = []
        for name, x, y, c in ends:
            ly = np.log10(y)
            if placed and ly - placed[-1] < gap:
                ly = placed[-1] + gap
            placed.append(ly)
            ax.annotate(name, (x, 10 ** ly), textcoords="offset points",
                        xytext=(9, 0), va="center", fontsize=8.5, color=INK,
                        annotation_clip=False)
            ax.plot([x, x * 1.045], [y, 10 ** ly], color=c, linewidth=0.7,
                    alpha=0.55, zorder=2, clip_on=False)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Reservoir size  N", fontsize=10, color=MUTED)
        ax.set_ylabel("Training wall time  (s)", fontsize=10, color=MUTED)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=12, color=INK,
                     loc="left", pad=10)
        ax.grid(True, which="both", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        xmax = max(max(by[k]) for k in by)
        ax.set_xlim(right=xmax * 2.6)

    from matplotlib.patches import Patch
    axes[0][0].legend(handles=[Patch(facecolor=BLUE, label="rate-coded ESN"),
                               Patch(facecolor=ORANGE, label="spiking LSM")],
                      loc="upper left", frameon=False, fontsize=9)
    fig.text(0.005, 0.015, "mean ± s.d. across repeats.  Intel i7-10510U, 4 cores, CPU only.  "
             "Frameworks run idiomatically (dt not matched); hyperparameters tuned per framework and task.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor="white")
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep_pinned.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "figures/scaling.pdf",
         (sys.argv[3].split(",") if len(sys.argv) > 3 else ["narma10", "mackey_glass"]))
