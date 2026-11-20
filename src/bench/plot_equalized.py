#!/usr/bin/env python
"""Equalized comparison figure: accuracy vs measured spectral radius.

Every framework runs the identical recurrent and input matrices (exported
from SPIRES), an identical readout (shared ridge, bias unpenalized), and
identical input encoding and pooling. The only remaining differences are the
neuron and synapse models, so the vertical spread at any radius is exactly
the quantity of interest.

Plotted against the *measured* radius of the exported matrix rather than the
requested one, because SPIRES's own scaling uses power iteration and misses
the request on roughly two thirds of sign-balanced draws.
"""
import json, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"

def main(path, out, title_extra=""):
    rs = [json.loads(l) for l in open(path) if json.loads(l).get("status") == "ok"]
    by = collections.defaultdict(dict); grp = {}
    for r in rs:
        by[r["framework"]][round(r["rho"], 4)] = r["accuracy"]
        grp[r["framework"]] = r["group"]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    order = sorted(by, key=lambda k: -max(by[k].values()))
    ends = []
    for name in order:
        xs = sorted(by[name]); ys = [by[name][x] for x in xs]
        c = BLUE if grp[name] == "esn" else ORANGE
        ls = "-" if grp[name] == "lsm" else "--"
        ax.plot(xs, ys, ls, marker="o", color=c, linewidth=2, markersize=5,
                alpha=0.92, zorder=3)
        ends.append([name, xs[-1], ys[-1], c])

    # De-collide right-hand labels: several frameworks converge at the largest
    # radius, which would overprint their names into an unreadable stack.
    ends.sort(key=lambda e: e[2])
    lo = min(e[2] for e in ends); hi = max(e[2] for e in ends)
    gap = (hi - lo) * 0.055 if hi > lo else 0.02
    placed = []
    for name, x, y, c in ends:
        ly = y
        if placed and ly - placed[-1] < gap:
            ly = placed[-1] + gap
        placed.append(ly)
        ax.annotate(name, (x, ly), textcoords="offset points", xytext=(10, 0),
                    va="center", fontsize=9, color=INK, annotation_clip=False)
        ax.plot([x, x * 1.02], [y, ly], color=c, linewidth=0.7, alpha=0.55,
                zorder=2, clip_on=False)

    ax.axhline(1/10, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.annotate("chance (10 classes)", (ax.get_xlim()[0], 0.10),
                textcoords="offset points", xytext=(4, 4), fontsize=8, color=MUTED)

    ax.set_xlabel("Measured spectral radius of the shared weight matrix",
                  fontsize=10, color=MUTED)
    ax.set_ylabel("Test accuracy", fontsize=10, color=MUTED)
    ax.set_title("Fully equalized comparison" + title_extra,
                 fontsize=12.5, color=INK, loc="left", pad=12)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    xs_all = sorted({x for f in by for x in by[f]})
    ax.set_xlim(min(xs_all) - 0.1, max(xs_all) * 1.22)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=2, label="spiking LSM"),
                       Line2D([], [], color=BLUE, lw=2, ls="--", label="rate-coded ESN")],
              loc="lower left", frameon=False, fontsize=9)

    fig.text(0.005, 0.012,
             "Identical weights (exported from SPIRES), identical readout (shared ridge,\n"
             "bias unpenalized), identical encoding and pooling. Remaining spread is\n"
             "neuron/synapse model only.",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out, facecolor="white")
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "")
