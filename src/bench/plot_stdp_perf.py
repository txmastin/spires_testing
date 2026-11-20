#!/usr/bin/env python
"""What inhibitory STDP buys on NARMA-10: robustness, not peak accuracy.

Form: change-over-a-continuous-variable (input gain, log-spaced) for three
conditions -> line chart. NRMSE and firing rate are different quantities on
different scales, so they get two stacked panels sharing an x axis, never a
second y axis. Bands span one standard deviation over 10 seeds; the static condition's
band is the point of the figure, so it is drawn even though it is wide.

Only two hues, both from the validated categorical order used by plot_bar.py:
the two STDP conditions are the same method at different rho_0, so line style
separates them rather than a third hue. All three are direct-labelled, so
identity never rests on colour alone.
"""
import json, sys, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"

CONDS = [
    # name, colour, linestyle, label, label y-offset (multiplicative, log axis)
    ("static",        BLUE,   "-",  "static reservoir",       1.00),
    ("stdp rho=0.35", ORANGE, "-",  "iSTDP  $\\rho_0=0.35$",  1.16),
    ("stdp rho=0.50", ORANGE, "--", "iSTDP  $\\rho_0=0.50$",  0.86),
]


def load(path):
    rs = [json.loads(l.split("@@STDP@@ ", 1)[1]) for l in open(path) if "@@STDP@@" in l]
    key = lambda r: "static" if not r["stdp"] else f"stdp rho={r['rho_0']:.2f}"
    by = collections.defaultdict(list)
    for r in rs:
        by[(r["gain"], key(r))].append(r)
    return by, sorted({r["gain"] for r in rs})


def style(ax):
    ax.set_xscale("log")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def main(src, out):
    by, gains = load(src)
    g = np.array(gains)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1]})

    for name, col, ls, lab, loff in CONDS:
        vals = [[x["nrmse"] for x in by[(q, name)]] for q in gains]
        med = np.array([np.mean(v) for v in vals])
        sd  = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals])
        a1.fill_between(g, np.maximum(med - sd, 1e-12), med + sd,
                        color=col, alpha=0.13, lw=0)
        a1.plot(g, med, color=col, ls=ls, lw=2.0, zorder=3, solid_capstyle="round")
        a1.text(g[-1] * 1.15, med[-1] * loff, lab, color=col, fontsize=8.5,
                va="center", ha="left")

        rv = [[x["rate"] for x in by[(q, name)]] for q in gains]
        rate = np.array([np.mean(v) for v in rv])
        rsd  = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in rv])
        a2.fill_between(g, rate - rsd, rate + rsd, color=col, alpha=0.13, lw=0)
        a2.plot(g, rate, color=col, ls=ls, lw=2.0, zorder=3, solid_capstyle="round")

    a1.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    # Parked in the empty upper-left rather than beside the line it describes,
    # which sits in the busiest part of the panel.
    a1.text(g[0], 6.0, "grey dashes: NRMSE = 1,\nno better than predicting the mean",
            color=MUTED, fontsize=7.5, va="center", ha="left")
    a1.set_yscale("log")
    a1.set_yticks([0.8, 1, 2, 4, 8])
    a1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_ylabel("NRMSE  (held-out tail)", color=INK, fontsize=9)
    a1.set_title("Inhibitory STDP trades peak accuracy for robustness",
                 color=INK, fontsize=10, loc="left", pad=6)
    style(a1)

    a2.set_ylabel("firing rate\n(spikes / neuron / step)", color=INK, fontsize=9)
    a2.set_xlabel("input gain", color=INK, fontsize=9)
    a2.set_title("...because it holds the firing rate fixed as drive grows",
                 color=INK, fontsize=10, loc="left", pad=6)
    style(a2)

    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/stdp_curve.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "figures/stdp_perf.pdf")
