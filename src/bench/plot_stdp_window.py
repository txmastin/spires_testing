#!/usr/bin/env python
"""STDP windows measured from SPIRES by two-neuron pairing.

Form: change-over-a-continuous-variable for two series that must not share a
scale (excitatory dw and inhibitory d|w| differ in sign convention and
magnitude) -> two stacked panels sharing one x axis, never a dual y axis.
Zero lines and dt=0 are marked because both windows are read relative to them.
Palette slots 1-2 of the validated categorical order, matching plot_bar.py.
"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dededa"


def panel(ax, dt, y, colour, title, note):
    ax.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax.axvline(0.0, color=GRID, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.plot(dt, y, color=colour, lw=2.0, zorder=3, solid_capstyle="round")
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
    ax.text(0.985, 0.06, note, transform=ax.transAxes, ha="right", va="bottom",
            color=MUTED, fontsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def main(src, out):
    d = np.genfromtxt(src, delimiter=",", names=True)
    dt, exc, inh = d["dt"], d["dw_exc"], d["dmag_inh"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True)

    panel(a1, dt, exc, BLUE,
          "Excitatory synapse: asymmetric, causal",
          r"$\tau_+=17$, $\tau_-=34$ ms")
    a1.set_ylabel("$\\Delta w$", color=INK, fontsize=9)
    a1.annotate("pre before post\npotentiates", xy=(14, exc[dt == 14][0]),
                xytext=(28, max(exc) * 0.62), color=MUTED, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1.0))
    a1.annotate("post before pre\ndepresses", xy=(-22, exc[dt == -22][0]),
                xytext=(-58, max(exc) * 0.52), color=MUTED, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1.0))

    panel(a2, dt, inh, ORANGE,
          "Inhibitory synapse: symmetric, homeostatic",
          r"$\tau_i=20$ ms, $\alpha=2\tau_i\rho_0=0.8$")
    a2.set_ylabel("$\\Delta |w|$", color=INK, fontsize=9)
    a2.set_xlabel("$\\Delta t = t_{post} - t_{pre}$   (ms)", color=INK, fontsize=9)
    a2.annotate("either order strengthens\ninhibition near coincidence",
                xy=(9, inh[dt == 9][0]), xytext=(24, min(inh) * 0.20),
                color=MUTED, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1.0))
    # The one-sample notch at dt=0 is a property of the update order, not the
    # rule; call it out rather than leave a reader guessing.
    a2.annotate("$\\Delta t=0$: coincident pair,\nno trace accumulated yet",
                xy=(0, inh[dt == 0][0]), xytext=(-58, min(inh) * 0.20),
                color=MUTED, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1.0))

    fig.suptitle("STDP windows measured from SPIRES", color=INK, fontsize=11,
                 x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/stdp_window.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/stdp_window.pdf"
    main(src, out)
