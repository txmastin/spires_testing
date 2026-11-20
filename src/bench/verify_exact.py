#!/usr/bin/env python
"""Fractional neurons against the exact solution (revision, R1 comment 1).

collect_verification.py shows that SPIRES's Grunwald-Letnikov neuron matches
an independent implementation of the *same scheme*. That establishes the
implementation, not the scheme. This script checks the scheme itself: a single
SPIRES neuron (GL and Caputo, compiled from the library's own sources) is
driven with constant input, spiking disabled, and compared with the exact
solution of the linear subthreshold equation, which is a Mittag-Leffler
function:

  GL:     D^a (V - V_rest) = -(V - V_rest)/tau + I,  V(0) = V_rest
          V(t) = V_rest + I tau (1 - E_a(-t^a / tau))
  Caputo: C D^a V = -g (V - V_l) + I,                 V(0) = V_0
          V(t) = V_inf + (V_0 - V_inf) E_a(-(g/C) t^a),  V_inf = V_l + I/g

Each case runs at four step sizes. Convergence is judged on the error at
fixed times (t = 1, 5, 10), which should halve as dt halves. The maximum over
the trajectory is reported separately: it sits at the first step, where the
exact solution rises like t^alpha with an unbounded slope that an explicit
step cannot follow, so for small alpha it stays roughly constant even as the
interval it affects shrinks with dt.
The memory window spans the whole run, so this measures discretization error
only; truncating the window is a separate, deliberate approximation (see
mastin2026). Accuracy only, so machine load does not matter.

  python src/bench/verify_exact.py        # -> results/verify_exact.json
"""
import json, os, pathlib, subprocess, numpy as np, mpmath as mp

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "src" / "bench"
SPIRES = pathlib.Path(os.environ.get(
    "SPIRES_DIR", pathlib.Path.home() / "programming/projects/spiking-reservoir/spires"))
BIN = ROOT / "results" / ".verify_exact_harness"

mp.mp.dps = 50

def build():
    nsrc = SPIRES / "src" / "neurons"
    subprocess.run(["clang", "-O2", "-o", str(BIN), str(BENCH / "verify_exact_harness.c"),
                    str(nsrc / "flif_gl.c"), str(nsrc / "flif_caputo.c"),
                    f"-I{nsrc}", "-lm"], check=True)

def mittag_leffler(a, z):
    z = mp.mpf(z); s = mp.mpf(0)
    for k in range(400):
        t = z**k / mp.gamma(a*k + 1); s += t
        if k > 10 and abs(t) < mp.mpf(10)**-30:
            break
    return float(s)

def exact(model, a, t, vr, v0, I):
    if model == 0:
        return vr + I*2.0*(1 - mittag_leffler(a, -t**a / 2.0))
    vinf = vr + I/0.5
    return vinf + (v0 - vinf)*mittag_leffler(a, -0.5*t**a)

CASES = [  # name, model, alpha, V_rest (GL) or V_l (Caputo), V_0, input
    ("GL alpha=0.7, V_rest=0, I=1",         0, 0.7,  0.0,  0.0, 1.0),
    ("GL alpha=0.7, V_rest=-0.7, I=1",      0, 0.7, -0.7, -0.7, 1.0),
    ("GL alpha=0.7, V_rest=-0.7, at rest",  0, 0.7, -0.7, -0.7, 0.0),
    ("GL alpha=1, V_rest=-0.7, I=1",        0, 1.0, -0.7, -0.7, 1.0),
    ("Caputo alpha=0.7, V_l=0, I=1",        1, 0.7,  0.0,  0.0, 1.0),
    ("Caputo alpha=0.7, V_l=-0.7, relax",   1, 0.7, -0.7,  0.0, 0.0),
    ("Caputo alpha=0.4, V_l=0, I=1",        1, 0.4,  0.0,  0.0, 1.0),
    ("Caputo alpha=1, V_l=0, I=1",          1, 1.0,  0.0,  0.0, 1.0),
]
T = 10.0
DTS = (0.1, 0.05, 0.025, 0.0125)
T_FIXED = (1.0, 5.0, 10.0)

def main():
    build()
    rows = []
    for name, m, a, vr, v0, I in CASES:
        fixed, maxe = [], []
        for dt in DTS:
            out = subprocess.run([str(BIN), str(m), str(a), str(dt), str(T), str(vr), str(v0), str(I)],
                                 capture_output=True, text=True, check=True).stdout.split()
            d = np.array(out, float).reshape(-1, 2)          # row k is t = (k+1) dt
            fixed.append(max(abs(d[int(round(t / dt)) - 1, 1] - exact(m, a, t, vr, v0, I))
                             for t in T_FIXED))
            idx = slice(None, None, max(1, len(d) // 400))
            ex = np.array([exact(m, a, t, vr, v0, I) for t in d[idx, 0]])
            maxe.append(float(np.max(np.abs(d[idx, 1] - ex))))
        ok = all(e2 <= e1 * 0.6 or e2 < 1e-12 for e1, e2 in zip(fixed, fixed[1:]))
        rows.append(dict(case=name, dt=list(DTS), t_fixed=list(T_FIXED),
                         err_at_fixed_times=fixed, max_err_over_trajectory=maxe, converging=ok))
        print(f"{name:38s} |err| at t=1,5,10: " + "  ".join(f"{e:.2e}" for e in fixed)
              + "   max over run: " + "  ".join(f"{e:.2e}" for e in maxe)
              + ("" if ok else "   NOT CONVERGING"))
    print(f"dt = {DTS}")
    dst = ROOT / "results" / "verify_exact.json"
    dst.write_text(json.dumps(dict(T=T, cases=rows), indent=2))
    print(f"wrote {dst}")

if __name__ == "__main__":
    main()
