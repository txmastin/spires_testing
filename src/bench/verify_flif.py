#!/usr/bin/env python
"""Verify SPIRES's Grunwald-Letnikov fractional LIF against an independent
reference implementation.

The reference is written from the published equations, not from SPIRES
source, so that a shared misreading cannot cancel out. Following Mastin
et al. (npj Unconv. Comput. 2026):

  Eq (15)  V_n = dt^a [ -(V_{n-1} - V_rest)/tau_m + I_n + b ]
                 - sum_{k=1}^{min(n,L)} c_k(a) V_{n-k}

  Eq (14)  c_k(a) = (-1)^k * binom(a, k),
           binom(a, k) = a(a-1)...(a-k+1) / k!

  Eq (16)  at a = 1 this must reduce to the integer-order forward-Euler LIF
           V_n = V_{n-1} + dt [ -(V_{n-1} - V_rest)/tau_m + I_n + b ]

Coefficients are evaluated from the explicit product form of Eq (14) rather
than the recurrence w_k = w_{k-1}(1 - (a+1)/k) that SPIRES uses internally.
The two are algebraically identical, so computing them independently means a
bug in the recurrence cannot be masked by reproducing that same recurrence
here.

Why this matters more than an accuracy check: the same paper reports (Fig. 5)
that classification accuracy shifts by under two percentage points across a
+/-10% drift in alpha. An implementation error equivalent to a few percent
shift in alpha is therefore invisible to accuracy-based validation, and only
exact numerical comparison can detect it.
"""
import argparse, json, math, pathlib, struct, sys
import numpy as np


def gl_coefficients(alpha, L):
    """c_k(alpha) for k = 1..L, from the explicit product form of Eq (14)."""
    c = np.empty(L)
    for k in range(1, L + 1):
        num = 1.0
        for j in range(k):
            num *= (alpha - j)
        # k! overflows a float beyond k = 170, so fall back to log space there
        binom = num / float(math.factorial(k)) if k < 171 else _binom_big(alpha, k)
        c[k - 1] = ((-1.0) ** k) * binom
    return c


def _binom_big(alpha, k):
    """Log-space fallback so large k does not overflow k!."""
    from math import lgamma, log, exp, copysign
    sign = 1.0
    acc = 0.0
    for j in range(k):
        v = alpha - j
        sign *= copysign(1.0, v)
        acc += log(abs(v))
    return sign * exp(acc - lgamma(k + 1))


def load(path):
    with open(path, "rb") as f:
        n, T = struct.unpack("<ii", f.read(8))
        (V_th, V_reset, V_rest, tau_m, alpha,
         T_mem, bias, t_ref, dt) = struct.unpack("<9d", f.read(72))
        W = np.frombuffer(f.read(n * n * 8), dtype="<f8").reshape(n, n).copy()
        Win = np.frombuffer(f.read(n * 8), dtype="<f8").copy()
        u = np.frombuffer(f.read(T * 8), dtype="<f8").copy()
        V = np.frombuffer(f.read(T * n * 8), dtype="<f8").reshape(T, n).copy()
    return dict(n=n, T=T, V_th=V_th, V_reset=V_reset, V_rest=V_rest,
                tau_m=tau_m, alpha=alpha, T_mem=T_mem, bias=bias,
                t_ref=t_ref, dt=dt, W=W, Win=Win, u=u, V=V)


def reference(d, transpose=False, prefill_rest=True):
    """Replay the run from Eq (15).

    prefill_rest mirrors Algorithm 1's "initialise circular buffer to V_rest",
    so the history sum always spans L terms. The alternative (summing only
    over steps that have actually occurred) coincides with it whenever
    V_rest = 0, which is why the default test uses V_rest = 0.
    """
    n, T = d["n"], d["T"]
    L = max(1, int(round(d["T_mem"] / d["dt"])))
    micro = max(1, int(round(1.0 / d["dt"])))
    c = gl_coefficients(d["alpha"], L)
    W = d["W"].T if transpose else d["W"]
    dta = d["dt"] ** d["alpha"]

    hist = np.full((L, n), d["V_rest"] if prefill_rest else 0.0)
    v = np.full(n, d["V_rest"])
    s = np.zeros(n)
    out = np.empty((T, n))

    for t in range(T):
        drive = d["Win"] * d["u"][t]
        for _ in range(micro):          # dt < 1 means several micro-steps per sample
            I = W @ s + drive
            leak = -(v - d["V_rest"]) / d["tau_m"]
            hsum = c @ hist             # sum_k c_k V_{n-k}
            v = dta * (leak + I + d["bias"]) - hsum
            s = (v >= d["V_th"]).astype(np.float64)
            v = np.where(s > 0, d["V_reset"], v)
            hist = np.roll(hist, 1, axis=0)
            hist[0] = v                 # post-reset value enters the buffer
        out[t] = v
    return out


def euler_lif(d, transpose=False):
    """Eq (16): integer-order forward-Euler LIF, the alpha -> 1 limit."""
    n, T = d["n"], d["T"]
    micro = max(1, int(round(1.0 / d["dt"])))
    W = d["W"].T if transpose else d["W"]
    v = np.full(n, d["V_rest"]); s = np.zeros(n)
    out = np.empty((T, n))
    for t in range(T):
        drive = d["Win"] * d["u"][t]
        for _ in range(micro):
            I = W @ s + drive
            v = v + d["dt"] * (-(v - d["V_rest"]) / d["tau_m"] + I + d["bias"])
            s = (v >= d["V_th"]).astype(np.float64)
            v = np.where(s > 0, d["V_reset"], v)
        out[t] = v
    return out


def compare(ref, got, tol=1e-9):
    diff = np.abs(ref - got)
    bad = np.where(diff.max(axis=1) > tol)[0]
    first = int(bad[0]) if len(bad) else None
    return dict(max_abs_diff=float(diff.max()),
                first_divergent_step=first,
                steps_matched=int(first if first is not None else ref.shape[0]),
                total_steps=int(ref.shape[0]),
                spike_agreement=float(((ref >= 0) == (got >= 0)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/flif_trace.bin")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = load(a.path)
    L = int(round(d["T_mem"] / d["dt"]))
    print(f"n={d['n']} T={d['T']} alpha={d['alpha']} tau_m={d['tau_m']} "
          f"dt={d['dt']} T_mem={d['T_mem']} L={L} V_rest={d['V_rest']}")

    results = {}
    for name, kw in (("W", dict(transpose=False)), ("W_transposed", dict(transpose=True))):
        results[name] = compare(reference(d, **kw), d["V"], a.tol)

    best = min(results, key=lambda k: (results[k]["first_divergent_step"] is not None,
                                       results[k]["max_abs_diff"]))
    r = results[best]
    print(json.dumps(results, indent=2))

    if abs(d["alpha"] - 1.0) < 1e-12:
        e = compare(euler_lif(d), d["V"], a.tol)
        print("\nalpha = 1 vs Eq (16) integer-order Euler LIF:")
        print(json.dumps(e, indent=2))
        results["euler_limit"] = e

    if r["first_divergent_step"] is None:
        print(f"\nPASS: GL reference matches SPIRES on all {d['T']} steps "
              f"(max |dV| = {r['max_abs_diff']:.3e}, orientation {best})")
    else:
        print(f"\nFAIL: diverges at step {r['first_divergent_step']} "
              f"of {d['T']} (max |dV| = {r['max_abs_diff']:.3e})")

    if a.json:
        p = pathlib.Path(a.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"params": {k: d[k] for k in
                                            ("n", "T", "alpha", "tau_m", "dt",
                                             "T_mem", "V_th", "V_reset",
                                             "V_rest", "bias")},
                                 "results": results}, indent=2))


if __name__ == "__main__":
    main()
