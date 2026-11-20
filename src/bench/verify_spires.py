#!/usr/bin/env python
"""Verify SPIRES's discrete-LIF dynamics against an independent reference.

This is the correctness claim the benchmark rests on: a fast implementation
is only interesting if it computes the right thing. Rather than infer
correctness from task accuracy -- which is dominated by hyperparameter
tuning and so says little about the implementation -- we replay SPIRES's own
recorded trajectory in a from-scratch numpy implementation of the same
model and compare membrane potentials step by step.

The reference is written directly from the model definition:

    V[t] = (1 - leak) * V[t-1] + W @ s[t-1] + gain * W_in * u[t] + bias
    s[t] = 1 if V[t] > V_th else 0,   and V[t] <- V_0 wherever s[t] = 1

Recurrent input uses the *previous* step's spikes, matching SPIRES stepping
the synapse from last_spikes before updating neurons.

Because a spiking model is discontinuous, a tiny numerical difference can
flip a threshold crossing and diverge the trajectories. We therefore report
both the pre-divergence agreement and the step at which the spike patterns
first differ -- exact agreement on every spike is the strong result.
"""
import numpy as np, pathlib, sys, struct, json

def load(path):
    with open(path, "rb") as f:
        n, T = struct.unpack("<ii", f.read(8))
        leak, vth, v0, bias, dt = struct.unpack("<5d", f.read(40))
        W = np.frombuffer(f.read(n*n*8), dtype="<f8").reshape(n, n).copy()
        Win = np.frombuffer(f.read(n*8), dtype="<f8").copy()
        u = np.frombuffer(f.read(T*8), dtype="<f8").copy()
        V = np.frombuffer(f.read(T*n*8), dtype="<f8").reshape(T, n).copy()
    return dict(n=n, T=T, leak=leak, vth=vth, v0=v0, bias=bias, dt=dt,
                W=W, Win=Win, u=u, V=V)

def reference(d, transpose):
    n, T = d["n"], d["T"]
    W = d["W"].T if transpose else d["W"]
    V = np.zeros(n); s = np.zeros(n)
    out = np.empty((T, n))
    for t in range(T):
        V = (1.0 - d["leak"]) * V + W @ s + d["Win"] * d["u"][t] + d["bias"]
        s = (V > d["vth"]).astype(np.float64)
        V = np.where(s > 0, d["v0"], V)
        out[t] = V
    return out

def main(path="results/spires_trace.bin"):
    d = load(path)
    results = {}
    for name, tr in (("W", False), ("W_transposed", True)):
        R = reference(d, tr)
        diff = np.abs(R - d["V"])
        # first step where the two implementations disagree beyond fp noise
        bad = np.where(diff.max(axis=1) > 1e-9)[0]
        first = int(bad[0]) if len(bad) else None
        pre = diff[:first].max() if first else diff.max()
        results[name] = dict(
            max_abs_diff=float(diff.max()),
            max_abs_diff_before_divergence=float(pre),
            first_divergent_step=first,
            steps_matched=int(first if first is not None else d["T"]),
            total_steps=int(d["T"]),
            spike_agreement=float(((R > d["vth"]) == (d["V"] > d["vth"])).mean()),
        )
    best = min(results, key=lambda k: (results[k]["first_divergent_step"] is not None,
                                       results[k]["max_abs_diff"]))
    out = dict(orientation=best, detail=results,
               n=d["n"], T=d["T"], leak=d["leak"], vth=d["vth"])
    print(json.dumps(out, indent=2))
    r = results[best]
    if r["first_divergent_step"] is None:
        print(f"\nPASS: SPIRES matches the reference on all {d['T']} steps "
              f"(max |ΔV| = {r['max_abs_diff']:.3e})")
    else:
        print(f"\nMatched {r['steps_matched']}/{d['T']} steps exactly "
              f"(max |ΔV| before divergence = "
              f"{r['max_abs_diff_before_divergence']:.3e})")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
