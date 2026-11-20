#!/usr/bin/env python
"""Why do frameworks running 'the same LIF reservoir' disagree?

Accuracy differences between frameworks persisted even after equalising
weights, readout and encoding, so the cause must lie in the neuron/synapse
semantics themselves. Reading each implementation turned up four concrete
divergences:

  reset       SPIRES resets the membrane to V_0; snnTorch and rockpool
              subtract the threshold, keeping the supra-threshold residue.
  synapse     norse and rockpool filter spikes through an exponential
              synaptic current; SPIRES (simple) and snnTorch inject spikes
              directly.
  normalisation  SPIRES's PSC filter is normalised to DC gain 1
              (trace = trace*decay + (1-decay)*spike), while rockpool and
              norse are unnormalised (DC gain ~ tau_syn/dt), so identical
              tau_syn means ~5x different drive at tau_syn=5.
  multi-spike rockpool emits floor(v/threshold) spikes per step; every other
              framework emits at most one.

Rather than infer their effects from framework accuracy -- which confounds
them with each other and with tuning -- this runs a factorial ablation in
plain numpy, identical weights and input throughout, toggling one property
at a time. The winner tells us which semantics actually matter.
"""
import argparse, itertools, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

def simulate(X, W, Win, *, leak, vth, syn, tau_syn, normalized,
             reset, multispike):
    """One reservoir pass over (T, C) input -> (T, N) membrane trace."""
    n = W.shape[0]
    T = X.shape[0]
    decay = np.exp(-1.0 / tau_syn) if syn else 0.0
    v = np.zeros(n); s = np.zeros(n); tr = np.zeros(n)
    out = np.empty((T, n))
    for t in range(T):
        drive = Win @ X[t]
        if syn:
            inj = (1.0 - decay) * s if normalized else s
            tr = tr * decay + inj
            rec = W @ tr
        else:
            rec = W @ s
        v = (1.0 - leak) * v + rec + drive
        if multispike:
            s = np.where(v >= vth, np.floor(v / vth), 0.0)
        else:
            s = (v > vth).astype(np.float64)
        if reset == "zero":
            v = np.where(s > 0, 0.0, v)
        else:                       # subtractive
            v = v - s * vth
        out[t] = v
    return out

def features(X, **kw):
    return np.stack([common.pool_states(simulate(x.astype(np.float64), **kw))
                     for x in X])

def score(Ftr, ytr, Fte, yte, nc):
    return max(common.accuracy(common.classify(Fte,
               common.ridge_classify(Ftr, ytr, nc, reg=r)), yte)
               for r in (1e-2, 1e0, 1e2, 1e4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--task", default="fsdd")
    ap.add_argument("--limit-train", type=int, default=900)
    ap.add_argument("--limit-test", type=int, default=350)
    ap.add_argument("--gains", default="0.2,1.0,4.0")
    ap.add_argument("--wscales", default="0.18,0.5,1.0,2.0",
                    help="recurrent weight scale. This is the axis the "
                         "normalisation difference acts on: an unnormalised "
                         "PSC multiplies the recurrent drive by ~1/(1-decay) "
                         "(5.5x at tau_syn=5), which input gain cannot offset. "
                         "Each configuration is reported at its own best.")
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--tau-syn", type=float, default=5.0)
    ap.add_argument("--out", default="results/model_ablation.json")
    a = ap.parse_args()

    Xtr, ytr, Xte, yte, meta = common.load_classification(a.task)
    Xtr, ytr = Xtr[:a.limit_train], ytr[:a.limit_train]
    Xte, yte = Xte[:a.limit_test], yte[:a.limit_test]
    C, nc = meta["C"], meta["n_classes"]

    common.set_weight_source("spires", conn=0.1, ei=0.5)
    W, _ = common.make_reservoir(a.n, seed=3, spectral_radius=1.2)
    rho = float(np.abs(np.linalg.eigvals(W)).max())
    gains = [float(g) for g in a.gains.split(",")]
    wscales = [float(w) for w in a.wscales.split(",")]

    rows = []
    combos = itertools.product(
        [("none", False, False), ("exp-norm", True, True), ("exp-raw", True, False)],
        ["zero", "subtract"],
        [False, True],
    )
    for (syn_name, syn, normd), reset, multi in combos:
        best_acc, best_cfg = -1.0, None
        for g in gains:
            Win = common.make_input_weights(a.n, C, seed=3, gain=g)
            for ws in wscales:
                kw = dict(W=W * ws, Win=Win, leak=a.leak, vth=1.0, syn=syn,
                          tau_syn=a.tau_syn, normalized=normd, reset=reset,
                          multispike=multi)
                Ftr = features(Xtr, **kw); Fte = features(Xte, **kw)
                acc = score(Ftr, ytr, Fte, yte, nc)
                if acc > best_acc:
                    best_acc, best_cfg = acc, (g, ws)
        rows.append(dict(synapse=syn_name, reset=reset, multispike=multi,
                         accuracy=best_acc, gain=best_cfg[0],
                         wscale=best_cfg[1],
                         rho_eff=rho * best_cfg[1]))
        print(f"syn={syn_name:9s} reset={reset:9s} multi={str(multi):5s} "
              f"acc={best_acc:.4f} (gain {best_cfg[0]}, wscale {best_cfg[1]}, "
              f"rho_eff {rho*best_cfg[1]:.2f})", flush=True)

    p = pathlib.Path(common.ROOT) / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rho": rho, "n": a.n, "task": a.task,
                             "rows": rows}, indent=2))
    best = max(rows, key=lambda r: r["accuracy"])
    print(f"\nbest: {best}")
    print(f"spread: {min(r['accuracy'] for r in rows):.4f} - "
          f"{max(r['accuracy'] for r in rows):.4f}")

if __name__ == "__main__":
    main()
