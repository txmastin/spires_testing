#!/usr/bin/env python
"""Generate the shared benchmark series.

Two tasks, each written once and exported in three formats so every
framework trains on byte-identical data:
  <task>.npy  -> numpy  (reservoirpy, torch-based, brian2, nest, ...)
  <task>.csv  -> Julia  (ReservoirComputing.jl)
  <task>.bin  -> raw f64 (u,y) pairs (SPIRES, via fread)

narma10       -- NARMA-10 system identification. Input is uniform noise;
                 the target depends on a 10-step history, so the reservoir
                 must supply memory. Rate-coded ESNs do well here.
mackey_glass  -- Mackey-Glass (tau=17) one-step-ahead prediction. A chaotic
                 delay system; input is the series itself, target is the
                 next sample. Smooth and autocorrelated, unlike NARMA's
                 white-noise drive.
"""
import numpy as np, pathlib, json

SEED = 20260220
T_TOTAL = 4000
WASHOUT = 200

def narma10(T, seed=SEED):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=T + 10)
    y = np.zeros(T + 10)
    for t in range(9, T + 9):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t]
                    + 0.1)
    return u[10:T + 10].copy(), y[10:T + 10].copy()

def mackey_glass(T, tau=17.0, beta=0.2, gamma=0.1, n=10.0,
                 dt=0.1, sample_every=10, discard=1000, seed=SEED):
    """RK4 integration of dx/dt = beta*x(t-tau)/(1+x(t-tau)^n) - gamma*x(t).

    Integrated at dt=0.1 and sampled every 10 steps, giving unit-time
    samples. `discard` samples of transient are dropped so the series sits
    on the chaotic attractor rather than the initial approach to it.
    """
    rng = np.random.default_rng(seed)
    lag = int(round(tau / dt))
    steps = (T + discard) * sample_every + lag + 10
    x = np.empty(steps)
    x[:lag + 1] = 1.2 + 0.02 * rng.standard_normal(lag + 1)

    def f(xt, xd):
        return beta * xd / (1.0 + xd ** n) - gamma * xt

    for i in range(lag, steps - 1):
        xt = x[i]
        xd = x[i - lag]                 # delayed value, held across the step
        k1 = f(xt, xd)
        k2 = f(xt + 0.5 * dt * k1, xd)
        k3 = f(xt + 0.5 * dt * k2, xd)
        k4 = f(xt + dt * k3, xd)
        x[i + 1] = xt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    s = x[lag::sample_every][discard:discard + T + 1]
    return s[:T].copy(), s[1:T + 1].copy()      # input x(t), target x(t+1)

TASKS = {"narma10": narma10, "mackey_glass": mackey_glass}

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    out = root / "data"; out.mkdir(exist_ok=True)

    for name, fn in TASKS.items():
        u, y = fn(T_TOTAL)
        assert np.all(np.isfinite(y)), f"{name} diverged"
        np.save(out / f"{name}.npy", np.stack([u, y]))
        np.savetxt(out / f"{name}.csv", np.stack([u, y], axis=1),
                   delimiter=",", header="u,y", comments="")
        np.stack([u, y], axis=1).astype(np.float64).tofile(out / f"{name}.bin")
        meta = {"seed": SEED, "T_total": T_TOTAL, "washout": WASHOUT,
                "task": name,
                "u_mean": float(u.mean()), "u_std": float(u.std()),
                "y_mean": float(y.mean()), "y_std": float(y.std())}
        (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{name:14s} u[{u.min():.3f},{u.max():.3f}] "
              f"y mean={y.mean():.4f} std={y.std():.4f}")

if __name__ == "__main__":
    main()
