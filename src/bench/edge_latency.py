#!/usr/bin/env python
"""Per-step latency and jitter.

The main benchmark reports total harvest time, which is the right measure for
batch training. Real-time embedded control is governed instead by worst-case
per-step latency: a controller with a 1 ms budget is broken by a single 5 ms
step even if the mean is 50 us. This measures each individual reservoir step
and reports the distribution.

Only the recurrent update is timed -- one input sample in, one state vector
out -- which is the operation an embedded deployment actually performs in its
loop. SPIRES is measured by a C probe (edge_latency.c) to avoid attributing
Python interpreter overhead to it; the Python frameworks are measured in
Python because that is how they would be deployed.
"""
import argparse, json, pathlib, struct, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "bench"))
import common

def summarize(lat, name, extra=None):
    lat = np.asarray(lat) * 1e6           # microseconds
    d = dict(framework=name,
             median_us=float(np.median(lat)),
             p99_us=float(np.percentile(lat, 99)),
             max_us=float(lat.max()),
             mean_us=float(lat.mean()),
             jitter_ratio=float(lat.max() / max(np.median(lat), 1e-12)),
             n_steps=int(len(lat)))
    if extra: d.update(extra)
    return d

def spires(n, T):
    out = "/tmp/spires_latency.bin"
    subprocess.run([str(ROOT / "src/bench/edge_latency"), "--n", str(n),
                    "--T", str(T), "--out", out], cwd=ROOT, check=True,
                   capture_output=True)
    with open(out, "rb") as f:
        steps, isolation = struct.unpack("<ii", f.read(8))
        (overhead,) = struct.unpack("<d", f.read(8))
        lat = np.frombuffer(f.read(steps * 8), dtype="<f8")
    # What the probe actually obtained: the tail is only interpretable if the
    # measurement was protected from the scheduler.
    got = [name for bit, name in ((1, "affinity"), (2, "sched_fifo"), (4, "mlock"))
           if isolation & bit]
    return summarize(lat, "SPIRES", {"clock_overhead_us": overhead * 1e6,
                                     "isolation": got})

def torch_frameworks(n, T, which):
    import torch
    torch.set_grad_enabled(False)
    u, y, wash = common.load_data("narma10")
    u = u[:T]
    W, Win = common.make_reservoir(n, seed=0, gain=5.0, spectral_radius=1.2)
    Wt = torch.tensor(W, dtype=torch.float32)
    Wit = torch.tensor(Win, dtype=torch.float32)
    ut = torch.tensor(u, dtype=torch.float32)
    out = []

    if "snnTorch" in which:
        import snntorch as snn
        lif = snn.Leaky(beta=0.95, threshold=1.0, init_hidden=False)
        mem = torch.zeros(n); spk = torch.zeros(n); lat = np.empty(T)
        for _ in range(min(200, T)):        # warmup
            spk, mem = lif(Wit * ut[0] + Wt @ spk, mem)
        mem = torch.zeros(n); spk = torch.zeros(n)
        for t in range(T):
            a = time.perf_counter()
            spk, mem = lif(Wit * ut[t] + Wt @ spk, mem)
            _ = mem.numpy()
            lat[t] = time.perf_counter() - a
        out.append(summarize(lat, "snnTorch"))

    if "norse" in which:
        from norse.torch.module.lif import LIFCell
        from norse.torch.functional.lif import LIFParameters
        cell = LIFCell(p=LIFParameters(v_th=torch.tensor(1.0)))
        st = None; spk = torch.zeros(1, n); lat = np.empty(T)
        for _ in range(min(200, T)):
            spk, st = cell((Wit * ut[0] + Wt @ spk[0]).unsqueeze(0), st)
        st = None; spk = torch.zeros(1, n)
        for t in range(T):
            a = time.perf_counter()
            spk, st = cell((Wit * ut[t] + Wt @ spk[0]).unsqueeze(0), st)
            _ = st.v[0].numpy()
            lat[t] = time.perf_counter() - a
        out.append(summarize(lat, "norse"))

    if "reservoirpy" in which:
        from reservoirpy.nodes import Reservoir
        res = Reservoir(units=n, lr=0.3, sr=0.9, input_scaling=1.0,
                        rc_connectivity=0.1, seed=0)
        res.initialize(np.zeros((1, 1)))
        # single-step call takes a 1-D input vector
        x0 = np.array([u[0]], dtype=np.float64)
        for _ in range(min(200, T)):
            res(x0)
        lat = np.empty(T)
        xs = [np.array([v], dtype=np.float64) for v in u]
        for t in range(T):
            a = time.perf_counter()
            res(xs[t])
            lat[t] = time.perf_counter() - a
        out.append(summarize(lat, "reservoirpy"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--frameworks", default="SPIRES,snnTorch,norse,reservoirpy")
    ap.add_argument("--reps", type=int, default=10,
                    help="independent measurement runs; statistics are over these")
    ap.add_argument("--out", default="results/edge_latency.json")
    a = ap.parse_args()
    which = a.frameworks.split(",")

    # Repeat the whole measurement: a single pass gives the spread of steps
    # within one run, not the reproducibility of the run itself. Tail
    # statistics in particular need the latter.
    reps = []
    for i in range(a.reps):
        r = []
        if "SPIRES" in which:
            r.append(spires(a.n, a.T))
        r += torch_frameworks(a.n, a.T, which)
        reps.append({x["framework"]: x for x in r})
        print(f"  rep {i+1}/{a.reps} done", flush=True)

    rows = []
    for fw in reps[0]:
        agg = {"framework": fw, "reps": a.reps}
        for k in ("median_us", "p99_us", "max_us", "mean_us", "jitter_ratio"):
            v = np.array([rp[fw][k] for rp in reps], dtype=float)
            agg[k] = float(v.mean())
            agg[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        agg["n_steps"] = reps[0][fw]["n_steps"]
        # Keep the per-run values, not just their spread. A high standard
        # deviation on p99 can mean either one bad run or steady variation,
        # and the aggregate alone cannot tell those apart.
        agg["per_run"] = {k: [rp[fw][k] for rp in reps]
                          for k in ("median_us", "p99_us", "max_us")}
        if "isolation" in reps[0][fw]:
            agg["isolation"] = reps[0][fw]["isolation"]
        rows.append(agg)

    p = ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n": a.n, "T": a.T, "rows": rows}, indent=2))

    print(f"\nper-step latency, N={a.n}, {a.T} steps, {a.reps} runs (microseconds, mean +/- sd)")
    print(f"{'framework':13s} {'median':>16s} {'p99':>16s} {'max':>16s}")
    for r in sorted(rows, key=lambda r: r["median_us"]):
        print(f"{r['framework']:13s} "
              f"{r['median_us']:>9.1f}+-{r['median_us_sd']:<5.1f} "
              f"{r['p99_us']:>9.1f}+-{r['p99_us_sd']:<5.1f} "
              f"{r['max_us']:>9.1f}+-{r['max_us_sd']:<5.1f}")
    print(f"\nwrote {p}")

if __name__ == "__main__":
    main()
