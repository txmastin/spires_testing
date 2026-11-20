#!/usr/bin/env python
"""Deployability metrics for edge / embedded targets.

The speed comparison is CPU-only by design: SPIRES targets embedded and
neuromorphic deployment, where no GPU exists and the PyTorch-based
frameworks' usual escape hatch is unavailable. On those targets the numbers
that decide whether a framework is usable at all are not throughput but:

  footprint   -- what has to be installed on the device
  cold start  -- time from process launch to the first computation, which
                 dominates when a device wakes, infers once, and sleeps
  peak RSS    -- resident memory high-water mark during inference
  per-step    -- worst-case latency and jitter, which real-time control
                 cares about far more than an aggregate mean

Each measurement runs the framework in an isolated subprocess so its cost is
attributed cleanly.
"""
import argparse, json, os, pathlib, resource, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENV = pathlib.Path(os.environ.get("BENCH_VENV", pathlib.Path.home() / ".pyvenv"))
SITE = next(iter(sorted((VENV / "lib").glob("python3*/site-packages"))),
            VENV / "lib" / "python3.14" / "site-packages")
PY = str(VENV / "bin" / "python")
SPIRES = pathlib.Path(os.environ.get(
    "SPIRES_DIR", pathlib.Path.home() / "programming/projects/spiking-reservoir/spires"))
JULIA_DEPOT = pathlib.Path(os.environ.get("JULIA_DEPOT_PATH",
                                          str(pathlib.Path.home() / ".julia")).split(":")[0])

# import line used to bring each framework to a ready-to-compute state
IMPORTS = {
    "reservoirpy": "from reservoirpy.nodes import Reservoir, Ridge",
    "pytorch-esn": "import torch; from torchesn.nn import ESN",
    "snnTorch":    "import torch, snntorch as snn",
    "norse":       "import torch; from norse.torch.module.lif import LIFCell",
    "rockpool":    "import torch; from rockpool.nn.modules import LIFTorch",
    "brian2":      "import brian2",
    "NEST":        "import nest",
    "ANNarchy":    "import sys; sys.argv=sys.argv[:1]; import ANNarchy",
}
# directories that must ship with each framework
PKG_DIRS = {
    "reservoirpy": ["reservoirpy", "scipy", "numpy"],
    "pytorch-esn": ["torchesn", "torch", "numpy"],
    "snnTorch":    ["snntorch", "torch", "numpy"],
    "norse":       ["norse", "torch", "numpy"],
    "rockpool":    ["rockpool", "torch", "numpy"],
    "brian2":      ["brian2", "numpy", "scipy", "cython"],
    "NEST":        ["nest", "numpy", "scipy"],
    "ANNarchy":    ["ANNarchy", "numpy", "scipy", "nanobind"],
}

def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return total

def footprint():
    out = {}
    # SPIRES: the static library plus the system BLAS/LAPACK it links against
    lib = SPIRES / "lib" / "libspires.a"
    blas = 0
    for cand in ("/usr/lib64/libopenblas.so", "/usr/lib/libopenblas.so",
                 "/usr/lib64/liblapacke.so", "/usr/lib/liblapacke.so"):
        if os.path.exists(cand):
            blas += os.path.getsize(os.path.realpath(cand))
    lib_bytes = lib.stat().st_size if lib.exists() else 0
    out["SPIRES"] = {"bytes": lib_bytes + blas,
                     "lib_only_bytes": lib_bytes,
                     "detail": "libspires.a + openblas/lapacke",
                     "runtime": "none (C, static link)",
                     "note": "dominated by OpenBLAS; libspires.a alone is "
                             f"{lib_bytes/1e6:.2f} MB, and an embedded build "
                             "could link a reference or size-optimised BLAS"}
    for fw, dirs in PKG_DIRS.items():
        total = sum(dir_size(SITE / d) for d in dirs if (SITE / d).exists())
        out[fw] = {"bytes": total, "detail": " + ".join(dirs),
                   "runtime": "CPython"}
    # Julia's depot excludes the runtime itself, which must also ship.
    jl = JULIA_DEPOT / "packages"
    jl_rt = 0
    try:
        import shutil
        exe = shutil.which("julia")
        if exe:
            root_dir = pathlib.Path(os.path.realpath(exe)).parent.parent
            jl_rt = dir_size(root_dir)
    except Exception:
        pass
    if jl.exists():
        out["RC.jl"] = {"bytes": dir_size(jl) + jl_rt,
                        "detail": "julia runtime + depot packages",
                        "runtime": "Julia + JIT"}
    return out

def _run_isolated(cmd, cwd=ROOT):
    """Run a child and return (wall_seconds, peak_rss_bytes)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, env=_env(), capture_output=True)
    dt = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # ru_maxrss is a high-water mark across all children, so it only ever
    # rises; a larger child raises it, a smaller one leaves it unchanged.
    return dt, max(after - before, 0) * 1024, after * 1024

def _env():
    e = os.environ.copy()
    e["VIRTUAL_ENV"] = str(VENV)
    e["PATH"] = f"{VENV}/bin:" + e.get("PATH", "")
    e["JULIA_DEPOT_PATH"] = str(JULIA_DEPOT)
    return e

def cold_start(reps=10):
    """Wall time from process launch to framework ready to compute."""
    out = {}
    base = []
    for _ in range(reps):
        dt, _, _ = _run_isolated([PY, "-c", "pass"])
        base.append(dt)
    out["_python_baseline"] = float(np.median(base))
    for fw, imp in IMPORTS.items():
        ts = []
        for _ in range(reps):
            dt, _, _ = _run_isolated([PY, "-c", imp])
            ts.append(dt)
        out[fw] = float(np.median(ts))
    # SPIRES: a linked C binary that creates a reservoir and exits
    ts = []
    for _ in range(reps):
        dt, _, _ = _run_isolated([str(ROOT / "src/bench/spires_rho_probe"),
                                  "--n", "50", "--out", "/tmp/cs.bin"])
        ts.append(dt)
    out["SPIRES"] = float(np.median(ts))
    ts = []
    for _ in range(reps):
        dt, _, _ = _run_isolated(["julia", "-e", "using ReservoirComputing"])
        ts.append(dt)
    out["RC.jl"] = float(np.median(ts))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="results/edge_metrics.json")
    ap.add_argument("--rss-n", type=int, default=200)
    ap.add_argument("--rss-task", default="fsdd")
    ap.add_argument("--skip-rss", action="store_true")
    a = ap.parse_args()

    res = {"footprint": footprint(), "cold_start_s": cold_start(a.reps)}
    if not a.skip_rss:
        print(f"\nmeasuring peak RSS (N={a.rss_n}, {a.rss_task}) ...")
        res["peak_rss_bytes"] = measure_rss(a.rss_n, a.rss_task)
    p = ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))

    print(f"\n{'framework':14s} {'footprint':>12s}  {'cold start':>11s}  runtime")
    fp, cs = res["footprint"], res["cold_start_s"]
    for fw in sorted(fp, key=lambda k: fp[k]["bytes"]):
        mb = fp[fw]["bytes"] / 1e6
        s = cs.get(fw)
        print(f"{fw:14s} {mb:>10.1f} MB  {('%.3f s' % s) if s else '—':>11s}  "
              f"{fp[fw]['runtime']}")
    rss = res.get("peak_rss_bytes", {})
    if rss:
        print(f"\n{'framework':14s} {'peak RSS':>11s}")
        for fw in sorted(rss, key=lambda k: rss[k]):
            print(f"{fw:14s} {rss[fw]/1e6:>9.1f} MB")
    print(f"\n(bare python interpreter start: {cs['_python_baseline']:.3f} s)")
    print(f"wrote {p}")



# ---------------------------------------------------------------------------
# Peak resident memory.
#
# getrusage(RUSAGE_CHILDREN).ru_maxrss is a cumulative high-water mark across
# every child ever reaped, so it cannot attribute memory to a particular run:
# once a large child has run, later smaller ones leave it unchanged. Instead
# each child is polled through /proc/<pid>/status VmHWM, which is that
# process's own peak.
# ---------------------------------------------------------------------------
def peak_rss(cmd, cwd=ROOT, poll=0.02):
    p = subprocess.Popen(cmd, cwd=cwd, env=_env(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hwm = 0
    status = pathlib.Path(f"/proc/{p.pid}/status")
    while p.poll() is None:
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    hwm = max(hwm, int(line.split()[1]) * 1024)
                    break
        except (OSError, ValueError):
            pass
        time.sleep(poll)
    try:
        out, _ = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill(); out = b""
    return hwm, out.decode(errors="replace")

CLS_CMD = {
    "SPIRES":      lambda n, t: [str(ROOT / "src/bench/cls_spires"), "--n", str(n), "--task", t],
    "reservoirpy": lambda n, t: [PY, str(ROOT / "src/bench/cls_reservoirpy.py"), "--n", str(n), "--task", t],
    "pytorch-esn": lambda n, t: [PY, str(ROOT / "src/bench/cls_pytorch_esn.py"), "--n", str(n), "--task", t],
    "snnTorch":    lambda n, t: [PY, str(ROOT / "src/bench/cls_snntorch.py"), "--n", str(n), "--task", t],
    "norse":       lambda n, t: [PY, str(ROOT / "src/bench/cls_norse.py"), "--n", str(n), "--task", t],
    "rockpool":    lambda n, t: [PY, str(ROOT / "src/bench/cls_rockpool.py"), "--n", str(n), "--task", t],
    "brian2":      lambda n, t: [PY, str(ROOT / "src/bench/cls_brian2.py"), "--n", str(n), "--task", t],
    "NEST":        lambda n, t: [PY, str(ROOT / "src/bench/cls_nest.py"), "--n", str(n), "--task", t],
    "ANNarchy":    lambda n, t: [PY, str(ROOT / "src/bench/cls_annarchy.py"), "--n", str(n), "--task", t],
    "RC.jl":       lambda n, t: ["julia", str(ROOT / "src/bench/cls_rc.jl"), "--n", str(n), "--task", t],
}

def measure_rss(n=200, task="fsdd", frameworks=None):
    out = {}
    for fw in (frameworks or list(CLS_CMD)):
        hwm, _ = peak_rss(CLS_CMD[fw](n, task))
        out[fw] = hwm
        print(f"  {fw:14s} peak RSS {hwm/1e6:8.1f} MB", flush=True)
    return out


if __name__ == "__main__":
    main()
