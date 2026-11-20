#!/usr/bin/env python
"""Interleaving benchmark driver.

Cells (framework x size x topology x repeat) are run round-robin rather than
framework-by-framework. On a ULV laptop that throttles under sustained load,
running each framework's sweep to completion would confound framework identity
with package temperature -- whichever ran last would be penalised. Interleaving
spreads any thermal drift evenly across frameworks, and repeats let the
aggregator report a median with spread instead of a single sample.

Usage:
  python orchestrate.py --sizes 100,200,500 --reps 3 --out results/run.jsonl
"""
import argparse, itertools, json, os, pathlib, random, resource, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENV = pathlib.Path(os.environ.get("BENCH_VENV", pathlib.Path.home() / ".pyvenv"))
PY   = str(VENV / "bin" / "python")
BENCH = ROOT / "src" / "bench"

# (a) Every framework otherwise grabs as many cores as it likes, so the
# measurement reports how well each one happens to exploit this machine rather
# than how fast it is. Pinning makes the comparison controlled; 1 thread is
# also the honest setting for an embedded-target argument. Override with
# BENCH_THREADS to measure a multicore configuration.
NUM_THREADS = os.environ.get("BENCH_THREADS", "1")

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "JULIA_NUM_THREADS", "RAYON_NUM_THREADS", "TORCH_NUM_THREADS")

def cpu_governor():
    try:
        return pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return "unknown"

def _env(extra=None):
    e = os.environ.copy()
    for v in THREAD_VARS:
        e[v] = NUM_THREADS
    # ANNarchy's CMake resolves Python via find_package; without VIRTUAL_ENV it
    # picks the system interpreter and cannot find nanobind.
    e["VIRTUAL_ENV"] = str(VENV)
    e["PATH"] = f"{VENV}/bin:" + e.get("PATH", "")
    e["JULIA_DEPOT_PATH"] = os.environ.get("JULIA_DEPOT_PATH", str(pathlib.Path.home() / ".julia"))
    if extra: e.update(extra)
    return e

# framework -> (command builder, supports_topology)
# The Python runners load config/tuned.json themselves; the SPIRES C binary
# and the Julia runner cannot, so the orchestrator passes their tuned values
# on the command line.
TUNED_PATH = ROOT / "config" / "tuned.json"

def tuned(fw, task):
    if not TUNED_PATH.exists():
        return {}
    return json.loads(TUNED_PATH.read_text()).get(f"{fw}|{task}", {})

def _flags(fw, task, keys):
    out = []
    for k, v in tuned(fw, task).items():
        if k in keys:
            out += [f"--{k}", repr(float(v))]
    return out

FRAMEWORKS = {
    "SPIRES":       (lambda n, r, t, k: [str(BENCH / "run_spires"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k]
                                        + _flags("SPIRES", k, {"gain", "sr", "leak", "vth", "ridge"}), True),
    "reservoirpy":  (lambda n, r, t, k: [PY, str(BENCH / "run_reservoirpy.py"), "--n", str(n), "--rep", str(r), "--task", k], False),
    "pytorch-esn":  (lambda n, r, t, k: [PY, str(BENCH / "run_pytorch_esn.py"), "--n", str(n), "--rep", str(r), "--task", k], False),
    "RC.jl":        (lambda n, r, t, k: ["julia", str(BENCH / "run_rc.jl"), "--n", str(n), "--rep", str(r), "--model", "esn", "--task", k]
                                        + _flags("RC.jl", k, {"sr", "leak", "gain", "ridge"}), False),
    "snnTorch":     (lambda n, r, t, k: [PY, str(BENCH / "run_snntorch.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k], True),
    "norse":        (lambda n, r, t, k: [PY, str(BENCH / "run_norse.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k], True),
    "rockpool":     (lambda n, r, t, k: [PY, str(BENCH / "run_rockpool.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k], True),
    "brian2":       (lambda n, r, t, k: [PY, str(BENCH / "run_brian2.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k], True),
    "NEST":         (lambda n, r, t, k: [PY, str(BENCH / "run_nest.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k], True),
    "ANNarchy":     (lambda n, r, t, k: [PY, str(BENCH / "run_annarchy.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k], True),
    # Equal-work variants (R1 comment 3): one integration step per input
    # sample, as SPIRES takes, with the same tuned parameters. Reported
    # alongside the idiomatic 0.1 ms rows, not instead of them.
    "brian2-1step": (lambda n, r, t, k: [PY, str(BENCH / "run_brian2.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k, "--one-step"], True),
    "NEST-1step":   (lambda n, r, t, k: [PY, str(BENCH / "run_nest.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k, "--one-step"], True),
}

# Classification tasks use separate runners: the loop structure differs
# (per-sample reset, temporal pooling, multi-class readout), so overloading
# the regression runners would have made both harder to follow.
CLS_FRAMEWORKS = {
    "SPIRES":      lambda n, r, t, k: [str(BENCH / "cls_spires"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k]
                                      + _flags("SPIRES", k, {"gain", "sr", "leak", "vth", "ei"}),
    "reservoirpy": lambda n, r, t, k: [PY, str(BENCH / "cls_reservoirpy.py"), "--n", str(n), "--rep", str(r), "--task", k],
    "pytorch-esn": lambda n, r, t, k: [PY, str(BENCH / "cls_pytorch_esn.py"), "--n", str(n), "--rep", str(r), "--task", k],
    "RC.jl":       lambda n, r, t, k: ["julia", str(BENCH / "cls_rc.jl"), "--n", str(n), "--rep", str(r), "--task", k]
                                      + _flags("RC.jl", k, {"sr", "leak", "gain"}),
    "snnTorch":    lambda n, r, t, k: [PY, str(BENCH / "cls_snntorch.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k],
    "norse":       lambda n, r, t, k: [PY, str(BENCH / "cls_norse.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k],
    "rockpool":    lambda n, r, t, k: [PY, str(BENCH / "cls_rockpool.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k],
    "brian2":      lambda n, r, t, k: [PY, str(BENCH / "cls_brian2.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k],
    "NEST":        lambda n, r, t, k: [PY, str(BENCH / "cls_nest.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k],
    "ANNarchy":    lambda n, r, t, k: [PY, str(BENCH / "cls_annarchy.py"), "--n", str(n), "--rep", str(r), "--topology", t, "--task", k],
}

CLASSIFICATION_TASKS = ("shd", "fsdd")


def _spin(seconds):
    """Busy-wait, to bring the core's clock up before a short measurement."""
    end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < end:
        x += 1.0
    return x

def run_cell(name, n, rep, topo, task, timeout):
    if task in CLASSIFICATION_TASKS:
        cmd = CLS_FRAMEWORKS[name](n, rep, topo, task)
    else:
        build, _ = FRAMEWORKS[name]
        cmd = build(n, rep, topo, task)
    # (e) The fastest frameworks are measured in tens of milliseconds, short
    # enough that a core still ramping out of a low-frequency state dominates
    # the result. Burn a little CPU first so every cell starts from a boosted
    # clock rather than whatever state the previous idle left behind.
    _spin(0.15)

    load1 = os.getloadavg()[0]
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=_env(), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"framework": name, "n": n, "rep": rep, "topology": topo,
                "task": task, "status": "timeout", "wall_s": time.time() - t0}
    for line in p.stdout.splitlines():
        if line.startswith("@@BENCH@@"):
            rec = json.loads(line[len("@@BENCH@@"):])
            rec["status"] = "ok"; rec["wall_s"] = time.time() - t0
            # (c)/(d) Cells run sequentially, so the RUSAGE_CHILDREN delta is
            # this child's CPU time. With threads pinned to 1, cpu_s well below
            # wall_s means the process was descheduled -- i.e. this cell was
            # measured against competing load and should not be trusted.
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            rec["cpu_s"] = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
            rec["load1"] = load1
            rec["threads"] = int(NUM_THREADS)
            rec["governor"] = cpu_governor()
            return rec
    return {"framework": name, "n": n, "rep": rep, "topology": topo,
            "task": task, "status": "error", "wall_s": time.time() - t0,
            "stderr": p.stderr[-400:]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,200,400,800,1400,2000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--topologies", default="random")
    ap.add_argument("--tasks", default="narma10")
    ap.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="results/run.jsonl")
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    topos = a.topologies.split(",")
    names = [x for x in a.frameworks.split(",") if x in FRAMEWORKS]

    tasks = a.tasks.split(",")
    cells = []
    for rep in range(a.reps):
        for n in sizes:
            for task in tasks:
                for topo in topos:
                    for name in names:
                        _, supports = FRAMEWORKS[name]
                        if not supports and topo != topos[0]:
                            continue   # ESN libs use their native random init
                        cells.append((name, n, rep, topo, task))
        # shuffle within a repeat so ordering effects do not align with any
        # one framework across repeats
    # Shuffle the whole cell list so ordering effects (thermal drift, cache
    # state) do not align with any one framework. A per-repeat slice shuffle
    # would have been a no-op: list slices are copies.
    random.Random(1234).shuffle(cells)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out.open("w") as fh:
        for name, n, rep, topo, task in cells:
            rec = run_cell(name, n, rep, topo, task, a.timeout)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            done += 1
            th = rec.get("t_harvest"); st = rec.get("status")
            msg = f"{th:.4f}s" if isinstance(th, float) else st
            print(f"[{done}/{len(cells)}] {name:14s} n={n:<5d} {task:13s} {topo:12s} "
                  f"rep={rep} -> {msg}", flush=True)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
