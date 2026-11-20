#!/usr/bin/env python
"""Per-(framework, task) hyperparameter search, minimising NRMSE.

Motivation: NARMA-10 and Mackey-Glass reward very different reservoir
settings, and spiking frameworks need different drive than rate-coded ones.
Benchmarking one tuned framework against several detuned ones would make the
speed comparison meaningless, so each framework gets its own small grid
search per task and the winners are written to config/tuned.json, which the
runners load automatically.

Search is at a single modest N with one repeat -- we want the parameter
region, not a precise timing. Grids are deliberately small: the point is to
get every framework into a comparable accuracy band, not to find the global
optimum for any one of them.
"""
import argparse, itertools, json, pathlib, subprocess, sys, os, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENV = pathlib.Path(os.environ.get("BENCH_VENV", pathlib.Path.home() / ".pyvenv"))
PY = str(VENV / "bin" / "python")
BENCH = ROOT / "src" / "bench"

def env():
    e = os.environ.copy()
    e["VIRTUAL_ENV"] = str(VENV)
    e["PATH"] = f"{VENV}/bin:" + e.get("PATH", "")
    e["JULIA_DEPOT_PATH"] = os.environ.get("JULIA_DEPOT_PATH", str(pathlib.Path.home() / ".julia"))
    return e

def grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*(kw[k] for k in keys))]

# Knob sets differ by framework because the models differ; each grid covers
# the parameters that actually move NRMSE for that library.
ESN_G  = grid(sr=[0.5, 0.9, 1.2], leak=[0.1, 0.3, 1.0],
              gain=[0.1, 0.5, 1.0], ridge=[1e-6, 1e-4])
SPIKE_G = grid(sr=[0.5, 0.9, 1.2], leak=[0.05, 0.2, 0.5],
               gain=[1.0, 5.0, 20.0], vth=[0.5, 1.0])
SIM_G  = grid(gain=[1.0, 5.0, 20.0], wscale=[0.5, 1.0, 2.0], leak=[0.05, 0.1])
ROCK_G = grid(alpha=[0.8, 0.95], beta=[0.3, 0.5], gain=[1.0, 5.0],
              wscale=[0.5, 1.0], vth=[1.0])

# Revision (R1 comment 3): --ridge-only. Accuracy in this benchmark is a
# validity check -- evidence that each timed reservoir does the task -- not a
# result, so the reservoir operating points stay as tuned for the submission
# and only the readout's ridge lambda is re-chosen. Scoring held-out exposed
# that the spiking readouts overfit at the default lambda = 1e-6 (SPIRES
# NARMA-10 at N=500: train 0.61, test 0.94), which would make the check
# meaningless. Lambda is selected on the *validation* segment of the split in
# common.py; the test segment is never used for selection.
RIDGE_ONLY = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4]

# SHD/FSDD drive the reservoir through 700 input channels carrying thousands
# of spikes per sample, so the useful input gain is three orders of magnitude
# below the single-channel regression tasks: gain=5 saturates the reservoir to
# near-chance accuracy, and the optimum sits around 1e-3. Classification also
# needs its own grids because the objective is accuracy, not NRMSE.
CLS_SPIKE_G = grid(gain=[1e-3, 1e-2, 0.3, 1.0, 3.0],
                   leak=[0.1, 0.3], sr=[1.2], vth=[1.0])
CLS_ESN_G   = grid(gain=[1e-2, 0.1, 0.5, 1.0, 2.0], leak=[0.3, 0.5, 1.0],
                   sr=[0.9, 1.2], ridge=[1e-6])
CLS_ROCK_G  = grid(gain=[1e-2, 0.3, 1.0, 3.0], alpha=[0.8, 0.95],
                   beta=[0.3, 0.5], wscale=[1.0], vth=[1.0])

# SPIRES gets ei_ratio in its grid: it proved to be the single most
# influential parameter on FSDD (0.624 -> 0.798 from 0.8 -> 0.5), because the
# torch-based LSMs use symmetric +/- weights while SPIRES was being run at
# 80% excitatory -- a different network, not a different implementation.
# The simulators cost 30-80 s per evaluation on a classification corpus, so
# they get a deliberately coarse grid. Under the revised framing accuracy is
# in-band evidence rather than a leaderboard, so a coarse tune that lands each
# framework in a sensible regime is sufficient -- exhaustive search would cost
# hours and change nothing about the speed result.
CLS_SLOW_G = grid(gain=[1e-2, 0.3, 1.0, 3.0], leak=[0.3], sr=[1.2], wscale=[1.0])

CLS_SPIRES_G = grid(gain=[1e-3, 1e-2, 0.3, 1.0, 3.0],
                    leak=[0.1, 0.3], sr=[1.2], vth=[1.0], ei=[0.5, 0.8])

CLS_GRIDS = {
    "SPIRES": CLS_SPIRES_G, "snnTorch": CLS_SPIKE_G, "norse": CLS_SPIKE_G,
    "brian2": CLS_SLOW_G, "NEST": CLS_SLOW_G, "ANNarchy": CLS_SLOW_G,
    "rockpool": CLS_ROCK_G,
    "reservoirpy": CLS_ESN_G, "pytorch-esn": CLS_ESN_G, "RC.jl": CLS_ESN_G,
}

GRIDS = {
    "SPIRES":       SPIKE_G,
    "reservoirpy":  ESN_G,
    "pytorch-esn":  ESN_G,
    "RC.jl":        ESN_G,
    "snnTorch":     SPIKE_G,
    "norse":        SPIKE_G,
    "rockpool":     ROCK_G,
    "brian2":       SIM_G,
    "NEST":         SIM_G,
    "ANNarchy":     SIM_G,
}

CLS_BASE = {
    "SPIRES":      lambda B: [str(BENCH / "cls_spires")],
    "reservoirpy": lambda B: [PY, str(BENCH / "cls_reservoirpy.py")],
    "pytorch-esn": lambda B: [PY, str(BENCH / "cls_pytorch_esn.py")],
    "RC.jl":       lambda B: ["julia", str(BENCH / "cls_rc.jl")],
    "snnTorch":    lambda B: [PY, str(BENCH / "cls_snntorch.py")],
    "norse":       lambda B: [PY, str(BENCH / "cls_norse.py")],
    "rockpool":    lambda B: [PY, str(BENCH / "cls_rockpool.py")],
    "brian2":      lambda B: [PY, str(BENCH / "cls_brian2.py")],
    "NEST":        lambda B: [PY, str(BENCH / "cls_nest.py")],
    "ANNarchy":    lambda B: [PY, str(BENCH / "cls_annarchy.py")],
}


def cmd(fw, n, task, P):
    base = {
        "SPIRES":      [str(BENCH / "run_spires")],
        "reservoirpy": [PY, str(BENCH / "run_reservoirpy.py")],
        "pytorch-esn": [PY, str(BENCH / "run_pytorch_esn.py")],
        "RC.jl":       ["julia", str(BENCH / "run_rc.jl")],
        "snnTorch":    [PY, str(BENCH / "run_snntorch.py")],
        "norse":       [PY, str(BENCH / "run_norse.py")],
        "rockpool":    [PY, str(BENCH / "run_rockpool.py")],
        "brian2":      [PY, str(BENCH / "run_brian2.py")],
        "NEST":        [PY, str(BENCH / "run_nest.py")],
        "ANNarchy":    [PY, str(BENCH / "run_annarchy.py")],
    }[fw]
    if task in ("shd", "fsdd"):
        base = CLS_BASE[fw](BENCH)
    c = base + ["--n", str(n), "--task", task, "--no-tuned"]
    if fw == "SPIRES":
        c = base + ["--n", str(n), "--task", task]      # C runner: own flags
    for k, v in P.items():
        c += [f"--{k}", repr(float(v))]
    return c

# Minimum mean firing rate for a configuration to count as an LSM.
#
# Without this the search can select a reservoir that never fires: its
# membrane potential degenerates into a leaky linear filter of the input,
# which on SHD scores ~0.34 -- within noise of the best genuinely spiking
# configuration. Such a result is not a liquid state machine and must not be
# what the benchmark reports. Frameworks that do not report a firing rate
# (the rate-coded ESN libraries) are exempt, since the field is absent.
MIN_RATE_HZ = 1.0


def run(fw, n, task, P, timeout, min_rate=MIN_RATE_HZ):
    try:
        p = subprocess.run(cmd(fw, n, task, P), cwd=ROOT, env=env(),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for line in p.stdout.splitlines():
        if line.startswith("@@BENCH@@"):
            r = json.loads(line[len("@@BENCH@@"):])
            rate = r.get("spike_rate_hz")
            if rate is not None and rate < min_rate:
                return None            # not spiking: reject regardless of score
            if "accuracy" in r and r["accuracy"] is not None:
                return -float(r["accuracy"])   # negate: caller minimises
            # Regression: select on the validation segment; the test segment
            # ("nrmse") is reserved for the reported result.
            e = r.get("nrmse_val")
            return e if (e is not None and np.isfinite(e)) else None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tasks", default="narma10,mackey_glass")
    ap.add_argument("--frameworks", default=",".join(GRIDS))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="config/tuned.json")
    ap.add_argument("--min-rate", type=float, default=MIN_RATE_HZ,
                    help="reject spiking configs below this mean firing rate (Hz)")
    ap.add_argument("--ridge-only", action="store_true",
                    help="keep each framework's tuned reservoir parameters and "
                         "re-select only the ridge lambda, on the validation segment")
    a = ap.parse_args()

    out = ROOT / a.out
    table = json.loads(out.read_text()) if out.exists() else {}

    for task in a.tasks.split(","):
        for fw in [f for f in a.frameworks.split(",") if f in GRIDS]:
            is_cls = task in ("shd", "fsdd")
            if is_cls and fw not in CLS_BASE:
                print(f"{fw}|{task:14s} SKIP (no classification runner yet)", flush=True)
                continue
            g = (CLS_GRIDS if is_cls else GRIDS)[fw]
            if a.ridge_only:
                if is_cls:
                    print(f"{fw}|{task:14s} SKIP (--ridge-only is for regression)", flush=True)
                    continue
                base = {k: v for k, v in table.get(f"{fw}|{task}", {}).items() if k != "ridge"}
                g = [dict(base, ridge=r) for r in RIDGE_ONLY]
            best, best_e = None, float("inf")
            t0 = time.time()
            for i, P in enumerate(g):
                e = run(fw, a.n, task, P, a.timeout, a.min_rate)
                if e is not None and e < best_e:
                    best, best_e = P, e
            key = f"{fw}|{task}"
            if best is not None:
                table[key] = best
                score = (f"acc={-best_e:7.4f}" if task in ("shd", "fsdd")
                         else f"nrmse={best_e:8.4f}")
                print(f"{key:34s} {score}  {best}  "
                      f"({len(g)} combos, {time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"{key:34s} NO VALID RESULT ({len(g)} combos)", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(table, indent=2, sort_keys=True))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
