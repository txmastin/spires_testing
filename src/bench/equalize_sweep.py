#!/usr/bin/env python
"""Equalized comparison: every framework on the same network, across rho.

Removes every controllable difference between frameworks:
  * identical recurrent and input weights (exported from SPIRES, so no
    framework gets a luckier draw and the weight distribution, density and
    realisation are shared),
  * identical readout (shared ridge, bias unpenalized in all of them),
  * identical input encoding, pooling and lambda sweep.

Spectral radius is swept rather than fixed. A single value would always
favour whichever genus prefers it -- rate-coded and spiking reservoirs have
different optima, and the rho<=1 echo-state rule is a heuristic from the
linearised zero-input case rather than the real boundary (which is where the
driven system's largest Lyapunov exponent crosses zero). Sweeping lets each
framework be read at its own optimum on a shared network.

Because SPIRES's own spectral-radius scaling is unreliable when weights are
sign-balanced -- csr_spectral_radius uses power iteration, which converges
only for a real dominant eigenvalue -- results are plotted against the
*measured* radius of the exported matrix, not the requested one.
"""
import argparse, json, os, pathlib, subprocess, struct, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "src" / "bench"
VENV = pathlib.Path(os.environ.get("BENCH_VENV", pathlib.Path.home() / ".pyvenv"))
PY = str(VENV / "bin" / "python")

RUNNERS = {
    "SPIRES":      lambda: [str(BENCH / "cls_spires")],
    "snnTorch":    lambda: [PY, str(BENCH / "cls_snntorch.py")],
    "norse":       lambda: [PY, str(BENCH / "cls_norse.py")],
    "rockpool":    lambda: [PY, str(BENCH / "cls_rockpool.py")],
    "reservoirpy": lambda: [PY, str(BENCH / "cls_reservoirpy.py")],
}
GROUP = {"SPIRES": "lsm", "snnTorch": "lsm", "norse": "lsm",
         "rockpool": "lsm", "reservoirpy": "esn"}

def env():
    e = os.environ.copy()
    e["VIRTUAL_ENV"] = str(VENV)
    e["PATH"] = f"{VENV}/bin:" + e.get("PATH", "")
    return e

def measured_rho(n, c, sr, ei, seed, topology):
    """Export SPIRES's matrix for this config and measure its true radius."""
    out = pathlib.Path("/tmp") / f"eqsweep_{n}_{c}_{sr}_{ei}_{seed}.bin"
    subprocess.run([str(BENCH / "spires_export"), "--n", str(n), "--c", str(c),
                    "--sr", str(sr), "--ei", str(ei), "--seed", str(seed),
                    "--topology", topology, "--out", str(out)],
                   cwd=ROOT, env=env(), check=True, capture_output=True)
    with open(out, "rb") as f:
        nn, cc = np.frombuffer(f.read(8), dtype="<i4")
        W = np.frombuffer(f.read(nn*nn*8), dtype="<f8").reshape(nn, nn)
    return float(np.abs(np.linalg.eigvals(W)).max())

def run(fw, n, task, sr, ei, gain, leak, seed, timeout):
    cmd = RUNNERS[fw]() + ["--n", str(n), "--task", task, "--sr", str(sr),
                           "--gain", str(gain), "--leak", str(leak),
                           "--seed", str(seed)]
    if fw == "SPIRES":
        cmd += ["--ei", str(ei), "--vth", "1.0"]
    else:
        cmd += ["--weights", "spires", "--no-tuned", "--vth", "1.0"]
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env(), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    for line in p.stdout.splitlines():
        if line.startswith("@@BENCH@@"):
            return json.loads(line[len("@@BENCH@@"):])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--task", default="fsdd")
    ap.add_argument("--srs", default="0.3,0.6,0.9,1.2,1.6,2.0")
    ap.add_argument("--ei", type=float, default=0.5)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--frameworks", default=",".join(RUNNERS))
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out", default="results/equalized.jsonl")
    a = ap.parse_args()

    srs = [float(x) for x in a.srs.split(",")]
    fws = [f for f in a.frameworks.split(",") if f in RUNNERS]
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as fh:
        for sr in srs:
            rho = measured_rho(a.n, 40 if a.task == "fsdd" else 700,
                               sr, a.ei, a.seed, "random")
            for fw in fws:
                t0 = time.time()
                r = run(fw, a.n, a.task, sr, a.ei, a.gain, a.leak, a.seed, a.timeout)
                if r is None:
                    rec = {"framework": fw, "requested_sr": sr, "rho": rho,
                           "status": "error"}
                else:
                    rec = {"framework": fw, "group": GROUP[fw],
                           "requested_sr": sr, "rho": rho,
                           "accuracy": r.get("accuracy"),
                           "spike_rate_hz": r.get("spike_rate_hz"),
                           "t_harvest": r.get("t_harvest"),
                           "n": a.n, "task": a.task, "status": "ok"}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                acc = rec.get("accuracy")
                print(f"sr={sr:<5} rho={rho:6.3f}  {fw:12s} "
                      f"acc={acc if acc is None else round(acc,4)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
