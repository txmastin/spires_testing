#!/usr/bin/env python
"""Record the firing rate of each spiking framework at its tuned configuration.

A reservoir that never fires can still score well -- its membrane potential
degenerates into a leaky linear filter -- so the benchmark must show that the
reported LSM results come from networks that actually spike.
"""
import json, os, pathlib, re, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "src" / "bench"
VENV = pathlib.Path(os.environ.get("BENCH_VENV", pathlib.Path.home() / ".pyvenv"))
PY = str(VENV / "bin" / "python")

def env():
    e = os.environ.copy()
    e["VIRTUAL_ENV"] = str(VENV); e["PATH"] = f"{VENV}/bin:" + e.get("PATH", "")
    return e

TUNED = json.loads((ROOT / "config" / "tuned.json").read_text())

DIRECT = {  # runners that report spike_rate_hz themselves
    "SPIRES":   lambda t: [str(BENCH/"run_spires"), "--n","300","--task",t] +
                          sum([[f"--{k}", str(v)] for k,v in TUNED.get(f"SPIRES|{t}",{}).items()], []),
    "snnTorch": lambda t: [PY, str(BENCH/"run_snntorch.py"), "--n","300","--task",t],
    "norse":    lambda t: [PY, str(BENCH/"run_norse.py"), "--n","300","--task",t],
    "rockpool": lambda t: [PY, str(BENCH/"run_rockpool.py"), "--n","300","--task",t],
}
AUDIT = ["brian2", "NEST", "ANNarchy"]   # measured by audit_spiking.py

out = {}
for task in ("narma10", "mackey_glass"):
    out[task] = {}
    for fw, mk in DIRECT.items():
        p = subprocess.run(mk(task), cwd=ROOT, env=env(), capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith("@@BENCH@@"):
                r = json.loads(line[len("@@BENCH@@"):])
                out[task][fw] = dict(rate_hz=r.get("spike_rate_hz"),
                                     frac_active=r.get("frac_active"))
        print(f"  {task:14s} {fw:10s} {out[task].get(fw)}", flush=True)
    for fw in AUDIT:
        p = subprocess.run([PY, str(BENCH/"audit_spiking.py"), fw, task, "300"],
                           cwd=ROOT, env=env(), capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith("AUDIT"):
                # audit_spiking pads the value, so "rate=" and the number are
                # separate whitespace tokens; match the number directly.
                m = re.search(r"rate=\s*([0-9.]+)", line)
                if m:
                    out[task][fw] = dict(rate_hz=float(m.group(1)))
        print(f"  {task:14s} {fw:10s} {out[task].get(fw)}", flush=True)

p = ROOT / "results" / "firing_rates.json"
p.write_text(json.dumps(out, indent=2))
print(f"\nwrote {p}")
