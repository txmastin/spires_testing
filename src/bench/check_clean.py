#!/usr/bin/env python
"""Verify the timing sweeps were measured on an idle machine.

Each cell records CPU time over the same region as wall time. Under a
single-thread budget the two should coincide; a cell whose CPU time falls
materially short of its wall time was descheduled, meaning something else was
running and that measurement is not trustworthy. This turns "the machine was
quiet" from an assumption into a checkable property of the data.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

def check(path, threads, floor=0.90):
    p = ROOT / "results" / path
    if not p.exists():
        print(f"  {path}: missing"); return False
    rs = [json.loads(l) for l in open(p) if l.strip()]
    ok = [r for r in rs if r.get("status") == "ok" and r.get("cpu_total")]
    if not ok:
        print(f"  {path}: no cpu_total recorded"); return False
    ratios = [(r["cpu_total"] / r["t_total"], r) for r in ok if r["t_total"] > 0]
    if threads == 1:
        bad = [(x, r) for x, r in ratios if x < floor]
        verdict = "CLEAN" if not bad else f"{len(bad)} SUSPECT CELLS"
        print(f"  {path}: {len(ok)} cells, cpu/wall "
              f"min {min(x for x,_ in ratios):.2f} median "
              f"{sorted(x for x,_ in ratios)[len(ratios)//2]:.2f} -> {verdict}")
        for x, r in bad[:5]:
            print(f"      {r['framework']} n={r['n']} rep={r['rep']} ratio={x:.2f}")
        return not bad
    # multi-threaded: the ratio is parallel efficiency, not cleanliness
    by = {}
    for x, r in ratios:
        by.setdefault(r["framework"], []).append(x)
    print(f"  {path}: {len(ok)} cells; cpu/wall is cores-utilised here, not cleanliness")
    for fw in sorted(by, key=lambda k: -sum(by[k]) / len(by[k])):
        v = by[fw]
        print(f"      {fw:>22} {sum(v)/len(v):5.2f} cores utilised")
    return True

if __name__ == "__main__":
    print("timing-run cleanliness / parallel utilisation")
    a = check("sweep_pinned.jsonl", 1)
    b = check("sweep_t8.jsonl", 8)
    sys.exit(0 if a else 1)
