#!/usr/bin/env python
"""Combined summary table across tasks and sizes -> markdown.

Reports median with min-max across repeats, harvest and fit split out, and
NRMSE alongside, so a fast-but-inaccurate result cannot be mistaken for a
good one. Also prints the per-framework speedup relative to SPIRES.
"""
import json, sys, collections, statistics as st

TASK_LABEL = {"narma10": "NARMA-10", "mackey_glass": "Mackey-Glass"}

def load(path):
    return [json.loads(l) for l in open(path) if json.loads(l).get("status") == "ok"]

def main(path, n=None, ref="SPIRES"):
    rs = load(path)
    tasks = sorted({r.get("task", "narma10") for r in rs})
    sizes = sorted({r["n"] for r in rs}) if n is None else [n]

    for task in tasks:
        for size in sizes:
            sub = [r for r in rs if r.get("task", "narma10") == task and r["n"] == size]
            if not sub: continue
            by = collections.defaultdict(list)
            for r in sub: by[r["framework"]].append(r)
            rows = []
            for k, v in by.items():
                tot = [x["t_harvest"] + x["t_fit"] for x in v]
                rows.append(dict(
                    name=k, reps=len(v), med=st.median(tot), lo=min(tot), hi=max(tot),
                    harv=st.median(x["t_harvest"] for x in v),
                    fit=st.median(x["t_fit"] for x in v),
                    nrmse=st.median(x["nrmse"] for x in v),
                    grp="ESN" if v[0]["group"] == "esn" else "LSM"))
            rows.sort(key=lambda r: r["med"])
            base = next((r["med"] for r in rows if r["name"] == ref), None)

            print(f"\n### {TASK_LABEL.get(task, task)} — N = {size} "
                  f"(median of {rows[0]['reps']} repeats)\n")
            print("| Framework | Class | Total (s) | Harvest (s) | Fit (s) | "
                  "min–max (s) | NRMSE | vs SPIRES |")
            print("|---|---|---|---|---|---|---|---|")
            for r in rows:
                rel = f"{r['med']/base:.1f}x" if base else "—"
                print(f"| {r['name']} | {r['grp']} | {r['med']:.3f} | {r['harv']:.3f} | "
                      f"{r['fit']:.4f} | {r['lo']:.3f}–{r['hi']:.3f} | "
                      f"{r['nrmse']:.4f} | {rel} |")

if __name__ == "__main__":
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None)
