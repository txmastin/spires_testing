#!/usr/bin/env python
"""Driver for cmp_spectral: power iteration vs Arnoldi vs numpy ground truth.

Runs the C comparator over a grid of reservoir configurations, loads the
matrix it dumps, and computes the true spectral radius with a full dense
eigendecomposition. Reports both SPIRES routines against that reference.

This runs while calc_spectral_radius still points at the power iteration, so
the new routine is judged before any call site depends on it.
"""
import argparse, json, pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "src" / "bench" / "cmp_spectral"

def run_case(n, conn, ei, sr, seed, topology):
    p = subprocess.run(
        [str(EXE), "--n", str(n), "--conn", str(conn), "--ei", str(ei),
         "--sr", str(sr), "--seed", str(seed), "--topology", topology,
         "--out", "/tmp/cmp_W.bin"],
        cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-400:])
    nn, rho_pow, rho_arn = p.stdout.split()
    W = np.fromfile("/tmp/cmp_W.bin", dtype=np.float64).reshape(int(nn), int(nn))
    true = float(np.abs(np.linalg.eigvals(W)).max())
    return true, float(rho_pow), float(rho_arn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--sr", type=float, default=1.2)
    ap.add_argument("--out", default="results/cmp_spectral.json")
    a = ap.parse_args()

    rows = []
    print(f"{'topology':<12} {'ei':>5} {'seed':>4} {'true':>9} "
          f"{'power':>9} {'err%':>8} {'arnoldi':>9} {'err%':>8}")
    for topology in ("random", "small_world", "scale_free"):
        for ei in (0.5, 0.65, 0.8):
            for seed in range(a.seeds):
                true, pw, ar = run_case(a.n, 0.1, ei, a.sr, seed, topology)
                e_pw = (pw - true) / true * 100.0
                e_ar = (ar - true) / true * 100.0
                rows.append(dict(topology=topology, ei=ei, seed=seed, n=a.n,
                                 true=true, power=pw, arnoldi=ar,
                                 power_err_pct=e_pw, arnoldi_err_pct=e_ar))
                print(f"{topology:<12} {ei:>5} {seed:>4} {true:>9.4f} "
                      f"{pw:>9.4f} {e_pw:>+7.2f}% {ar:>9.4f} {e_ar:>+7.2f}%")

    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(rows, indent=2))

    print("\n=== summary (worst |relative error|) ===")
    print(f"{'topology':<12} {'ei':>5} {'power':>10} {'arnoldi':>10}")
    for topology in ("random", "small_world", "scale_free"):
        for ei in (0.5, 0.65, 0.8):
            sub = [r for r in rows if r["topology"] == topology and r["ei"] == ei]
            wp = max(abs(r["power_err_pct"]) for r in sub)
            wa = max(abs(r["arnoldi_err_pct"]) for r in sub)
            print(f"{topology:<12} {ei:>5} {wp:>9.2f}% {wa:>9.2f}%")
    allp = max(abs(r["power_err_pct"]) for r in rows)
    alla = max(abs(r["arnoldi_err_pct"]) for r in rows)
    print(f"\noverall worst: power {allp:.2f}%   arnoldi {alla:.2f}%")
    print(f"wrote {ROOT / a.out}")

if __name__ == "__main__":
    main()
