#!/usr/bin/env python
"""Re-run every correctness check and record the results as JSON.

Consolidates the three verification strands so the paper's correctness
section is reproducible from a single command:
  * integer-order discrete LIF equivalence (verify_spires)
  * fractional GL equivalence and the alpha -> 1 Euler limit (verify_flif)
  * config-surface conformance (connectivity, ei_ratio, spectral_radius)
"""
import json, os, pathlib, struct, subprocess, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "src" / "bench"
PY = str(pathlib.Path(os.environ.get("BENCH_VENV", pathlib.Path.home() / ".pyvenv"))
         / "bin" / "python")

def sh(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

# ---------- 1. integer-order discrete LIF ----------
def lif_equivalence(reps=10):
    """Each configuration is verified on `reps` independent networks, so the
    reported agreement carries a spread rather than resting on one draw."""
    import verify_spires as vs
    rows = []
    for n, T, leak, seed0 in [(64,200,0.15,7), (256,1000,0.05,1), (500,2000,0.3,2),
                              (1000,1000,0.1,3), (128,5000,0.5,4)]:
        md, ctrl, agree = [], [], []
        for k in range(reps):
            out = f"/tmp/vs_{n}_{T}_{k}.bin"
            sh([str(BENCH/"verify_spires"), "--n", str(n), "--T", str(T),
                "--leak", str(leak), "--seed", str(seed0 + k), "--out", out])
            d = vs.load(out)
            R = vs.reference(d, False)
            diff = np.abs(R - d["V"])
            md.append(float(diff.max()))
            agree.append(float(((R > d["vth"]) == (d["V"] > d["vth"])).mean()))
            Rt = vs.reference(d, True)
            bad = np.where(np.abs(Rt - d["V"]).max(axis=1) > 1e-9)[0]
            ctrl.append(int(bad[0]) if len(bad) else T)
        md, ctrl, agree = np.array(md), np.array(ctrl, float), np.array(agree)
        rows.append(dict(n=n, T=T, leak=leak, seeds=reps,
                         max_abs_diff_mean=float(md.mean()),
                         max_abs_diff_sd=float(md.std(ddof=1)),
                         max_abs_diff_max=float(md.max()),
                         spike_agreement_mean=float(agree.mean()),
                         spike_agreement_sd=float(agree.std(ddof=1)),
                         control_divergent_step_mean=float(ctrl.mean()),
                         control_divergent_step_sd=float(ctrl.std(ddof=1))))
        print(f"  LIF n={n} T={T}: max|dV|={md.mean():.2e}+-{md.std(ddof=1):.1e} "
              f"over {reps} nets; control diverges at "
              f"{ctrl.mean():.1f}+-{ctrl.std(ddof=1):.1f}")
    return rows

# ---------- 2. fractional GL ----------
def gl_equivalence():
    import verify_flif as vf
    def gl_ref(d):
        L = int(round(d["T_mem"]/d["dt"])); c = vf.gl_coefficients(d["alpha"], L)
        W, Win, u = d["W"], d["Win"], d["u"]; dta = d["dt"]**d["alpha"]
        n = d["n"]; hist = np.zeros((L, n)); v = np.full(n, d["V_rest"]); s = np.zeros(n)
        out = np.empty((d["T"], n))
        for t in range(d["T"]):
            I = W@s + Win*u[t]
            v = dta*(-(v-d["V_rest"])/d["tau_m"] + I + d["bias"]) - c[:-1] @ hist[:-1]
            s = (v >= d["V_th"]).astype(float); v = np.where(s>0, d["V_reset"], v)
            hist = np.roll(hist,1,axis=0); hist[0] = v; out[t] = v
        return out
    def euler_ref(d):
        W, Win, u = d["W"], d["Win"], d["u"]
        n=d["n"]; v=np.full(n,d["V_rest"]); s=np.zeros(n); out=np.empty((d["T"],n))
        for t in range(d["T"]):
            I = W@s + Win*u[t]
            v = v + d["dt"]*(-(v-d["V_rest"])/d["tau_m"] + I + d["bias"])
            s=(v>=d["V_th"]).astype(float); v=np.where(s>0,d["V_reset"],v); out[t]=v
        return out

    REPS = 10
    rows = []
    for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        md, eul = [], []
        for k in range(REPS):
            sh([str(BENCH/"verify_flif"), "--n","32","--T","200","--alpha",str(alpha),
                "--dt","1.0","--tmem","50","--seed",str(100+k),"--out","/tmp/gl.bin"])
            d = vf.load("/tmp/gl.bin"); R = gl_ref(d)
            md.append(float(np.abs(R - d["V"]).max()))
            if abs(alpha-1.0) < 1e-12:
                eul.append(float(np.abs(euler_ref(d) - d["V"]).max()))
        md = np.array(md)
        row = dict(alpha=alpha, L=int(round(d["T_mem"]/d["dt"])), seeds=REPS,
                   max_abs_diff_mean=float(md.mean()),
                   max_abs_diff_sd=float(md.std(ddof=1)),
                   max_abs_diff_max=float(md.max()))
        if eul:
            e = np.array(eul)
            row["euler_limit_mean"] = float(e.mean())
            row["euler_limit_sd"] = float(e.std(ddof=1))
        rows.append(row)
        print(f"  GL alpha={alpha}: max|dV|={md.mean():.2e}+-{md.std(ddof=1):.1e} over {REPS} nets")

    lrows = []
    for tmem in [20,50,100]:
        for alpha in [0.3,0.7,0.9]:
            md = []
            for k in range(REPS):
                sh([str(BENCH/"verify_flif"), "--n","32","--T","200","--alpha",str(alpha),
                    "--dt","1.0","--tmem",str(tmem),"--seed",str(200+k),"--out","/tmp/gl2.bin"])
                d = vf.load("/tmp/gl2.bin"); R = gl_ref(d)
                md.append(float(np.abs(R-d["V"]).max()))
            md = np.array(md)
            lrows.append(dict(L=tmem, alpha=alpha, seeds=REPS,
                              max_abs_diff_mean=float(md.mean()),
                              max_abs_diff_sd=float(md.std(ddof=1))))
    return dict(alpha_sweep=rows, L_sweep=lrows)

# ---------- 3. config conformance ----------
def conformance():
    out = {"connectivity": [], "ei_ratio": [], "spectral_radius": []}
    def getW(n, conn, ei, sr, seed):
        sh([str(BENCH/"spires_rho_probe"), "--n",str(n), "--conn",str(conn),
            "--ei",str(ei), "--sr",str(sr), "--seed",str(seed), "--out","/tmp/cw.bin"])
        return np.fromfile("/tmp/cw.bin", dtype=np.float64).reshape(n,n)
    for conn in [0.05,0.1,0.2]:
        W = getW(300, conn, 0.65, 1.0, 11)
        m = float((W!=0).mean())
        out["connectivity"].append(dict(requested=conn, measured=m,
                                        rel_error=abs(m-conn)/conn))
    for ei in [0.5,0.65,0.8,0.95]:
        W = getW(300, 0.1, ei, 1.0, 11); nz = W[W!=0]
        m = float((nz>0).mean())
        out["ei_ratio"].append(dict(requested=ei, measured=m,
                                    rel_error=abs(m-ei)/ei))
    for ei in [0.5,0.65,0.8]:
        ratios = []
        for seed in range(1,25):
            W = getW(300, 0.1, ei, 1.2, seed)
            ratios.append(float(np.abs(np.linalg.eigvals(W)).max())/1.2)
        r = np.array(ratios)
        out["spectral_radius"].append(dict(
            ei_ratio=ei, seeds=len(r),
            n_off_by_more_than_2pct=int((np.abs(r-1.0)>0.02).sum()),
            ratio_min=float(r.min()), ratio_max=float(r.max()),
            ratio_median=float(np.median(r))))
        print(f"  rho conformance ei={ei}: "
              f"{out['spectral_radius'][-1]['n_off_by_more_than_2pct']}/{len(r)} off >2%")
    return out

if __name__ == "__main__":
    res = {}
    print("1/3 integer-order discrete LIF equivalence")
    res["lif_equivalence"] = lif_equivalence()
    print("2/3 fractional GL equivalence")
    res["gl_equivalence"] = gl_equivalence()
    print("3/3 config-surface conformance")
    res["conformance"] = conformance()
    p = ROOT / "results" / "verification.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")
