#!/usr/bin/env python
"""Brian2 as an LSM, run idiomatically.

Brian2 is a biophysical simulator: it integrates with its own default
dt (0.1 ms) rather than one step per input sample. We keep that default --
the finer timestep is part of what Brian2 is for, and therefore part of its
cost. Each input sample is held for 1 ms, so a T-sample series is a T-ms
simulation = 10*T integration steps, with membrane potential recorded once
per sample.

Brian2 is run under its `cpp_standalone` device rather than the `cython`
runtime target. This is the only mode in which Brian2 exposes OpenMP, so it
is the only mode in which a thread budget means anything to it; measuring the
runtime target under eight threads would report a property of the mode rather
than of the simulator. Standalone generates, compiles and then executes a
separate binary, which moves the cost structure: code generation and
compilation happen once at build time and are reported separately as
`codegen_s`, while the timed harvest region is the execution of that binary.
Because the work happens in a child process, harvest CPU time is taken from
RUSAGE_CHILDREN rather than from the parent's process clock.

Brian2 emits a warning that its OpenMP support is not well tested. That
caveat belongs to the simulator, not to this harness, and is recorded in
`notes` so it reaches the results file.
"""
import sys, os, pathlib, shutil, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("brian2").parse_args()
    import brian2 as b2

    # The orchestrator pins every framework through the standard thread vars.
    # openmp_threads=0 disables OpenMP entirely, which is what a one-thread
    # budget should mean; anything above 1 turns it on with that many threads.
    # BRIAN2_MODE=runtime measures the Python-facing default instead: no
    # explicit build step, code generated and cached transparently on first
    # use. It is the mode a user gets without opting into standalone, and it
    # exposes no OpenMP, so its eight-thread column measures the mode.
    mode = os.environ.get("BRIAN2_MODE", "standalone")
    nthreads = int(os.environ.get("OMP_NUM_THREADS", "1"))

    if mode == "standalone":
        b2.prefs.devices.cpp_standalone.openmp_threads = (
            0 if nthreads <= 1 else nthreads)
        build_dir = tempfile.mkdtemp(prefix="brian2_sa_")
        b2.set_device("cpp_standalone", build_on_run=False, directory=build_dir)
    else:
        build_dir = None
        b2.prefs.codegen.target = os.environ.get("BRIAN2_TARGET", "cython")
    b2.seed(a.seed + a.rep)

    # Idiomatic: Brian2's default 0.1 ms clock, ten steps per 1 ms sample.
    # --one-step: a 1 ms clock, one step per sample, the same work per sample
    # as SPIRES (equal-work comparison). Integration is "exact", so the
    # subthreshold dynamics are unchanged; only spike-time resolution differs.
    dt_ms = 1.0 if a.one_step else 0.1
    steps_per_sample = int(round(1.0 / dt_ms))
    b2.defaultclock.dt = dt_ms * b2.ms
    step_note = ("equal work: dt=1ms (1 step/sample)" if a.one_step
                 else "idiomatic brian2 dt=0.1ms (10 steps/sample)")

    u, y, wash = common.load_data(a.task)
    T = len(u)
    P = common.resolve(a, "brian2")
    W, Win = common.make_reservoir(a.n, seed=a.seed + a.rep, topology=a.topology,
                                   gain=P["gain"], spectral_radius=P["sr"])

    tau = (1.0 / P["leak"]) * b2.ms
    stim = b2.TimedArray(u, dt=1 * b2.ms)
    eqs = """dv/dt = (-v + I)/tau : 1 (unless refractory)
             I = gain * stim(t) : 1
             gain : 1"""
    # namespace passed explicitly: `tau`/`stim` are locals of this function,
    # and Brian2 resolves identifiers lazily at run() time, by which point
    # this frame is gone.
    G = b2.NeuronGroup(a.n, eqs, threshold=f"v>{P['vth']}", reset="v=0",
                       refractory=1 * b2.ms, method="exact",
                       namespace={"tau": tau, "stim": stim})
    G.gain = Win
    S = b2.Synapses(G, G, "w : 1", on_pre="v_post += w")
    pre, post = np.nonzero(W)
    S.connect(i=pre, j=post)
    S.w = W[pre, post] * P["wscale"]
    M = b2.StateMonitor(G, "v", record=True, dt=1 * b2.ms)
    net = b2.Network(G, S, M)

    if mode == "standalone":
        # Under build_on_run=False this records the run rather than performing it.
        net.run(T * b2.ms)

        # Code generation and compilation, excluded from the timed region
        # exactly as the cython warmup was.
        with common.timer(children=True) as tc:
            b2.device.build(directory=build_dir, compile=True, run=False)

        with common.timer(children=True) as th:
            b2.device.run(directory=build_dir)
        codegen_s = tc.dt
    else:
        # Runtime target: the run performs the work. Cython code is compiled on
        # first use and cached on disk, so across repeats this measures the
        # cached path, which is what a user sees in steady state. There is no
        # separable build step to report.
        # In-process, so CPU comes from this process rather than RUSAGE_CHILDREN.
        with common.timer() as th:
            net.run(T * b2.ms)
        codegen_s = None
    states = np.asarray(M.v).T.astype(np.float64)

    states = states[:T]
    with common.timer() as tf:
        w = common.ridge_fit(states[wash:common.TRAIN_END], y[wash:common.TRAIN_END], reg=P["ridge"])

    pred = common.ridge_predict(states, w)
    if mode == "standalone":
        name, note = "brian2", (
            "cpp_standalone device, the only mode exposing OpenMP; "
            "build/compile excluded from t_harvest and reported as codegen_s; "
            "harvest CPU from RUSAGE_CHILDREN; brian2 warns its OpenMP is not "
            "well tested")
    else:
        name, note = "brian2 (runtime)", (
            f"runtime device, codegen target "
            f"{b2.prefs.codegen.target}, the Python-facing default; no "
            "separable build step, cython output cached on disk across "
            "repeats; single-threaded, exposes no OpenMP")
    if a.one_step:
        name += " (1 step/sample)"
    common.emit(framework=name, group="lsm", n=a.n, rep=a.rep, task=a.task,
                topology=a.topology, t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y, wash), native_readout=False,
                dt=dt_ms * 1e-3, sim_steps=steps_per_sample * T, codegen_s=codegen_s,
                notes=step_note + "; " + note + "; shared numpy ridge")
    if build_dir:
        shutil.rmtree(build_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
