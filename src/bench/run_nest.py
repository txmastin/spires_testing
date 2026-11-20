#!/usr/bin/env python
"""NEST as an LSM, run idiomatically.

NEST is a point-neuron simulator for large networks. We keep its default
0.1 ms resolution rather than forcing one step per input sample: the finer
timestep is intrinsic to how NEST is meant to run, and so is part of its
cost. Each input sample is held for 1 ms via a per-neuron
step_current_generator, and V_m is recorded once per sample.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("nest").parse_args()
    import nest
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.local_num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))  # honour the orchestrator's thread budget
    # Idiomatic: NEST's default 0.1 ms resolution, ten steps per 1 ms sample.
    # --one-step: 1 ms resolution, one step per sample, the same work per
    # sample as SPIRES (equal-work comparison). iaf_psc_alpha is integrated
    # exactly, so only spike-time resolution changes. The 1 ms synaptic delay
    # remains valid, since it must be at least the resolution.
    dt_ms = 1.0 if a.one_step else 0.1
    nest.resolution = dt_ms
    nest.rng_seed = a.seed + a.rep + 1

    u, y, wash = common.load_data(a.task)
    T = len(u)
    P = common.resolve(a, "NEST")
    W, Win = common.make_reservoir(a.n, seed=a.seed + a.rep, topology=a.topology,
                                   gain=P["gain"], spectral_radius=P["sr"])

    pop = nest.Create("iaf_psc_alpha", a.n)

    # one step_current_generator per neuron: amplitude follows gain_i * u(t),
    # scaled to pA. This is NEST's idiomatic route for a predetermined drive.
    times = (np.arange(T) + 1).astype(float)          # ms, must be > 0
    amps = np.outer(u, Win) * 500.0 * P["wscale"]                   # (T, n) pA
    gens = nest.Create("step_current_generator", a.n)
    gens.set([{"amplitude_times": times, "amplitude_values": amps[:, i]}
              for i in range(a.n)])
    nest.Connect(gens, pop, "one_to_one")

    # recurrent connectivity from the shared topology
    pre, post = np.nonzero(W)
    # weight and delay must both be arrays matching the connection count
    nest.Connect(pre.astype(int) + pop[0].global_id,
                 post.astype(int) + pop[0].global_id,
                 conn_spec="one_to_one",
                 syn_spec={"weight": np.ascontiguousarray(W[pre, post] * 100.0 * P["wscale"]),
                           "delay": np.full(len(pre), 1.0)})

    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 1.0})
    nest.Connect(mm, pop)

    with common.timer() as th:
        nest.Simulate(float(T) + 1.0)
        ev = mm.get("events")
        vm = ev["V_m"]; sender = ev["senders"]
        order = np.argsort(sender, kind="stable")
        states = vm[order].reshape(a.n, -1).T.astype(np.float64)

    states = states[:T]
    m = min(len(states), len(y))
    with common.timer() as tf:
        w = common.ridge_fit(states[wash:common.TRAIN_END], y[wash:common.TRAIN_END], reg=P["ridge"])

    pred = common.ridge_predict(states[:m], w)
    common.emit(framework="NEST (1 step/sample)" if a.one_step else "NEST",
                group="lsm", n=a.n, rep=a.rep, task=a.task,
                topology=a.topology, t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y[:m], wash), native_readout=False,
                dt=dt_ms * 1e-3, sim_steps=int(round(1.0 / dt_ms)) * T,
                notes=("equal work: resolution 1ms (1 step/sample)" if a.one_step
                       else "idiomatic NEST resolution 0.1ms (10 steps/sample)")
                      + f", {nest.local_num_threads} threads; shared numpy ridge; V_m as state")

if __name__ == "__main__":
    main()
