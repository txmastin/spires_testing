#!/usr/bin/env python
"""NEST LSM on a classification task.

The corpus is driven by per-neuron step_current_generators carrying the whole
concatenated stimulus, and simulated in per-sample chunks so V_m can be reset
at each boundary -- NEST has no periodic-reset primitive, and chunking costs
far less than rebuilding the network per sample. NEST keeps its default
0.1 ms resolution.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("nest cls").parse_args()
    import nest
    nest.set_verbosity("M_ERROR")

    def build(N, C, P, meta):
        W, _ = common.make_reservoir(N, seed=a.seed + a.rep, topology=a.topology,
                                     gain=1.0, spectral_radius=P["sr"])
        Win = common.make_input_weights(N, C, seed=a.seed + a.rep, gain=P["gain"])
        counter = {"spikes": 0, "steps": 0}

        def batch(X):
            ns, T, Cn = X.shape
            drive = (X.reshape(ns * T, Cn).astype(np.float64) @ Win.T) * 500.0 * P["wscale"]
            nest.ResetKernel()
            nest.local_num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))  # honour the orchestrator's thread budget
            nest.resolution = 0.1
            nest.rng_seed = a.seed + a.rep + 1
            pop = nest.Create("iaf_psc_alpha", N)
            times = (np.arange(ns * T) + 1).astype(float)
            gens = nest.Create("step_current_generator", N)
            gens.set([{"amplitude_times": times, "amplitude_values": drive[:, i]}
                      for i in range(N)])
            nest.Connect(gens, pop, "one_to_one")
            pre, post = np.nonzero(W)
            nest.Connect(pre.astype(int) + pop[0].global_id,
                         post.astype(int) + pop[0].global_id,
                         conn_spec="one_to_one",
                         syn_spec={"weight": np.ascontiguousarray(W[pre, post] * 100.0 * P["wscale"]),
                                   "delay": np.full(len(pre), 1.0)})
            mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 1.0})
            nest.Connect(mm, pop)
            sr = nest.Create("spike_recorder"); nest.Connect(pop, sr)

            E_L = float(nest.GetStatus(pop, "E_L")[0])
            for _ in range(ns):
                nest.Simulate(float(T))
                nest.SetStatus(pop, {"V_m": E_L})     # reset at sample boundary
            ev = mm.get("events")
            order = np.argsort(ev["senders"], kind="stable")
            V = ev["V_m"][order].reshape(N, -1).T.astype(np.float64)
            # NEST's multimeter emits its first sample at t=interval, not at
            # t=0, so the trace is one step short of the stimulus. Pad at the
            # front with the resting value to restore alignment.
            need = ns * T
            if len(V) < need:
                pad = np.full((need - len(V), N), float(nest.GetStatus(pop, "E_L")[0]))
                V = np.vstack([pad, V])
            V = V[:need].reshape(ns, T, N)
            counter["spikes"] += sr.get("n_events"); counter["steps"] += ns * T
            return np.stack([common.pool_states(v) for v in V])

        def stats():
            return common.spike_stats(counter["spikes"], N, counter["steps"])
        return batch, stats

    cls_common.run_batch("NEST", "lsm", a, build, dt=1e-4,
                         notes="chunked simulation with per-sample V_m reset; "
                               "idiomatic resolution 0.1ms; shared multi-class ridge")

if __name__ == "__main__":
    main()
