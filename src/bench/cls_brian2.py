#!/usr/bin/env python
"""Brian2 LSM on a classification task.

All samples are concatenated into one simulation, with membrane potential
reset to zero at each sample boundary by a run_regularly operation. Running
brian2 once per sample would dominate the measurement with per-run setup
cost rather than integration cost.

Brian2 keeps its own default dt (0.1 ms) -- the finer timestep is part of
what it is for, and therefore part of its cost.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("brian2 cls").parse_args()
    import brian2 as b2
    b2.prefs.codegen.target = "cython"
    b2.set_device("runtime")
    b2.seed(a.seed + a.rep)

    def build(N, C, P, meta):
        T = meta["T"]
        W, _ = common.make_reservoir(N, seed=a.seed + a.rep, topology=a.topology,
                                     gain=1.0, spectral_radius=P["sr"])
        Win = common.make_input_weights(N, C, seed=a.seed + a.rep, gain=P["gain"])
        counter = {"spikes": 0, "steps": 0}

        def batch(X):
            ns, Tn, Cn = X.shape
            drive = (X.reshape(ns * Tn, Cn).astype(np.float64) @ Win.T)
            b2.start_scope()
            tau = (1.0 / P["leak"]) * b2.ms
            stim = b2.TimedArray(drive, dt=1 * b2.ms)
            eqs = """dv/dt = (-v + stim(t, i))/tau : 1 (unless refractory)"""
            G = b2.NeuronGroup(N, eqs, threshold=f"v>{P['vth']}", reset="v=0",
                               refractory=1 * b2.ms, method="exact",
                               namespace={"tau": tau, "stim": stim})
            S = b2.Synapses(G, G, "w : 1", on_pre="v_post += w")
            pre, post = np.nonzero(W)
            S.connect(i=pre, j=post); S.w = W[pre, post] * P["wscale"]
            # zero the state at each sample boundary
            G.run_regularly("v = 0", dt=Tn * b2.ms, when="start")
            M = b2.StateMonitor(G, "v", record=True, dt=1 * b2.ms)
            SM = b2.SpikeMonitor(G)
            net = b2.Network(G, S, M, SM)
            net.run(ns * Tn * b2.ms)
            V = np.asarray(M.v).T[: ns * Tn].reshape(ns, Tn, N)
            counter["spikes"] += SM.num_spikes; counter["steps"] += ns * Tn
            return np.stack([common.pool_states(v.astype(np.float64)) for v in V])

        def stats():
            return common.spike_stats(counter["spikes"], N, counter["steps"])
        return batch, stats

    cls_common.run_batch("brian2", "lsm", a, build,
                         dt=1e-4,
                         notes="concatenated run with per-sample reset; "
                               "idiomatic dt=0.1ms; shared multi-class ridge")

if __name__ == "__main__":
    main()
