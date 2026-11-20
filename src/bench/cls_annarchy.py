#!/usr/bin/env python
"""ANNarchy LSM on a classification task.

Simulated in per-sample chunks with an explicit membrane reset at each
boundary. ANNarchy's default dt is 1.0 ms, so it already runs one step per
input frame. C++ compilation is paid once outside the timed region.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("ANNarchy cls").parse_args()
    sys.argv = sys.argv[:1]      # ANNarchy prefix-matches --rep to its --report
    import ANNarchy as ann
    ann.setup(dt=1.0, seed=a.seed + a.rep, num_threads=int(os.environ.get("OMP_NUM_THREADS", "1")))

    def build(N, C, P, meta):
        W, _ = common.make_reservoir(N, seed=a.seed + a.rep, topology=a.topology,
                                     gain=1.0, spectral_radius=P["sr"])
        Win = common.make_input_weights(N, C, seed=a.seed + a.rep, gain=P["gain"])
        counter = {"spikes": 0, "steps": 0}
        state = {}

        def batch(X):
            ns, T, Cn = X.shape
            drive = np.ascontiguousarray(
                X.reshape(ns * T, Cn).astype(np.float64) @ Win.T)
            LIF = ann.Neuron(
                parameters=f"tau = {1.0/P['leak']}; vt = {P['vth']}; vr = 0.0",
                equations="tau * dv/dt = -v + g_exc + g_inj",
                spike="v > vt", reset="v = vr")
            net = ann.Network()
            inp = net.create(ann.TimedArray(rates=drive))
            pop = net.create(geometry=N, neuron=LIF)
            pin = net.connect(ann.CurrentInjection(inp, pop, "inj")); pin.connect_current()
            prec = net.connect(pre=pop, post=pop, target="exc")
            prec.from_matrix(W * P["wscale"])
            mv = net.monitor(pop, "v"); msp = net.monitor(pop, "spike")
            d = pathlib.Path("/tmp/annarchy_cls") / f"n{N}_{a.task}_{a.rep}_{ns}"
            net.compile(silent=True, directory=str(d))
            state["compiled"] = True

            for _ in range(ns):
                net.simulate(float(T))
                pop.v = 0.0                      # reset at sample boundary
            V = np.asarray(mv.get("v"), dtype=np.float64)[: ns * T].reshape(ns, T, N)
            sp = msp.get("spike")
            counter["spikes"] += sum(len(v) for v in sp.values())
            counter["steps"] += ns * T
            return np.stack([common.pool_states(v) for v in V])

        def stats():
            return common.spike_stats(counter["spikes"], N, counter["steps"])
        return batch, stats

    cls_common.run_batch("ANNarchy", "lsm", a, build, dt=1e-3,
                         notes="chunked simulation with per-sample reset; "
                               "idiomatic dt=1.0ms; shared multi-class ridge")

if __name__ == "__main__":
    main()
