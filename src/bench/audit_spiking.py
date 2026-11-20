#!/usr/bin/env python
"""Audit: do the tuned configurations actually spike?

Separate from the runners so their timed paths stay untouched -- adding spike
monitors inside a timed region would change the published numbers. This
reproduces each simulator's tuned setup, attaches a spike recorder, and
reports firing rate. A near-zero rate means the reservoir is scoring as a
leaky linear filter rather than as an LSM.
"""
import os
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def tuned(fw, task):
    return json.load(open(common.ROOT / "config" / "tuned.json")).get(f"{fw}|{task}", {})

def P_for(fw, task):
    P = dict(common.DEFAULTS); P.update(tuned(fw, task)); return P

def audit_brian2(n, task):
    import brian2 as b2
    P = P_for("brian2", task)
    u, y, wash = common.load_data(task); T = len(u)
    W, Win = common.make_reservoir(n, seed=0, gain=P["gain"], spectral_radius=P["sr"])
    b2.prefs.codegen.target = "cython"; b2.start_scope()
    tau = (1.0 / P["leak"]) * b2.ms
    stim = b2.TimedArray(u, dt=1 * b2.ms)
    eqs = """dv/dt = (-v + I)/tau : 1 (unless refractory)
             I = gain * stim(t) : 1
             gain : 1"""
    G = b2.NeuronGroup(n, eqs, threshold=f"v>{P['vth']}", reset="v=0",
                       refractory=1 * b2.ms, method="exact",
                       namespace={"tau": tau, "stim": stim})
    G.gain = Win
    S = b2.Synapses(G, G, "w : 1", on_pre="v_post += w")
    pre, post = np.nonzero(W); S.connect(i=pre, j=post); S.w = W[pre, post] * P["wscale"]
    M = b2.SpikeMonitor(G)
    b2.Network(G, S, M).run(T * b2.ms)
    return M.num_spikes, n, T

def audit_nest(n, task):
    import nest
    nest.set_verbosity("M_ERROR"); nest.ResetKernel()
    nest.local_num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))  # honour the orchestrator's thread budget; nest.resolution = 0.1
    P = P_for("NEST", task)
    u, y, wash = common.load_data(task); T = len(u)
    W, Win = common.make_reservoir(n, seed=0, gain=P["gain"], spectral_radius=P["sr"])
    pop = nest.Create("iaf_psc_alpha", n)
    times = (np.arange(T) + 1).astype(float)
    amps = np.outer(u, Win) * 500.0 * P["wscale"]
    gens = nest.Create("step_current_generator", n)
    gens.set([{"amplitude_times": times, "amplitude_values": amps[:, i]} for i in range(n)])
    nest.Connect(gens, pop, "one_to_one")
    pre, post = np.nonzero(W)
    nest.Connect(pre.astype(int) + pop[0].global_id, post.astype(int) + pop[0].global_id,
                 conn_spec="one_to_one",
                 syn_spec={"weight": np.ascontiguousarray(W[pre, post] * 100.0 * P["wscale"]),
                           "delay": np.full(len(pre), 1.0)})
    sr = nest.Create("spike_recorder"); nest.Connect(pop, sr)
    nest.Simulate(float(T) + 1.0)
    return sr.get("n_events"), n, T

def audit_annarchy(n, task):
    sys.argv = sys.argv[:1]
    import ANNarchy as ann
    P = P_for("ANNarchy", task)
    u, y, wash = common.load_data(task); T = len(u)
    W, Win = common.make_reservoir(n, seed=0, gain=P["gain"], spectral_radius=P["sr"])
    LIF = ann.Neuron(parameters=f"tau = {1.0/P['leak']}; vt = {P['vth']}; vr = 0.0",
                     equations="tau * dv/dt = -v + g_exc + g_inj",
                     spike="v > vt", reset="v = vr")
    net = ann.Network()
    inp = net.create(ann.TimedArray(rates=np.ascontiguousarray(np.outer(u, Win))))
    pop = net.create(geometry=n, neuron=LIF)
    pin = net.connect(ann.CurrentInjection(inp, pop, "inj")); pin.connect_current()
    prec = net.connect(pre=pop, post=pop, target="exc"); prec.from_matrix(W * P["wscale"])
    mon = net.monitor(pop, "spike")
    net.compile(silent=True, directory=f"/tmp/ann_audit_{task}_{n}")
    net.simulate(float(T))
    sp = mon.get("spike")
    return sum(len(v) for v in sp.values()), n, T

if __name__ == "__main__":
    fw = sys.argv[1]; task = sys.argv[2]; n = int(sys.argv[3])
    fn = {"brian2": audit_brian2, "NEST": audit_nest, "ANNarchy": audit_annarchy}[fw]
    tot, n, T = fn(n, task)
    st = common.spike_stats(tot, n, T)
    print(f"AUDIT {fw:10s} {task:14s} rate={st['spike_rate_hz']:9.2f} Hz  spikes={tot}")
