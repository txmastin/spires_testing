#!/usr/bin/env python
"""ANNarchy as an LSM, run idiomatically.

ANNarchy code-generates C++ and compiles it. Its default dt is 1.0 ms, so
unlike Brian2/NEST it already runs one integration step per input sample.
Compilation is paid once outside the timed region and reported as
`codegen_s`. Input is driven by a rate-coded TimedArray projected onto the
spiking population.

NOTE: ANNarchy's CMake step resolves the Python interpreter via
find_package(Python); with a bare venv it picks up the system interpreter
and fails to locate nanobind. The orchestrator sets VIRTUAL_ENV so CMake
selects the venv. Running this file directly requires the same.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("ANNarchy").parse_args()
    # ANNarchy parses sys.argv itself on import, with a prefix-matching
    # parser: our "--rep" is treated as an abbreviation of its own
    # "--report" flag and aborts with a filename error. Clear argv now that
    # we have our own arguments.
    sys.argv = sys.argv[:1]
    import ANNarchy as ann
    ann.setup(dt=1.0, seed=a.seed + a.rep, num_threads=int(os.environ.get("OMP_NUM_THREADS", "1")))

    u, y, wash = common.load_data(a.task)
    T = len(u)
    P = common.resolve(a, "ANNarchy")
    W, Win = common.make_reservoir(a.n, seed=a.seed + a.rep, topology=a.topology,
                                   gain=P["gain"], spectral_radius=P["sr"])

    # g_exc is the spike-driven conductance carrying the recurrent
    # projection. The rate-coded TimedArray drive is delivered through a
    # separate target via CurrentInjection, which sets g_inj each step from
    # the pre-synaptic rate; a rate population cannot drive a spiking target
    # directly, and spiking neurons expose g_<target> rather than sum().
    LIF = ann.Neuron(
        parameters=f"tau = {1.0/P['leak']}; vt = {P['vth']}; vr = 0.0",
        equations="tau * dv/dt = -v + g_exc + g_inj",
        spike="v > vt", reset="v = vr")

    net = ann.Network()
    drive = np.ascontiguousarray(np.outer(u, Win), dtype=np.float64)
    inp = net.create(ann.TimedArray(rates=drive))
    pop = net.create(geometry=a.n, neuron=LIF)

    pin = net.connect(ann.CurrentInjection(inp, pop, "inj"))
    pin.connect_current()
    prec = net.connect(pre=pop, post=pop, target="exc")
    prec.from_matrix(W * P["wscale"])

    mon = net.monitor(pop, "v")

    d = pathlib.Path("/tmp/annarchy_bench") / f"n{a.n}_r{a.rep}_{a.topology}"
    with common.timer() as tc:
        net.compile(silent=True, directory=str(d))

    with common.timer() as th:
        net.simulate(float(T))
        states = np.asarray(mon.get("v"), dtype=np.float64)

    states = states[:T]
    m = min(len(states), len(y))
    with common.timer() as tf:
        w = common.ridge_fit(states[wash:common.TRAIN_END], y[wash:common.TRAIN_END], reg=P["ridge"])

    pred = common.ridge_predict(states[:m], w)
    common.emit(framework="ANNarchy", group="lsm", n=a.n, rep=a.rep, task=a.task,
                topology=a.topology, t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y[:m], wash), native_readout=False,
                dt=1e-3, sim_steps=T, codegen_s=tc.dt,
                notes="idiomatic ANNarchy dt=1.0ms (1 step/sample), 4 threads; "
                      "C++ compile excluded from t_harvest; shared numpy ridge")

if __name__ == "__main__":
    main()
