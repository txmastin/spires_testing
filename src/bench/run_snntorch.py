#!/usr/bin/env python
"""snnTorch as an LSM.

snnTorch provides neuron models, not reservoirs, so the recurrent topology
comes from common.make_reservoir() -- identical to every other spiking
framework here. Protocol: exactly one LIF update per input sample, matching
SPIRES at dt=1.0. State fed to the readout is the membrane potential, which
is what SPIRES exposes via spires_read_reservoir_state.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("snnTorch").parse_args()
    import torch, snntorch as snn
    torch.set_grad_enabled(False)

    u, y, wash = common.load_data(a.task)
    T = len(u)
    P = common.resolve(a, "snnTorch")
    W, Win = common.make_reservoir(a.n, seed=a.seed + a.rep, topology=a.topology,
                                   gain=P["gain"], spectral_radius=P["sr"])
    Wt  = torch.tensor(W,   dtype=torch.float32)
    Wit = torch.tensor(Win, dtype=torch.float32)
    ut  = torch.tensor(u,   dtype=torch.float32)

    lif = snn.Leaky(beta=1.0 - P["leak"], threshold=P["vth"], init_hidden=False)
    mem = torch.zeros(a.n); spk = torch.zeros(a.n)
    states = np.empty((T, a.n), dtype=np.float64)
    spikes = np.zeros(a.n)

    with common.timer() as th:
        for t in range(T):
            I = Wit * ut[t] + Wt @ spk
            spk, mem = lif(I, mem)
            states[t] = mem.numpy()
            spikes += spk.numpy()

    with common.timer() as tf:
        w = common.ridge_fit(states[wash:common.TRAIN_END], y[wash:common.TRAIN_END], reg=P["ridge"])

    pred = common.ridge_predict(states, w)
    common.emit(framework="snnTorch", group="lsm", n=a.n, rep=a.rep, task=a.task,
                topology=a.topology, t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y, wash), native_readout=False,
                **common.spike_stats(spikes.sum(), a.n, T,
                                     active_neurons=(spikes > 0).sum()),
                notes="shared numpy ridge; membrane potential as state")

if __name__ == "__main__":
    main()
