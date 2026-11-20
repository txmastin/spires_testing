#!/usr/bin/env python
"""Norse as an LSM. Same protocol and topology as the snnTorch runner;
the only difference is which library integrates the LIF dynamics."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("norse").parse_args()
    import torch
    from norse.torch.module.lif import LIFCell
    from norse.torch.functional.lif import LIFParameters
    torch.set_grad_enabled(False)

    u, y, wash = common.load_data(a.task)
    T = len(u)
    P = common.resolve(a, "norse")
    W, Win = common.make_reservoir(a.n, seed=a.seed + a.rep, topology=a.topology,
                                   gain=P["gain"], spectral_radius=P["sr"])
    Wt  = torch.tensor(W,   dtype=torch.float32)
    Wit = torch.tensor(Win, dtype=torch.float32)
    ut  = torch.tensor(u,   dtype=torch.float32)

    p = LIFParameters(v_th=torch.tensor(float(P["vth"])),
                      tau_mem_inv=torch.tensor(float(P["leak"]) * 1000.0))
    cell = LIFCell(p=p)
    state = None
    spk = torch.zeros(1, a.n)
    states = np.empty((T, a.n), dtype=np.float64)
    spikes = np.zeros(a.n)

    with common.timer() as th:
        for t in range(T):
            I = (Wit * ut[t] + Wt @ spk[0]).unsqueeze(0)
            spk, state = cell(I, state)
            states[t] = state.v[0].numpy()
            spikes += spk[0].numpy()

    with common.timer() as tf:
        w = common.ridge_fit(states[wash:common.TRAIN_END], y[wash:common.TRAIN_END], reg=P["ridge"])

    pred = common.ridge_predict(states, w)
    common.emit(framework="norse", group="lsm", n=a.n, rep=a.rep, task=a.task,
                topology=a.topology, t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y, wash), native_readout=False,
                **common.spike_stats(spikes.sum(), a.n, T,
                                     active_neurons=(spikes > 0).sum()),
                notes="shared numpy ridge; membrane potential as state")

if __name__ == "__main__":
    main()
