#!/usr/bin/env python
"""Rockpool as an LSM.

Unlike snnTorch/norse, Rockpool evolves a whole sequence in a single call
rather than a Python-level timestep loop, so its harvest is one vectorised
evolve() over all T samples. Same shared topology and the same one-step-
per-sample protocol; the readout is the shared numpy ridge.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("rockpool").parse_args()
    import torch
    from rockpool.nn.modules import LIFTorch
    torch.set_grad_enabled(False)

    u, y, wash = common.load_data(a.task)
    T = len(u)
    P = common.resolve(a, "rockpool")
    W, Win = common.make_reservoir(a.n, seed=a.seed + a.rep, topology=a.topology,
                                   gain=P["gain"], spectral_radius=P["sr"])

    # input current per neuron per timestep: (batch=1, time=T, n)
    I = torch.tensor((u[:, None] * Win[None, :]), dtype=torch.float32).unsqueeze(0)

    # Rockpool filters input through a synapse rather than injecting current
    # directly, so alpha/beta are tuned for this model rather than copied from
    # the SPIRES discrete-LIF settings: beta=0 zeroes the synapse entirely and
    # beta>=0.9 diverges. alpha=0.8, beta=0.5 is the best NRMSE on NARMA-10.
    mod = LIFTorch(shape=(a.n, a.n), has_rec=True,
                   w_rec=torch.tensor(W * P["wscale"], dtype=torch.float32),
                   leak_mode="decays",
                   alpha=P["alpha"], beta=P["beta"],
                   threshold=P["vth"], dt=1.0)

    with common.timer() as th:
        out, new_state, rec = mod(I, record=True)
        vmem = rec["vmem"][0].numpy().astype(np.float64)
    sp = out[0].numpy()

    with common.timer() as tf:
        w = common.ridge_fit(vmem[wash:common.TRAIN_END], y[wash:common.TRAIN_END], reg=P["ridge"])

    pred = common.ridge_predict(vmem, w)
    common.emit(framework="rockpool", group="lsm", n=a.n, rep=a.rep, task=a.task,
                topology=a.topology, t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y, wash), native_readout=False,
                **common.spike_stats(sp.sum(), a.n, T,
                                     active_neurons=(sp.sum(0) > 0).sum()),
                dt=1.0, sim_steps=T,
                notes="vectorised evolve (no python timestep loop); "
                      "shared numpy ridge; membrane potential as state")

if __name__ == "__main__":
    main()
