#!/usr/bin/env python
"""Rockpool LSM on a classification task (vectorised per sample)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("rockpool cls").parse_args()
    import torch
    from rockpool.nn.modules import LIFTorch
    torch.set_grad_enabled(False)

    def build(N, C, P, meta):
        W, _ = common.make_reservoir(N, seed=a.seed + a.rep, topology=a.topology,
                                     gain=1.0, spectral_radius=P["sr"])
        Win = common.make_input_weights(N, C, seed=a.seed + a.rep, gain=P["gain"])
        Wit = torch.tensor(Win.T, dtype=torch.float32)
        mod = LIFTorch(shape=(N, N), has_rec=True,
                       w_rec=torch.tensor(W * P["wscale"], dtype=torch.float32),
                       leak_mode="decays", alpha=P["alpha"], beta=P["beta"],
                       threshold=P["vth"], dt=1.0)

        acc = {"spikes": np.zeros(N), "steps": 0}

        def harvest(x):
            xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
            I = (xt @ Wit).unsqueeze(0)
            mod.reset_state()
            out, _, rec = mod(I, record=True)
            acc["spikes"] += out[0].numpy().sum(axis=0)
            acc["steps"] += xt.shape[0]
            return rec["vmem"][0].numpy()

        def stats():
            return common.spike_stats(acc["spikes"].sum(), N, acc["steps"],
                                      active_neurons=(acc["spikes"] > 0).sum())
        return harvest, stats

    cls_common.run("rockpool", "lsm", a, build,
                   notes="vectorised evolve per sample; membrane pooled (mean+last)")

if __name__ == "__main__":
    main()
