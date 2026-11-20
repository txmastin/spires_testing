#!/usr/bin/env python
"""snnTorch LSM on a classification task."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("snnTorch cls").parse_args()
    import torch, snntorch as snn
    torch.set_grad_enabled(False)

    def build(N, C, P, meta):
        W, _ = common.make_reservoir(N, seed=a.seed + a.rep, topology=a.topology,
                                     gain=1.0, spectral_radius=P["sr"])
        Win = common.make_input_weights(N, C, seed=a.seed + a.rep, gain=P["gain"])
        Wt = torch.tensor(W, dtype=torch.float32)
        Wit = torch.tensor(Win.T, dtype=torch.float32)   # (C, N)
        lif = snn.Leaky(beta=1.0 - P["leak"], threshold=P["vth"], init_hidden=False)

        acc = {"spikes": np.zeros(N), "steps": 0}

        def harvest(x):
            xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
            mem = torch.zeros(N); spk = torch.zeros(N)
            out = np.empty((xt.shape[0], N))
            for t in range(xt.shape[0]):
                spk, mem = lif(xt[t] @ Wit + Wt @ spk, mem)
                out[t] = mem.numpy()
                acc["spikes"] += spk.numpy()
            acc["steps"] += xt.shape[0]
            return out

        def stats():
            return common.spike_stats(acc["spikes"].sum(), N, acc["steps"],
                                      active_neurons=(acc["spikes"] > 0).sum())
        return harvest, stats

    cls_common.run("snnTorch", "lsm", a, build,
                   notes="shared topology; membrane potential pooled (mean+last)")

if __name__ == "__main__":
    main()
