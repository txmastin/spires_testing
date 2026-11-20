#!/usr/bin/env python
"""Norse LSM on a classification task."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("norse cls").parse_args()
    import torch
    from norse.torch.module.lif import LIFCell
    from norse.torch.functional.lif import LIFParameters
    torch.set_grad_enabled(False)

    def build(N, C, P, meta):
        W, _ = common.make_reservoir(N, seed=a.seed + a.rep, topology=a.topology,
                                     gain=1.0, spectral_radius=P["sr"])
        Win = common.make_input_weights(N, C, seed=a.seed + a.rep, gain=P["gain"])
        Wt = torch.tensor(W, dtype=torch.float32)
        Wit = torch.tensor(Win.T, dtype=torch.float32)
        p = LIFParameters(v_th=torch.tensor(float(P["vth"])),
                          tau_mem_inv=torch.tensor(float(P["leak"]) * 1000.0))
        cell = LIFCell(p=p)

        acc = {"spikes": np.zeros(N), "steps": 0}

        def harvest(x):
            xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
            st = None; spk = torch.zeros(1, N)
            out = np.empty((xt.shape[0], N))
            for t in range(xt.shape[0]):
                I = (xt[t] @ Wit + Wt @ spk[0]).unsqueeze(0)
                spk, st = cell(I, st)
                out[t] = st.v[0].numpy()
                acc["spikes"] += spk[0].numpy()
            acc["steps"] += xt.shape[0]
            return out

        def stats():
            return common.spike_stats(acc["spikes"].sum(), N, acc["steps"],
                                      active_neurons=(acc["spikes"] > 0).sum())
        return harvest, stats

    cls_common.run("norse", "lsm", a, build,
                   notes="shared topology; membrane potential pooled (mean+last)")

if __name__ == "__main__":
    main()
