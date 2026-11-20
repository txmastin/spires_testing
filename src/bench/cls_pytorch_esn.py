#!/usr/bin/env python
"""pytorch-esn on a classification task.

Uses the ESN reservoir only: pytorch-esn's readout is a single-output
regression head trained by accumulated normal equations, which does not fit
the multi-class pooled-feature setup. The shared ridge is used instead, so
the fit timing is comparable across frameworks rather than reflecting a
different algorithm.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("pytorch-esn cls").parse_args()
    import torch
    from torchesn.nn import ESN
    torch.set_grad_enabled(False)
    torch.manual_seed(a.seed + a.rep)

    def build(N, C, P, meta):
        m = ESN(C, N, 1, spectral_radius=P["sr"], density=0.1,
                leaking_rate=P["leak"], w_ih_scale=P["gain"],
                output_steps="all", readout_training="cholesky")

        def harvest(x):
            xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32).unsqueeze(1)
            out, _ = m.reservoir(xt, None)
            return out[:, 0, :].numpy()
        return harvest

    cls_common.run("pytorch-esn", "esn", a, build,
                   notes="reservoir only; shared multi-class ridge readout")

if __name__ == "__main__":
    main()
