#!/usr/bin/env python
"""reservoirpy ESN on a classification task.

The ESN libraries have no notion of spikes; SHD's binned spike counts are
fed as a real-valued multi-channel input, which is the natural way to give
a rate-coded reservoir event data. That encoding gap is the point of the
comparison, not a flaw in it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common, cls_common

def main():
    a = common.argparser("reservoirpy cls").parse_args()
    from reservoirpy.nodes import Reservoir

    def build(N, C, P, meta):
        if common.weight_source() == "spires":
            # Equalized mode: run reservoirpy on the network SPIRES actually
            # builds, so the tanh and spiking arms differ only in the neuron.
            # sr/rc_connectivity are omitted because the supplied matrix
            # already carries its own scaling and density.
            W, Win = common.spires_weights(N, C, seed=a.seed + a.rep, sr=P["sr"])
            res = Reservoir(units=N, lr=P["leak"], W=W, Win=Win * P["gain"],
                            seed=a.seed + a.rep)
        else:
            res = Reservoir(units=N, lr=P["leak"], sr=P["sr"],
                            input_scaling=P["gain"], rc_connectivity=0.1,
                            seed=a.seed + a.rep)

        # reservoirpy allocates node state lazily, so reset() raises until
        # the node has been initialized. Initialize eagerly (outside the
        # timed region) so every sample can start from a zeroed state.
        res.initialize(np.zeros((1, C), dtype=np.float64))

        def harvest(x):
            res.reset()
            return res.run(np.ascontiguousarray(x, dtype=np.float64))
        return harvest

    cls_common.run("reservoirpy", "esn", a, build,
                   notes="binned spike counts as real-valued input; states pooled (mean+last)")

if __name__ == "__main__":
    main()
