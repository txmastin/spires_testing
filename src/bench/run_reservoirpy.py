#!/usr/bin/env python
"""reservoirpy runner: native Reservoir + native Ridge readout."""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def main():
    a = common.argparser("reservoirpy").parse_args()
    import reservoirpy
    from reservoirpy.nodes import Reservoir, Ridge

    u, y, wash = common.load_data(a.task)
    U = u.reshape(-1, 1); Y = y.reshape(-1, 1)

    P = common.resolve(a, "reservoirpy")
    res = Reservoir(units=a.n, lr=P["leak"], sr=P["sr"],
                    input_scaling=P["gain"], rc_connectivity=0.1,
                    seed=a.seed + a.rep)
    with common.timer() as th:
        states = res.run(U)
    ro = Ridge(ridge=P["ridge"])
    with common.timer() as tf:
        ro.fit(states[wash:common.TRAIN_END], Y[wash:common.TRAIN_END])

    pred = ro.run(states)
    common.emit(framework="reservoirpy", group="esn", n=a.n, rep=a.rep, task=a.task,
                t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(pred, y, wash),
                native_readout=True)

if __name__ == "__main__":
    main()
