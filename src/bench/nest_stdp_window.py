#!/usr/bin/env python
"""Measure the inhibitory STDP window from NEST's vogels_sprekeler_synapse.

Same pairing protocol as test_stdp_window.c: one pre/post pair per trial at a
controlled emission lag, weight change recorded. Run at two delays to separate
the rule's shape from the effect of transmission delay, which is the mechanism
NEST uses to make exact coincidence impossible.

Parameters match the SPIRES test: tau=20, eta=0.01, alpha=2*tau*rho_0=0.8,
Wmax=1, w0=0.5. NEST carries inhibitory weights as positive magnitudes, so its
delta_w is directly comparable to the SPIRES d|w|.
"""
import sys
import numpy as np
import nest

TAU, ETA, ALPHA, WMAX, W0 = 20.0, 0.01, 0.8, 1.0, 0.5
RES = 1.0            # matches the SPIRES dt
T_PRE = 100.0
DT_MAX = 60


FLUSH = 600.0        # second pre spike, long after the pair


def trial(dt, delay):
    """One pre/post pair at emission lag dt; returns the weight change.

    NEST updates a plastic synapse only when the presynaptic neuron fires, so
    a post spike later than the last pre spike stays pending and never reaches
    the weight. A second pre spike far in the future flushes it. It contributes
    one extra constant depression and negligible potentiation (the post trace
    has decayed over FLUSH/tau time constants), so every trial carries the same
    offset and the window shape is unaffected.
    """
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = RES

    t_post = T_PRE + dt
    # spike_generator -> parrot pairs let us set emission times exactly; the
    # generator->parrot delay is identical on both paths, so it cancels.
    sg_pre = nest.Create("spike_generator",
                         params={"spike_times": [T_PRE, T_PRE + FLUSH]})
    sg_post = nest.Create("spike_generator", params={"spike_times": [t_post]})
    pre = nest.Create("parrot_neuron")
    # A parrot repeats every spike it receives, so the plastic connection would
    # make the target fire an extra time at a fixed lag behind the source. Use
    # an integrate-and-fire target driven hard by its own generator instead:
    # the plastic weight is far too small to make it spike.
    post = nest.Create("iaf_psc_delta")

    nest.Connect(sg_pre, pre, syn_spec={"delay": RES})
    nest.Connect(sg_post, post, syn_spec={"delay": RES, "weight": 1e4})
    nest.Connect(pre, post, syn_spec={
        "synapse_model": "vogels_sprekeler_synapse",
        "weight": W0, "Wmax": WMAX, "tau": TAU, "eta": ETA,
        "alpha": ALPHA, "delay": delay,
    })

    rec_pre = nest.Create("spike_recorder")
    rec_post = nest.Create("spike_recorder")
    nest.Connect(pre, rec_pre)
    nest.Connect(post, rec_post)

    nest.Simulate(T_PRE + FLUSH + abs(dt) + 400.0)

    tp = nest.GetStatus(rec_pre, "events")[0]["times"]
    tq = nest.GetStatus(rec_post, "events")[0]["times"]
    # measured lag between the spikes actually emitted, not the requested ones
    lag = (tq[0] - tp[0]) if len(tq) and len(tp) else float("nan")
    return nest.GetConnections(pre, post).get("weight") - W0, lag


def main(out):
    delays = [RES, 5.0]
    dts = np.arange(-DT_MAX, DT_MAX + 1, dtype=float)
    cols, lags = {}, {}
    for d in delays:
        res = [trial(float(dt), d) for dt in dts]
        cols[d] = np.array([o[0] for o in res])
        lags[d] = np.array([o[1] for o in res])
    off = np.unique(np.round(lags[delays[0]] - dts, 6))
    print(f"emitted lag minus requested dt: {off}  (constant => timing is clean)")

    hdr = "dt," + ",".join(f"dw_delay{d:g}" for d in delays)
    np.savetxt(out, np.column_stack([dts] + [cols[d] for d in delays]),
               delimiter=",", header=hdr, comments="", fmt="%.12g")
    print(f"wrote {out}\n")

    print(f"{'dt':>6}" + "".join(f"{'delay=' + format(d, 'g'):>16}" for d in delays))
    for k, dt in enumerate(dts):
        if dt % 10 == 0:
            print(f"{dt:6.0f}" + "".join(f"{cols[d][k]:16.3e}" for d in delays))

    print()
    for d in delays:
        y = cols[d]
        peak = dts[int(np.argmax(y))]
        zero = y[dts == 0.0][0]
        print(f"delay={d:g}: peak at dt={peak:+.0f}, value at dt=0 is {zero:.3e}, "
              f"max {y.max():.3e}, min {y.min():.3e}")
        # symmetry about the peak, which is where the delay shifts it to
        sym = max(abs(y[dts == peak + s][0] - y[dts == peak - s][0])
                  for s in range(1, 20) if (peak + s) <= DT_MAX and (peak - s) >= -DT_MAX)
        print(f"          worst asymmetry about that peak: {sym:.3e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/nest_stdp_window.csv")
