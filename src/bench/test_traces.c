/* Verify eligibility-trace dynamics against the analytic solution.
 *
 * A trace driven by a single unit impulse must satisfy x(t) = exp(-t/tau),
 * and each of the four traces must use its own time constant: tau_plus for
 * x_pre_e, tau_minus for x_post_e, tau_i for both inhibitory traces.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plasticity.h"

#define N 4

static struct plasticity_state make_state(double *params, double *bufs)
{
    struct plasticity_state ps;
    memset(&ps, 0, sizeof ps);
    ps.type = PLASTICITY_STDP;
    ps.params = params;
    ps.x_pre_e  = bufs + 0 * N;
    ps.x_post_e = bufs + 1 * N;
    ps.x_pre_i  = bufs + 2 * N;
    ps.x_post_i = bufs + 3 * N;
    return ps;
}

int main(void)
{
    double params[PL_N_PARAMS] = {0};
    params[PL_TAU_PLUS]  = 20.0;
    params[PL_TAU_MINUS] = 35.0;
    params[PL_TAU_I]     = 12.0;

    double bufs[4 * N];
    memset(bufs, 0, sizeof bufs);
    struct plasticity_state ps = make_state(params, bufs);

    const double dt = 1.0;
    double spikes[N] = { 1.0, 0.0, 0.0, 0.0 };

    /* single impulse into neuron 0 */
    plasticity_accumulate(&ps, spikes, N);
    printf("after one impulse: x_pre_e=%.4f x_post_e=%.4f x_pre_i=%.4f x_post_i=%.4f\n",
           ps.x_pre_e[0], ps.x_post_e[0], ps.x_pre_i[0], ps.x_post_i[0]);

    int fails = 0;
    if (fabs(ps.x_pre_e[0] - 1.0) > 1e-12) fails++;

    printf("\n%4s %12s %12s %12s %12s\n", "t", "x_pre_e", "exp(-t/20)", "x_post_e", "exp(-t/35)");
    double worst_pe = 0.0, worst_po = 0.0, worst_i = 0.0;
    for (int t = 1; t <= 60; t++) {
        plasticity_decay(&ps, N, dt);
        double want_pe = exp(-(double)t / params[PL_TAU_PLUS]);
        double want_po = exp(-(double)t / params[PL_TAU_MINUS]);
        double want_i  = exp(-(double)t / params[PL_TAU_I]);
        double e_pe = fabs(ps.x_pre_e[0]  - want_pe);
        double e_po = fabs(ps.x_post_e[0] - want_po);
        double e_i  = fabs(ps.x_pre_i[0]  - want_i);
        if (e_pe > worst_pe) worst_pe = e_pe;
        if (e_po > worst_po) worst_po = e_po;
        if (e_i  > worst_i)  worst_i  = e_i;
        if (t % 20 == 0)
            printf("%4d %12.6f %12.6f %12.6f %12.6f\n",
                   t, ps.x_pre_e[0], want_pe, ps.x_post_e[0], want_po);
    }
    printf("\nworst |error| over 60 steps: x_pre_e %.3e  x_post_e %.3e  x_pre_i %.3e\n",
           worst_pe, worst_po, worst_i);
    if (worst_pe > 1e-12 || worst_po > 1e-12 || worst_i > 1e-12) fails++;

    /* the two inhibitory traces share tau, so they must stay equal */
    if (fabs(ps.x_pre_i[0] - ps.x_post_i[0]) > 1e-15) fails++;
    printf("inhibitory traces share tau_i: %s\n",
           fabs(ps.x_pre_i[0] - ps.x_post_i[0]) <= 1e-15 ? "yes" : "NO");

    /* neurons that never spiked must stay at zero */
    int quiet_ok = 1;
    for (int k = 1; k < N; k++)
        if (ps.x_pre_e[k] != 0.0 || ps.x_post_e[k] != 0.0) quiet_ok = 0;
    printf("silent neurons stay at zero: %s\n", quiet_ok ? "yes" : "NO");
    if (!quiet_ok) fails++;

    /* PLASTICITY_NONE must leave everything untouched */
    double bufs2[4 * N];
    memset(bufs2, 0, sizeof bufs2);
    struct plasticity_state off = make_state(params, bufs2);
    off.type = PLASTICITY_NONE;
    plasticity_accumulate(&off, spikes, N);
    plasticity_decay(&off, N, dt);
    int untouched = 1;
    for (int k = 0; k < 4 * N; k++) if (bufs2[k] != 0.0) untouched = 0;
    printf("PLASTICITY_NONE is inert: %s\n", untouched ? "yes" : "NO");
    if (!untouched) fails++;

    printf("\n%s\n", fails ? "FAILURES PRESENT" : "all trace checks pass");
    return fails ? 1 : 0;
}
