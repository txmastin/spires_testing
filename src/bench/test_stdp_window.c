/* Reproduce the experimental STDP windows from a two-neuron pairing protocol.
 *
 * One synapse, source neuron 0 -> target neuron 1. A single pre/post spike
 * pair is presented at a controlled lag dt = t_post - t_pre, and the resulting
 * weight change is recorded. Sweeping dt traces out the STDP window, which is
 * the quantity the experimental literature actually reports:
 *
 *   excitatory  Bi & Poo 1998, J Neurosci -- asymmetric and causal: pre-before-
 *               post potentiates, post-before-pre depresses, with separate time
 *               constants (~17 ms and ~34 ms).
 *   inhibitory  Vogels et al. 2011; D'amour & Froemke 2015, Neuron -- symmetric:
 *               both orders potentiate inhibition, with a constant depression
 *               term that makes the rule homeostatic.
 *
 * This drives plasticity_* and synapse_simple_apply_stdp in the same order
 * step_reservoir does, but with spike times set by hand.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plasticity.h"
#include "simple.h"

#define N     2      /* 0 = source (presynaptic), 1 = target (postsynaptic) */
#define T0    100    /* time of the presynaptic spike */
#define TMAX  260
#define DT_MAX 60

/* Bi & Poo windows for the excitatory rule; Vogels/D'amour for the inhibitory. */
static double params[PL_N_PARAMS] = {
    [PL_TAU_PLUS] = 17.0, [PL_TAU_MINUS] = 34.0,
    [PL_LAMBDA]   = 0.01, [PL_ALPHA_E]   = 1.05,
    [PL_MU_PLUS]  = 1.0,  [PL_MU_MINUS]  = 1.0,
    [PL_TAU_I]    = 20.0, [PL_ETA]       = 0.01, [PL_RHO_0] = 0.02,
    [PL_W_MAX]    = 1.0
};

/* Present one pre/post pair at lag dt and return the change in the weight. */
static double pairing(double w0, signed char sign_src, int dt)
{
    double dense[N * N];
    memset(dense, 0, sizeof dense);
    dense[1 * N + 0] = w0;                  /* W[target][source] */
    struct simple_synapse_data *d = synapse_simple_build_dense(dense, N);
    if (!d) return NAN;

    double bufs[4 * N];
    memset(bufs, 0, sizeof bufs);
    signed char sign[N] = { sign_src, 1 };

    struct plasticity_state ps;
    memset(&ps, 0, sizeof ps);
    ps.type = PLASTICITY_STDP;
    ps.params = params;
    ps.x_pre_e  = bufs + 0 * N;  ps.x_post_e = bufs + 1 * N;
    ps.x_pre_i  = bufs + 2 * N;  ps.x_post_i = bufs + 3 * N;
    ps.neuron_sign = sign;

    for (int t = 0; t < TMAX; t++) {
        double spikes[N] = { 0.0, 0.0 };
        if (t == T0)      spikes[0] = 1.0;
        if (t == T0 + dt) spikes[1] = 1.0;
        plasticity_decay(&ps, N, 1.0);
        synapse_simple_apply_stdp(d, spikes, &ps);
        plasticity_accumulate(&ps, spikes, N);
    }

    synapse_simple_to_dense(d, dense);
    double w = dense[1 * N + 0];
    synapse_simple_free(d);
    return w - w0;
}

/* Two-point estimate of the decay constant of A*exp(-|dt|/tau). */
static double est_tau(double d1, int t1, double d2, int t2)
{
    if (d1 == 0.0 || d2 == 0.0) return NAN;
    return (double)(t2 - t1) / log(fabs(d1) / fabs(d2));
}

int main(void)
{
    int fails = 0;
    const double w_exc = 0.5, w_inh = -0.5;

    FILE *csv = fopen("results/stdp_window.csv", "w");
    if (csv) fprintf(csv, "dt,dw_exc,dmag_inh\n");

    printf("pairing protocol: one pre/post pair per trial, w0 = %+.2f / %+.2f\n",
           w_exc, w_inh);
    printf("dt = t_post - t_pre;  dt > 0 is pre-before-post\n\n");
    printf("%6s %14s %16s\n", "dt", "exc  dw", "inh  d|w|");

    double dw_e[2 * DT_MAX + 1], dm_i[2 * DT_MAX + 1];
    for (int k = 0; k <= 2 * DT_MAX; k++) {
        int dt = k - DT_MAX;
        dw_e[k] = pairing(w_exc, +1, dt);
        dm_i[k] = -pairing(w_inh, -1, dt);   /* more negative w = stronger inhibition */
        if (csv) fprintf(csv, "%d,%.12g,%.12g\n", dt, dw_e[k], dm_i[k]);
        if (dt % 10 == 0)
            printf("%6d %14.3e %16.3e\n", dt, dw_e[k], dm_i[k]);
    }
    if (csv) fclose(csv);

#define IDX(dt) ((dt) + DT_MAX)

    /* --- excitatory: causal and asymmetric --- */
    int bad_sign = 0;
    for (int dt = 1; dt <= DT_MAX; dt++) {
        if (!(dw_e[IDX(dt)]  > 0.0)) bad_sign++;   /* pre before post: potentiate */
        if (!(dw_e[IDX(-dt)] < 0.0)) bad_sign++;   /* post before pre: depress    */
    }
    printf("\nexcitatory window is causal (dt>0 potentiates, dt<0 depresses): %s\n",
           bad_sign ? "NO" : "yes");
    if (bad_sign) fails++;

    int mono = 1;
    for (int dt = 1; dt < DT_MAX; dt++) {
        if (fabs(dw_e[IDX(dt + 1)])  > fabs(dw_e[IDX(dt)]))  mono = 0;
        if (fabs(dw_e[IDX(-dt - 1)]) > fabs(dw_e[IDX(-dt)])) mono = 0;
    }
    printf("excitatory window decays monotonically with |dt|: %s\n", mono ? "yes" : "NO");
    if (!mono) fails++;

    double tau_p = est_tau(dw_e[IDX(10)], 10, dw_e[IDX(40)], 40);
    double tau_m = est_tau(dw_e[IDX(-10)], 10, dw_e[IDX(-40)], 40);
    printf("recovered tau_plus  %.4f (set %.1f)\n", tau_p, params[PL_TAU_PLUS]);
    printf("recovered tau_minus %.4f (set %.1f)\n", tau_m, params[PL_TAU_MINUS]);
    if (fabs(tau_p - params[PL_TAU_PLUS])  > 1e-9) { fails++; printf("  !! tau_plus off\n"); }
    if (fabs(tau_m - params[PL_TAU_MINUS]) > 1e-9) { fails++; printf("  !! tau_minus off\n"); }

    /* The two halves must differ, or the window is not asymmetric at all. */
    int asym = fabs(tau_p - tau_m) > 1e-6;
    printf("the two halves have different time constants: %s\n", asym ? "yes" : "NO");
    if (!asym) fails++;

    /* --- inhibitory: symmetric in |dt| --- */
    double worst_sym = 0.0;
    for (int dt = 1; dt <= DT_MAX; dt++) {
        double e = fabs(dm_i[IDX(dt)] - dm_i[IDX(-dt)]);
        if (e > worst_sym) worst_sym = e;
    }
    printf("\ninhibitory window is symmetric: worst |dw(+dt) - dw(-dt)| = %.3e\n", worst_sym);
    if (worst_sym > 1e-15) { fails++; printf("  !! not symmetric\n"); }

    /* Near-coincident pairing strengthens inhibition; far-apart pairing leaves
     * only the constant depression term, so the window crosses zero. */
    double alpha_i = 2.0 * params[PL_TAU_I] * params[PL_RHO_0];
    double want_cross = -params[PL_TAU_I] * log(alpha_i);
    int cross = -1;
    for (int dt = 1; dt <= DT_MAX; dt++)
        if (dm_i[IDX(dt)] <= 0.0) { cross = dt; break; }
    printf("inhibitory alpha = 2*tau*rho_0 = %.3f, predicted zero crossing at dt = %.2f\n",
           alpha_i, want_cross);
    printf("observed crossing between dt = %d and %d\n", cross - 1, cross);
    if (cross < 1 || want_cross < cross - 1 || want_cross > cross) {
        fails++; printf("  !! crossing not where the rule predicts\n");
    }

    double far = dm_i[IDX(DT_MAX)];
    printf("far-apart pairing leaves net depression: %s (d|w| = %.3e)\n",
           far < 0.0 ? "yes" : "NO", far);
    if (!(far < 0.0)) fails++;

    /* --- multiplicative weight dependence (mu = 1) --- */
    double p_low  = pairing(0.1, +1, 10);
    double p_high = pairing(0.9, +1, 10);
    double d_low  = pairing(0.1, +1, -10);
    double d_high = pairing(0.9, +1, -10);
    printf("\npotentiation at w=0.1: %.3e   at w=0.9: %.3e  (shrinks near w_max: %s)\n",
           p_low, p_high, p_high < p_low ? "yes" : "NO");
    printf("depression   at w=0.1: %.3e   at w=0.9: %.3e  (shrinks near zero:  %s)\n",
           d_low, d_high, fabs(d_low) < fabs(d_high) ? "yes" : "NO");
    if (!(p_high < p_low)) fails++;
    if (!(fabs(d_low) < fabs(d_high))) fails++;

    printf("\n%s\n", fails ? "FAILURES PRESENT" : "all window checks pass");
    return fails ? 1 : 0;
}
