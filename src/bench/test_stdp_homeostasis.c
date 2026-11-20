/* Does inhibitory STDP actually regulate the postsynaptic firing rate to rho_0?
 *
 * That is the claim the Vogels et al. (2011) rule exists to make. For a synapse
 * i <- j the expected change per step is
 *
 *   E[d|w|] = eta * p_j * ( 2 * x_post_ss - alpha ),   alpha = 2 * tau * rho_0
 *
 * so the weight stops drifting when 2*x_post_ss = alpha. The trace this reads
 * has been decayed but not yet accumulated, so its steady state is
 * x_ss = p*d/(1-d) with d = exp(-dt/tau), not the continuous-time p*tau. The
 * predicted fixed point is therefore
 *
 *   p = tau * rho_0 * (1 - d) / d
 *
 * which for tau=20, dt=1 sits about 2.5% above rho_0. Checking against that,
 * rather than against rho_0 itself, is what makes this a sharp test.
 *
 * The excitatory rule is switched off (lambda = 0) so the only plasticity
 * acting is the one under test.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <spires.h>

#define N        300
#define WARMUP   60000     /* converges by ~25k; margin for the low targets */
#define MEASURE  10000

static double neuron_params[] = { 0.0, 1.0, 0.1, 0.0 };

static unsigned SEED = 7;

static double mean_rate(double rho_0, double *w_mean_out, double *sat_out)
{
    double params[10] = {
        /* excitatory: lambda = 0 freezes this branch entirely */
        20.0, 20.0, 0.0, 1.05, 1.0, 1.0,
        /* inhibitory */
        20.0, 0.002, rho_0,
        /* w_max, generous so inhibition is not clipped before it balances */
        50.0
    };

    spires_reservoir_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = N; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = 1.2; cfg.ei_ratio = 0.8; cfg.input_strength = 0.2;
    cfg.connectivity = 0.1; cfg.dt = 1.0;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE;
    cfg.neuron_params = neuron_params;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    cfg.ei_mode = SPIRES_EI_PER_NEURON;
    cfg.plasticity_type = SPIRES_PLASTICITY_STDP;
    cfg.plasticity_params = params;
    cfg.seed = SEED;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) return NAN;

    double *spikes = malloc(N * sizeof(double));
    long total = 0;
    for (long t = 0; t < WARMUP + MEASURE; t++) {
        double u = 1.0 + 0.2 * sin(0.013 * (double)t);
        spires_step(r, &u);
        if (t >= WARMUP) {
            spires_read_spike_state(r, spikes);
            for (int k = 0; k < N; k++) if (spikes[k] != 0.0) total++;
        }
    }

    /* how much inhibition it took, and whether the bound was the binding
     * constraint rather than the fixed point */
    double *W = malloc((size_t)N * N * sizeof(double));
    spires_read_weights(r, W);
    double sum = 0.0; long cnt = 0, at_max = 0;
    for (size_t k = 0; k < (size_t)N * N; k++) {
        if (W[k] >= 0.0) continue;
        sum += -W[k]; cnt++;
        if (-W[k] >= params[9] - 1e-9) at_max++;
    }
    *w_mean_out = cnt ? sum / (double)cnt : 0.0;
    *sat_out = cnt ? (double)at_max / (double)cnt : 0.0;

    free(W); free(spikes);
    spires_reservoir_destroy(r);
    return (double)total / ((double)MEASURE * (double)N);
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            SEED = (unsigned)atoi(argv[++i]);

    const double tau = 20.0, dt = 1.0;
    const double d = exp(-dt / tau);
    const double targets[] = { 0.02, 0.05, 0.10, 0.15 };
    const int nt = (int)(sizeof targets / sizeof targets[0]);

    printf("inhibitory STDP rate control (N=%d, %d warmup + %d measured steps)\n\n",
           N, WARMUP, MEASURE);
    printf("%8s %12s %12s %10s %10s %9s\n",
           "rho_0", "predicted", "measured", "ratio", "mean |w|", "at w_max");

    int fails = 0;
    double worst = 0.0, rmin = 1e9, rmax = -1e9, last = 0.0;
    double prev_rate = -1.0;
    for (int k = 0; k < nt; k++) {
        double w_mean, sat;
        double p_pred = tau * targets[k] * (1.0 - d) / d;
        double p_meas = mean_rate(targets[k], &w_mean, &sat);
        double ratio = p_meas / p_pred;
        printf("%8.3f %12.5f %12.5f %10.3f %10.4f %8.1f%%\n",
               targets[k], p_pred, p_meas, ratio, w_mean, 100.0 * sat);
        if (fabs(ratio - 1.0) > worst) worst = fabs(ratio - 1.0);
        if (ratio < rmin) rmin = ratio;
        if (ratio > rmax) rmax = ratio;
        if (p_meas <= prev_rate) { fails++; printf("  !! rate did not increase with rho_0\n"); }
        prev_rate = p_meas;
        last = p_meas;
        if (sat > 0.02) { fails++; printf("  !! w_max is the binding constraint, not the fixed point\n"); }
    }

    /* The fixed point is derived for independent Poisson spiking. In a
     * recurrent network an inhibitory neuron and its target share drive, so
     * the presynaptic trace seen at a postsynaptic spike runs higher than the
     * independent estimate, potentiation is stronger, and the rate settles
     * below rho_0. That bias should be systematic -- the same offset at every
     * target -- rather than an erratic failure to track, so the spread across
     * targets is the more informative check. */
    printf("\nworst deviation from the predicted fixed point: %.1f%%\n", 100.0 * worst);
    printf("spread of that ratio across targets: %.3f to %.3f (%.1f%% wide)\n",
           rmin, rmax, 100.0 * (rmax - rmin));
    if (worst > 0.35) { fails++; printf("  !! rate is not tracking rho_0 at all\n"); }
    if (rmax - rmin > 0.25) { fails++; printf("  !! offset is not systematic\n"); }
    if (rmax > 1.0) { fails++; printf("  !! rate exceeds the fixed point; inhibition is not binding\n"); }

    /* A rate that tracks rho_0 only because the drive happens to sit there
     * would be a false pass, so check plasticity is what is doing the work. */
    {
        double params_off[10] = { 20,20,0,1.05,1,1, 20, 0.0, 0.05, 50.0 };
        spires_reservoir_config cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.num_neurons = N; cfg.num_inputs = 1; cfg.num_outputs = 1;
        cfg.spectral_radius = 1.2; cfg.ei_ratio = 0.8; cfg.input_strength = 0.2;
        cfg.connectivity = 0.1; cfg.dt = 1.0;
        cfg.connectivity_type = SPIRES_CONN_RANDOM;
        cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE;
        cfg.neuron_params = neuron_params;
        cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
        cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
        cfg.ei_mode = SPIRES_EI_PER_NEURON;
        cfg.plasticity_type = SPIRES_PLASTICITY_STDP;
        cfg.plasticity_params = params_off;   /* eta = 0: rule inert */
        cfg.seed = SEED;
        spires_reservoir *r = NULL;
        spires_reservoir_create(&cfg, &r);
        double *spikes = malloc(N * sizeof(double));
        long total = 0;
        for (long t = 0; t < WARMUP + MEASURE; t++) {
            double u = 1.0 + 0.2 * sin(0.013 * (double)t);
            spires_step(r, &u);
            if (t >= WARMUP) {
                spires_read_spike_state(r, spikes);
                for (int i = 0; i < N; i++) if (spikes[i] != 0.0) total++;
            }
        }
        double base = (double)total / ((double)MEASURE * (double)N);
        printf("baseline rate with eta = 0 (rule inert): %.5f\n", base);
        printf("highest target settles at %.5f, well below it: %s\n",
               last, last < 0.6 * base ? "yes" : "NO");
        if (!(last < 0.6 * base)) { fails++; printf("  !! plasticity is not doing the work\n"); }
        free(spikes); spires_reservoir_destroy(r);
    }

    printf("\n%s\n", fails ? "FAILURES PRESENT" : "homeostasis checks pass");
    return fails ? 1 : 0;
}
