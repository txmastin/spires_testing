/* First-contact checks on the STDP rules: weights move when plasticity is
 * enabled, stay put when it is not, never flip sign, and stay inside bounds.
 * The quantitative check against the experimental window comes separately.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <spires.h>

#define N 200

static double params[10] = {
    /* excitatory (Gutig) */ 20.0, 20.0, 0.01, 1.05, 1.0, 1.0,
    /* inhibitory (Vogels) */ 20.0, 0.01, 0.02,
    /* shared */ 1.0
};

static unsigned SEED = 12;

static spires_reservoir *make(spires_plasticity_type pl, spires_ei_mode ei)
{
    static double np[] = { 0.0, 1.0, 0.1, 0.0 };
    spires_reservoir_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = N; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = 0.9; cfg.ei_ratio = 0.8; cfg.input_strength = 3.0;
    cfg.connectivity = 0.1; cfg.dt = 1.0;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    cfg.ei_mode = ei;
    cfg.plasticity_type = pl;
    cfg.plasticity_params = params;
    cfg.seed = SEED;
    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) return NULL;
    return r;
}

static void run(spires_reservoir *r, int steps)
{
    for (int t = 0; t < steps; t++) {
        double u = 0.5 * (1.0 + sin(0.3 * t));
        spires_step(r, &u);
    }
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            SEED = (unsigned)atoi(argv[++i]);

    int fails = 0;
    double *W0 = malloc(N * N * sizeof(double));
    double *W1 = malloc(N * N * sizeof(double));

    /* --- plasticity OFF: weights must be frozen --- */
    spires_reservoir *r = make(SPIRES_PLASTICITY_NONE, SPIRES_EI_PER_NEURON);
    spires_read_weights(r, W0);
    run(r, 400);
    spires_read_weights(r, W1);
    int frozen = memcmp(W0, W1, N * N * sizeof(double)) == 0;
    printf("plasticity NONE leaves weights unchanged: %s\n", frozen ? "yes" : "NO");
    if (!frozen) fails++;
    spires_reservoir_destroy(r);

    /* --- plasticity ON under Dale's law --- */
    r = make(SPIRES_PLASTICITY_STDP, SPIRES_EI_PER_NEURON);
    spires_read_weights(r, W0);
    run(r, 400);
    spires_read_weights(r, W1);

    size_t changed = 0, nz = 0, flipped = 0, out_of_bounds = 0;
    double max_abs = 0.0;
    for (size_t k = 0; k < (size_t)N * N; k++) {
        if (W0[k] == 0.0 && W1[k] == 0.0) continue;
        nz++;
        if (W0[k] != W1[k]) changed++;
        if (W0[k] > 0.0 && W1[k] < 0.0) flipped++;
        if (W0[k] < 0.0 && W1[k] > 0.0) flipped++;
        if (fabs(W1[k]) > params[9] + 1e-12) out_of_bounds++;
        if (fabs(W1[k]) > max_abs) max_abs = fabs(W1[k]);
    }
    printf("plasticity STDP changed %zu of %zu weights (%.1f%%)\n",
           changed, nz, 100.0 * (double)changed / (double)nz);
    printf("sign flips: %zu   out of [0, w_max]: %zu   max |w| %.4f (w_max %.1f)\n",
           flipped, out_of_bounds, max_abs, params[9]);
    if (changed == 0) { fails++; printf("  !! no weights moved\n"); }
    if (flipped) { fails++; printf("  !! sign was not preserved\n"); }
    if (out_of_bounds) { fails++; printf("  !! bounds violated\n"); }
    spires_reservoir_destroy(r);

    /* --- PER_SYNAPSE has no presynaptic identity to select a rule with, so
     *     the combination must be refused rather than dispatched on the
     *     weight sign, which is lost as soon as a weight reaches zero. --- */
    r = make(SPIRES_PLASTICITY_STDP, SPIRES_EI_PER_SYNAPSE);
    printf("STDP + PER_SYNAPSE is refused: %s\n", r ? "NO" : "yes");
    if (r) { fails++; spires_reservoir_destroy(r); }

    /* --- unsupported combination must be refused, not silently ignored --- */
    {
        static double np[] = { 0.0, 1.0, 0.1, 0.0 };
        static double sp[] = { 2.0, 30.0 };
        spires_reservoir_config cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.num_neurons = 50; cfg.num_inputs = 1; cfg.num_outputs = 1;
        cfg.spectral_radius = 0.9; cfg.ei_ratio = 0.8; cfg.input_strength = 1.0;
        cfg.connectivity = 0.1; cfg.dt = 1.0;
        cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np;
        cfg.synapse_type = SPIRES_SYNAPSE_PSC_HETEROGENEOUS; cfg.synapse_params = sp;
        cfg.ei_mode = SPIRES_EI_PER_NEURON;   /* isolate the synapse-type check */
        cfg.plasticity_type = SPIRES_PLASTICITY_STDP; cfg.plasticity_params = params;
        spires_reservoir *bad = NULL;
        spires_status st = spires_reservoir_create(&cfg, &bad);
        printf("psc_heterogeneous + STDP is refused: %s\n",
               st == SPIRES_ERR_INVALID_ARG ? "yes" : "NO");
        if (st != SPIRES_ERR_INVALID_ARG) fails++;
        if (bad) spires_reservoir_destroy(bad);
    }

    free(W0); free(W1);
    printf("\n%s\n", fails ? "FAILURES PRESENT" : "all smoke checks pass");
    return fails ? 1 : 0;
}
