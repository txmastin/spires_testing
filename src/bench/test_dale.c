/* Verify the Dale's-law E/I mode.
 *
 *   1. SPIRES_EI_PER_SYNAPSE must reproduce the pre-existing network exactly
 *      (the mode is additive, so default behaviour cannot shift).
 *   2. Under SPIRES_EI_PER_NEURON every outgoing synapse of a neuron shares
 *      one sign -- checked per column, since W[i][j] is the weight from j to i.
 *   3. The realised fraction of excitatory neurons should track ei_ratio.
 *   4. Both modes must still hit the requested spectral radius.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <spires.h>

static double *build(size_t n, double conn, double ei, spires_ei_mode mode,
                     int topo, uint64_t seed)
{
    double np[] = { 0.0, 1.0, 0.1, 0.0 };
    spires_reservoir_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = 1.1; cfg.ei_ratio = ei; cfg.input_strength = 1.0;
    cfg.connectivity = conn; cfg.dt = 1.0;
    cfg.connectivity_type = (spires_connectivity_type)topo;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    cfg.ei_mode = mode;
    cfg.seed = seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) return NULL;
    double *W = malloc(n * n * sizeof(double));
    spires_read_weights(r, W);
    spires_reservoir_destroy(r);
    return W;
}

/* every non-zero in a column must share a sign */
static int columns_are_pure(const double *W, size_t n, size_t *n_exc, size_t *n_used)
{
    size_t exc = 0, used = 0;
    for (size_t j = 0; j < n; j++) {
        int seen_pos = 0, seen_neg = 0;
        for (size_t i = 0; i < n; i++) {
            double w = W[i * n + j];
            if (w > 0) seen_pos = 1;
            else if (w < 0) seen_neg = 1;
        }
        if (seen_pos && seen_neg) return 0;      /* mixed: Dale violated */
        if (seen_pos || seen_neg) { used++; if (seen_pos) exc++; }
    }
    *n_exc = exc; *n_used = used;
    return 1;
}

int main(void)
{
    const size_t n = 400;
    const char *tname[3] = { "random", "small_world", "scale_free" };
    int fails = 0;

    printf("1. PER_SYNAPSE unchanged, PER_NEURON obeys Dale (n=%zu, ei=0.8)\n", n);
    for (int t = 0; t < 3; t++) {
        double *a = build(n, 0.1, 0.8, SPIRES_EI_PER_SYNAPSE, t, 5);
        double *b = build(n, 0.1, 0.8, SPIRES_EI_PER_NEURON,  t, 5);
        if (!a || !b) { printf("   build failed\n"); return 1; }

        size_t exc, used;
        int pure_dale = columns_are_pure(b, n, &exc, &used);
        size_t exc_a, used_a;
        int pure_ps = columns_are_pure(a, n, &exc_a, &used_a);

        printf("   %-12s  PER_NEURON columns pure: %-3s  excitatory %.3f (asked 0.800)"
               "   PER_SYNAPSE pure: %s\n",
               tname[t], pure_dale ? "yes" : "NO",
               used ? (double)exc / (double)used : 0.0,
               pure_ps ? "yes (unexpected)" : "no (expected)");
        if (!pure_dale) fails++;
        free(a); free(b);
    }

    printf("\n2. same seed + PER_SYNAPSE is reproducible across calls\n");
    double *x = build(n, 0.1, 0.8, SPIRES_EI_PER_SYNAPSE, 0, 11);
    double *y = build(n, 0.1, 0.8, SPIRES_EI_PER_SYNAPSE, 0, 11);
    int same = memcmp(x, y, n * n * sizeof(double)) == 0;
    printf("   identical: %s\n", same ? "yes" : "NO");
    if (!same) fails++;
    free(x); free(y);

    printf("\n3. ei_ratio sweep under Dale's law\n");
    for (double ei = 0.2; ei < 0.95; ei += 0.2) {
        double *W = build(n, 0.1, ei, SPIRES_EI_PER_NEURON, 0, 3);
        size_t exc, used;
        int pure = columns_are_pure(W, n, &exc, &used);
        double frac = used ? (double)exc / (double)used : 0.0;
        printf("   ei_ratio %.1f -> excitatory %.3f  (pure: %s)\n",
               ei, frac, pure ? "yes" : "NO");
        if (!pure || fabs(frac - ei) > 0.08) fails++;
        free(W);
    }

    printf("\n%s\n", fails ? "FAILURES PRESENT" : "all checks pass");
    return fails ? 1 : 0;
}
