/* Establish, by running SPIRES's own propagation path, which index of
 * W[i][j] is the source neuron and which is the target -- and therefore
 * which index neuron_sign is keyed on.
 *
 * Method: drive a one-hot spike vector through the exact calls step_reservoir
 * makes (synapse_prepare then synapse_row_dot) and see which neurons receive
 * current. No reasoning about the code, only its behaviour.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "reservoir.h"
#include "synapse.h"

int main(void)
{
    const size_t n = 60;
    double np[] = { 0.0, 1.0, 0.1, 0.0 };

    struct reservoir_params p;
    memset(&p, 0, sizeof p);
    p.num_neurons = n; p.num_inputs = 1; p.num_outputs = 1;
    p.spectral_radius = 1.0; p.ei_ratio = 0.7; p.input_strength = 1.0;
    p.connectivity = 0.25; p.dt = 1.0;
    p.connectivity_type = RANDOM;
    p.neuron_type = LIF_DISCRETE; p.neuron_params = np;
    p.synapse_type = SYNAPSE_SIMPLE; p.synapse_backend = SYNAPSE_SPARSE;
    p.ei_mode = EI_PER_NEURON;
    p.seed = 21;

    struct reservoir *r = create_reservoir(&p);
    if (!r) { printf("create failed\n"); return 1; }
    if (init_reservoir(r) != 0) { printf("init failed\n"); return 1; }

    double *W = malloc(n * n * sizeof(double));
    read_reservoir_weights(r, W);

    /* pick a source neuron that actually has outgoing edges */
    size_t src = 0;
    for (size_t j = 0; j < n; j++) {
        int any = 0;
        for (size_t i = 0; i < n; i++) if (W[i * n + j] != 0.0) any = 1;
        if (any) { src = j; break; }
    }

    /* one-hot: only `src` fired */
    double *spikes = calloc(n, sizeof(double));
    spikes[src] = 1.0;

    /* exactly what step_reservoir does */
    const double *eff = synapse_prepare(&r->W, spikes, r->dt);
    double *recv = calloc(n, sizeof(double));
    for (size_t i = 0; i < n; i++)
        recv[i] = synapse_row_dot(&r->W, i, eff);

    /* Hypothesis A: W[i][j] is the weight FROM j TO i, so a spike at src
     *               delivers W[i][src] to each neuron i (src = column).
     * Hypothesis B: the transpose, delivering W[src][i] (src = row). */
    double errA = 0.0, errB = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dA = fabs(recv[i] - W[i * n + src]);
        double dB = fabs(recv[i] - W[src * n + i]);
        if (dA > errA) errA = dA;
        if (dB > errB) errB = dB;
    }
    printf("one-hot spike at neuron %zu\n", src);
    printf("  A: received[i] == W[i][src]  (src is the COLUMN) -> max err %.3e\n", errA);
    printf("  B: received[i] == W[src][i]  (src is the ROW)    -> max err %.3e\n", errB);
    printf("  => source neuron is the %s index\n",
           errA < errB ? "COLUMN (second)" : "ROW (first)");

    int fails = 0;
    if (!(errA < 1e-15)) { fails++; printf("  !! hypothesis A did not hold exactly\n"); }

    /* Given that, neuron_sign must be keyed on the column, and column `src`
     * must be sign-pure and agree with neuron_sign[src]. */
    if (r->neuron_sign) {
        int pos = 0, neg = 0, n_out = 0;
        for (size_t i = 0; i < n; i++) {
            double w = W[i * n + src];
            if (w > 0) { pos = 1; n_out++; }
            else if (w < 0) { neg = 1; n_out++; }
        }
        int expect = r->neuron_sign[src];
        int actual = pos ? 1 : (neg ? -1 : 0);
        printf("\nneuron %zu: neuron_sign=%+d, its %d outgoing weights are %s, observed sign %+d\n",
               src, expect, n_out, (pos && neg) ? "MIXED" : "uniform", actual);
        if (pos && neg) { fails++; printf("  !! outgoing weights are mixed\n"); }
        if (actual != expect) { fails++; printf("  !! sign table disagrees with W\n"); }

        /* and the receiving neurons should span both identities */
        int recv_pos = 0, recv_neg = 0;
        for (size_t i = 0; i < n; i++) {
            if (W[i * n + src] == 0.0) continue;
            if (r->neuron_sign[i] > 0) recv_pos = 1; else recv_neg = 1;
        }
        printf("targets of neuron %zu include excitatory: %s, inhibitory: %s\n",
               src, recv_pos ? "yes" : "no", recv_neg ? "yes" : "no");
    }

    printf("\n%s\n", fails ? "FAILURES PRESENT" : "convention confirmed");
    free(W); free(spikes); free(recv);
    free_reservoir(r);
    return fails ? 1 : 0;
}
