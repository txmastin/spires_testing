/* Numerical-equivalence harness for SPIRES.
 *
 * Dumps everything needed to replay a SPIRES run in an independent
 * implementation: the recurrent weight matrix, the effective input weights,
 * the driving input, and the full membrane trajectory.
 *
 * Recovering the input weights: SPIRES exposes no W_in accessor, but at t=0
 * no neuron has spiked, so the recurrent contribution is exactly zero and
 * the discrete-LIF update reduces to
 *     V_i = (1 - leak) * V_0 + gain * W_in_i * u + bias.
 * With V_0 = bias = 0 and a unit input, one step reads off gain * W_in
 * directly. A small probe amplitude keeps every neuron subthreshold so no
 * reset perturbs the measurement.
 *
 * Output (little-endian doubles unless noted):
 *   [int32 N, T]  [double leak, vth, v0, bias, dt]
 *   [double W[N*N]] [double Win_eff[N]] [double u[T]] [double V[T*N]]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <spires.h>

int main(int argc, char **argv)
{
    size_t n = 64; size_t T = 200; unsigned seed = 7;
    double leak = 0.15, vth = 1.0, v0 = 0.0, bias = 0.0;
    double dt = 1.0, sr = 0.9, gain = 1.0, conn = 0.15, ei = 0.5;
    double probe = 1e-6;                      /* keeps neurons subthreshold */
    const char *out = "results/spires_trace.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--T") && i+1 < argc) T = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--leak") && i+1 < argc) leak = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }
    srand(seed);

    double np_[] = { v0, vth, leak, bias };
    spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei;
    cfg.input_strength = gain; cfg.connectivity = conn; cfg.dt = dt;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE;
    cfg.neuron_params = np_;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) { fprintf(stderr, "create\n"); return 1; }

    double *W = malloc(n * n * sizeof(double));
    double *Win = malloc(n * sizeof(double));
    double *st = malloc(n * sizeof(double));
    double *u  = malloc(T * sizeof(double));
    double *V  = malloc(T * n * sizeof(double));
    if (!W || !Win || !st || !u || !V) { fprintf(stderr, "alloc\n"); return 1; }

    if (spires_read_weights(r, W) != SPIRES_OK) { fprintf(stderr, "read_weights\n"); return 1; }

    /* probe the effective input weights */
    spires_reservoir_reset(r);
    spires_step(r, &probe);
    spires_read_reservoir_state(r, st);
    for (size_t i = 0; i < n; i++) Win[i] = st[i] / probe;

    /* record a driven trajectory */
    for (size_t t = 0; t < T; t++) u[t] = 0.6 * ((double)rand() / RAND_MAX);
    spires_reservoir_reset(r);
    for (size_t t = 0; t < T; t++) {
        spires_step(r, &u[t]);
        spires_read_reservoir_state(r, &V[t * n]);
    }

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    int hdr[2] = { (int)n, (int)T };
    double par[5] = { leak, vth, v0, bias, dt };
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(par, sizeof(double), 5, f);
    fwrite(W, sizeof(double), n * n, f);
    fwrite(Win, sizeof(double), n, f);
    fwrite(u, sizeof(double), T, f);
    fwrite(V, sizeof(double), T * n, f);
    fclose(f);
    printf("wrote %s  N=%zu T=%zu\n", out, n, T);

    spires_reservoir_destroy(r);
    return 0;
}
