/* Dump SPIRES's generated recurrent weight matrix for a given configuration.
 *
 * Used to measure the spectral radius SPIRES actually produces, which differs
 * from the requested value whenever weights are sign-balanced: SPIRES scales
 * using power iteration (csr_spectral_radius), which converges only when the
 * dominant eigenvalue is real. With ei_ratio near 0.5 the spectrum is a
 * complex disc and the estimate is low, so the matrix is scaled up too far.
 *
 * Rather than modify SPIRES, the benchmark measures the true radius here and
 * matches the other frameworks to it, so every framework runs at the same
 * effective spectral radius as SPIRES.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <spires.h>

int main(int argc, char **argv)
{
    size_t n = 300; double conn = 0.1, ei = 0.5, sr = 1.2; unsigned seed = 0;
    const char *out = "/tmp/spires_W.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--conn") && i+1 < argc) conn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ei") && i+1 < argc) ei = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sr") && i+1 < argc) sr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }
    srand(seed);
    double np_[] = { 0.0, 1.0, 0.1, 0.0 };
    spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei; cfg.input_strength = 1.0;
    cfg.connectivity = conn; cfg.dt = 1.0;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np_;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE; cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)seed;
    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) { fprintf(stderr, "create\n"); return 1; }
    double *W = malloc(n * n * sizeof(double));
    if (spires_read_weights(r, W) != SPIRES_OK) { fprintf(stderr, "read\n"); return 1; }
    FILE *f = fopen(out, "wb"); fwrite(W, sizeof(double), n * n, f); fclose(f);
    printf("%zu\n", n);
    spires_reservoir_destroy(r); free(W);
    return 0;
}
