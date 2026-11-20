/* Export the weights SPIRES generates, so every other framework can be run
 * on the identical network.
 *
 * This removes four sources of divergence at once: topology realisation,
 * weight distribution (SPIRES draws uniform magnitudes, the shared generator
 * drew normal), connection density, and effective spectral radius -- the last
 * of which SPIRES does not reproduce faithfully from its own request, since
 * csr_spectral_radius uses power iteration and that only converges when the
 * dominant eigenvalue is real. Rather than change SPIRES, we adopt whatever
 * network it actually builds as the common substrate.
 *
 * W_in has no public accessor, so it is recovered by impulse probing: at t=0
 * no neuron has spiked, so the recurrent term is exactly zero and one step
 * with a unit input on channel c yields gain * W_in[:, c]. A small probe
 * amplitude keeps every neuron subthreshold.
 *
 * Output: [int32 n, c] [double W[n*n]] [double W_in[n*c]]   (row-major)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <spires.h>

int main(int argc, char **argv)
{
    size_t n = 300, c = 1; double conn = 0.1, ei = 0.5, sr = 1.2, gain = 1.0;
    unsigned seed = 0; const char *out = "/tmp/spires_weights.bin";
    const char *topology = "random";
    double probe = 1e-6;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--c") && i+1 < argc) c = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--conn") && i+1 < argc) conn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ei") && i+1 < argc) ei = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sr") && i+1 < argc) sr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gain") && i+1 < argc) gain = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topology") && i+1 < argc) topology = argv[++i];
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }
    srand(seed);

    double np_[] = { 0.0, 1.0, 0.1, 0.0 };
    spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = c; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei; cfg.input_strength = gain;
    cfg.connectivity = conn; cfg.dt = 1.0;
    if (!strcmp(topology, "random")) cfg.connectivity_type = SPIRES_CONN_RANDOM;
    else if (!strcmp(topology, "small_world")) cfg.connectivity_type = SPIRES_CONN_SMALL_WORLD;
    else cfg.connectivity_type = SPIRES_CONN_SCALE_FREE;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np_;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE; cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) { fprintf(stderr, "create failed\n"); return 1; }

    double *W = malloc(n * n * sizeof(double));
    double *Win = malloc(n * c * sizeof(double));
    double *u = calloc(c, sizeof(double));
    double *st = malloc(n * sizeof(double));
    if (!W || !Win || !u || !st) { fprintf(stderr, "alloc\n"); return 1; }
    if (spires_read_weights(r, W) != SPIRES_OK) { fprintf(stderr, "read_weights\n"); return 1; }

    for (size_t k = 0; k < c; k++) {
        memset(u, 0, c * sizeof(double));
        u[k] = probe;
        spires_reservoir_reset(r);
        spires_step(r, u);
        spires_read_reservoir_state(r, st);
        for (size_t i = 0; i < n; i++) Win[i * c + k] = st[i] / probe;
    }

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    int hdr[2] = { (int)n, (int)c };
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(W, sizeof(double), n * n, f);
    fwrite(Win, sizeof(double), n * c, f);
    fclose(f);
    printf("exported n=%zu c=%zu -> %s\n", n, c, out);
    spires_reservoir_destroy(r);
    return 0;
}
