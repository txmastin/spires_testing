/* Compare SPIRES's two spectral-radius routines on identical matrices.
 *
 * Builds a reservoir so the weights come from SPIRES's own generator and
 * topology code, reads the matrix back, then evaluates both the existing
 * power iteration and the new restarted Arnoldi on it. The matrix is dumped
 * so an independent tool can supply ground truth.
 *
 * Runs while calc_spectral_radius still points at the power iteration, so
 * the new routine can be judged before anything depends on it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cblas.h>
#include <spires.h>
#include "math_utils.h"

struct dense_ctx { const double *A; size_t n; };

static void dense_mv(void *ctx, const double *x, double *y)
{
    const struct dense_ctx *c = (const struct dense_ctx *)ctx;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)c->n, (int)c->n,
                1.0, c->A, (int)c->n, x, 1, 0.0, y, 1);
}

int main(int argc, char **argv)
{
    size_t n = 300; double conn = 0.1, ei = 0.5, sr = 1.2;
    unsigned seed = 0; const char *topology = "random";
    const char *out = "/tmp/cmp_W.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--conn") && i+1 < argc) conn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ei") && i+1 < argc) ei = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sr") && i+1 < argc) sr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topology") && i+1 < argc) topology = argv[++i];
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }
    srand(seed);

    double np_[] = { 0.0, 1.0, 0.1, 0.0 };
    spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei; cfg.input_strength = 1.0;
    cfg.connectivity = conn; cfg.dt = 1.0;
    if (!strcmp(topology, "random"))            cfg.connectivity_type = SPIRES_CONN_RANDOM;
    else if (!strcmp(topology, "small_world"))  cfg.connectivity_type = SPIRES_CONN_SMALL_WORLD;
    else                                        cfg.connectivity_type = SPIRES_CONN_SCALE_FREE;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np_;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE; cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) {
        fprintf(stderr, "create failed\n"); return 1;
    }

    double *W = malloc(n * n * sizeof(double));
    if (!W) { fprintf(stderr, "alloc\n"); return 1; }
    if (spires_read_weights(r, W) != SPIRES_OK) {
        fprintf(stderr, "read_weights failed\n"); return 1;
    }

    /* calc_spectral_radius is still the power iteration at this point */
    double rho_pow = calc_spectral_radius(W, n);

    struct dense_ctx ctx = { W, n };
    double rho_arn = spectral_radius_arnoldi(dense_mv, &ctx, n);

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fwrite(W, sizeof(double), n * n, f);
    fclose(f);

    printf("%zu %.12g %.12g\n", n, rho_pow, rho_arn);

    free(W);
    spires_reservoir_destroy(r);
    return 0;
}
