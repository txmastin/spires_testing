/* Cost of the spectral-radius change: reservoir creation and the two
 * routines head to head on identical matrices. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include <spires.h>
#include "math_utils.h"
#include "sparse.h"

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv)
{
    size_t ns[] = { 200, 500, 1000, 2000 };
    double conn = 0.1, ei = 0.5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--conn") && i+1 < argc) conn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ei") && i+1 < argc) ei = atof(argv[++i]);
    }

    printf("%-7s %14s %14s %14s\n", "N", "create(s)", "arnoldi(s)", "power(s)");
    for (int s = 0; s < 4; s++) {
        size_t n = ns[s];
        srand(7);

        double np_[] = { 0.0, 1.0, 0.1, 0.0 };
        spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
        cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
        cfg.spectral_radius = 1.2; cfg.ei_ratio = ei; cfg.input_strength = 1.0;
        cfg.connectivity = conn; cfg.dt = 1.0;
        cfg.connectivity_type = SPIRES_CONN_RANDOM;
        cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np_;
        cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
        cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
        /* SPIRES draws from its own stream now; srand() no longer applies. */
        cfg.seed = (uint64_t)7;

        spires_reservoir *r = NULL;
        double t0 = now();
        if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) {
            fprintf(stderr, "create failed at n=%zu\n", n); return 1;
        }
        double t_create = now() - t0;

        double *W = malloc(n * n * sizeof(double));
        spires_read_weights(r, W);

        t0 = now();
        double rho_a = calc_spectral_radius(W, n);
        double t_arn = now() - t0;

        t0 = now();
        double rho_p = calc_spectral_radius_power_iteration(W, n);
        double t_pow = now() - t0;

        printf("%-7zu %14.4f %14.4f %14.4f   (arnoldi %.5f, power %.5f)\n",
               n, t_create, t_arn, t_pow, rho_a, rho_p);

        free(W);
        spires_reservoir_destroy(r);
    }
    return 0;
}
