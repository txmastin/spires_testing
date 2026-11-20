/* Trace dumper for SPIRES's fractional-order (Grunwald-Letnikov) LIF neuron.
 *
 * Companion to verify_spires.c, which established that the integer-order
 * discrete LIF matches an independent reference to machine precision. The
 * fractional model is the harder and more important case: the GL history
 * sum, the binomial coefficient recursion and the memory-horizon truncation
 * each have places to go subtly wrong while still producing plausible-looking
 * spike trains.
 *
 * Verification matters here specifically because accuracy cannot substitute
 * for it. Mastin et al. (npj Unconv. Comput. 2026, Fig. 5) report that
 * classification accuracy varies by under two percentage points across a
 * +/-10% drift in alpha, so an implementation error equivalent to a few
 * percent shift in alpha would be invisible in any accuracy-based check.
 *
 * Dumps everything needed to replay the run independently:
 *   [int32 n, T]
 *   [double V_th, V_reset, V_rest, tau_m, alpha, T_mem, bias, t_ref, dt]
 *   [double W[n*n]] [double Win_eff[n]] [double u[T]] [double V[T*n]]
 *
 * Win_eff is recovered by impulse probe: at the first step no neuron has
 * spiked, V equals V_rest so the leak term vanishes, and with bias 0 the
 * update reduces to V = dt^alpha * gain * W_in * u.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <spires.h>

int main(int argc, char **argv)
{
    size_t n = 64, T = 300; unsigned seed = 5;
    double V_th = 1.0, V_reset = 0.0, V_rest = 0.0, tau_m = 20.0;
    double alpha = 0.7, T_mem = 100.0, bias = 0.0, t_ref = 0.0;
    double dt = 1.0, sr = 0.9, gain = 1.0, conn = 0.15, ei = 0.65;
    double probe = 1e-6;
    const char *out = "results/flif_trace.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--T") && i+1 < argc) T = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--alpha") && i+1 < argc) alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tmem") && i+1 < argc) T_mem = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tau") && i+1 < argc) tau_m = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dt") && i+1 < argc) dt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bias") && i+1 < argc) bias = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vrest") && i+1 < argc) V_rest = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vth") && i+1 < argc) V_th = atof(argv[++i]);
        else if (!strcmp(argv[i], "--conn") && i+1 < argc) conn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }
    srand(seed);

    double np_[] = { V_th, V_reset, V_rest, tau_m, alpha, T_mem, bias, t_ref };
    spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei; cfg.input_strength = gain;
    cfg.connectivity = conn; cfg.dt = dt;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_FLIF_GL;
    cfg.neuron_params = np_;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) { fprintf(stderr, "create failed\n"); return 1; }

    double *W = malloc(n * n * sizeof(double));
    double *Win = malloc(n * sizeof(double));
    double *st = malloc(n * sizeof(double));
    double *u = malloc(T * sizeof(double));
    double *V = malloc(T * n * sizeof(double));
    if (!W || !Win || !st || !u || !V) { fprintf(stderr, "alloc\n"); return 1; }
    if (spires_read_weights(r, W) != SPIRES_OK) { fprintf(stderr, "read_weights\n"); return 1; }

    spires_reservoir_reset(r);
    spires_step(r, &probe);
    spires_read_reservoir_state(r, st);
    for (size_t i = 0; i < n; i++) Win[i] = st[i] / probe;

    for (size_t t = 0; t < T; t++) u[t] = 0.6 * ((double)rand() / RAND_MAX);
    spires_reservoir_reset(r);
    for (size_t t = 0; t < T; t++) {
        spires_step(r, &u[t]);
        spires_read_reservoir_state(r, &V[t * n]);
    }

    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    int hdr[2] = { (int)n, (int)T };
    double par[9] = { V_th, V_reset, V_rest, tau_m, alpha, T_mem, bias, t_ref, dt };
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(par, sizeof(double), 9, f);
    fwrite(W, sizeof(double), n * n, f);
    fwrite(Win, sizeof(double), n, f);
    fwrite(u, sizeof(double), T, f);
    fwrite(V, sizeof(double), T * n, f);
    fclose(f);
    printf("wrote %s n=%zu T=%zu alpha=%.3f T_mem=%.1f dt=%.3f L=%d\n",
           out, n, T, alpha, T_mem, dt, (int)(T_mem / dt));
    spires_reservoir_destroy(r);
    return 0;
}
