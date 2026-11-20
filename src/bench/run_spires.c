/* SPIRES runner for the reservoir speed benchmark.
 *
 * Timing strategy: spires_train_ridge() internally does
 *   (1) reset + step/read loop harvesting a T x N state matrix, then
 *   (2) normal equations + LAPACK solve.
 * There is no public entry point for phase (2) alone. Rather than derive
 * it by subtraction (too noisy -- the solve is smaller than the harvest's
 * run-to-run variance), we time each phase directly, calling the very same
 * internal primitives SPIRES uses: mat_transpose, mat_mat_mult, cblas_dgemv
 * and solve_linear_system_lud. The measured fit is therefore SPIRES's own
 * ridge computation, not a reimplementation of it.
 *
 * Neuron model is SPIRES_NEURON_LIF_DISCRETE with the simple synapse and
 * the sparse (CSR) backend: SPIRES's cheapest configuration.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <spires.h>
#include <cblas.h>
#include "math_utils.h"

/* Held-out split, mirroring common.py: the readout is fit on rows
 * [WASH, TRAIN_END), hyperparameters are chosen on [TRAIN_END, VAL_END), and
 * the reported NRMSE is the test segment [VAL_END, T). */
#define WASH      200
#define TRAIN_END 2000
#define VAL_END   3000

/* NRMSE of pred against y over rows [lo, hi); population std, as numpy. */
static double nrmse_range(const double *pred, const double *y, size_t lo, size_t hi)
{
    double mean = 0, se = 0, var = 0;
    size_t cnt = hi - lo;
    for (size_t t = lo; t < hi; t++) mean += y[t];
    mean /= (double)cnt;
    for (size_t t = lo; t < hi; t++) {
        se  += (pred[t] - y[t]) * (pred[t] - y[t]);
        var += (y[t] - mean) * (y[t] - mean);
    }
    return sqrt(se / var);
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Process CPU time over the same region. For a single-threaded run this
 * should track wall time closely; a large shortfall means the process was
 * descheduled and the measurement was taken against competing load. */
static double cpu_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    size_t n = 200; int rep = 0; unsigned seed = 0;
    double dt = 1.0, leak = 0.05, sr = 1.2, gain = 5.0, vth = 1.0, conn = 0.1;
    /* E/I ratio: a real tuning axis, not a constant. The torch-based LSMs
     * use symmetric +/- weights (effectively 0.5). */
    double ei = 0.8;
    /* Ridge lambda: tuned on the validation segment like every other
     * framework's, since a held-out score punishes an unregularized fit. */
    double ridge = 1e-6;
    const char *datafile = NULL;
    const char *topology = "random";
    const char *task = "narma10";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n")   && i+1 < argc) n    = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--rep")  && i+1 < argc) rep  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i+1 < argc) datafile = argv[++i];
        else if (!strcmp(argv[i], "--dt")   && i+1 < argc) dt   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--leak") && i+1 < argc) leak = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sr")   && i+1 < argc) sr   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gain") && i+1 < argc) gain = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vth")  && i+1 < argc) vth  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--conn") && i+1 < argc) conn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topology") && i+1 < argc) topology = argv[++i];
        else if (!strcmp(argv[i], "--ei") && i+1 < argc) ei = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ridge") && i+1 < argc) ridge = atof(argv[++i]);
        else if (!strcmp(argv[i], "--task") && i+1 < argc) task = argv[++i];
    }
    srand(seed + rep);

    char taskpath[512];
    if (!datafile) {
        snprintf(taskpath, sizeof taskpath, "data/%s.bin", task);
        datafile = taskpath;
    }

    /* --- load the shared NARMA-10 series (interleaved u,y float64) --- */
    FILE *f = fopen(datafile, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", datafile); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    size_t T = (size_t)bytes / (2 * sizeof(double));
    double *pairs = malloc((size_t)bytes);
    if (fread(pairs, 1, (size_t)bytes, f) != (size_t)bytes) { fprintf(stderr,"short read\n"); return 1; }
    fclose(f);

    double *u = malloc(T * sizeof(double));
    double *y = malloc(T * sizeof(double));
    for (size_t t = 0; t < T; t++) { u[t] = pairs[2*t]; y[t] = pairs[2*t+1]; }
    free(pairs);
    if (T <= VAL_END) { fprintf(stderr, "series too short for the split (T=%zu)\n", T); return 1; }

    /* --- configure: discrete LIF, simple synapse, sparse backend --- */
    double np[] = { 0.0, vth, leak, 0.0 };  /* V_0, V_th, leak_rate, bias */
    spires_reservoir_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei;
    cfg.input_strength = gain; cfg.connectivity = conn; cfg.dt = dt;
    if (!strcmp(topology, "random"))           cfg.connectivity_type = SPIRES_CONN_RANDOM;
    else if (!strcmp(topology, "small_world"))  cfg.connectivity_type = SPIRES_CONN_SMALL_WORLD;
    else if (!strcmp(topology, "scale_free"))   cfg.connectivity_type = SPIRES_CONN_SCALE_FREE;
    else { fprintf(stderr, "unknown topology %s\n", topology); return 1; }
    cfg.neuron_type       = SPIRES_NEURON_LIF_DISCRETE;
    cfg.neuron_params     = np;
    cfg.synapse_type      = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend   = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)(seed + rep);
    /* Native readout excludes the start-up transient, like every other
     * framework's fit on rows [WASH, TRAIN_END). */
    cfg.use_washout = 1;
    cfg.washout     = WASH;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) {
        fprintf(stderr, "reservoir create failed\n"); return 1;
    }

    /* --- phase 1: harvest (mirrors SPIRES's internal step 1) --- */
    double *X = malloc(T * n * sizeof(double));
    /* Spike accounting: a reservoir that never fires can still score well,
     * because its membrane potential degenerates into a leaky linear filter
     * of the input. Report firing statistics so such configurations can be
     * identified and rejected rather than silently tuned into. */
    double *spkbuf = malloc(n * sizeof(double));
    double *spkcnt = calloc(n, sizeof(double));
    if (!X || !spkbuf || !spkcnt) { fprintf(stderr, "alloc X failed\n"); return 1; }
    /* Warmup: touch every page of X and warm the caches, so that the
     * harvest and the subsequent native train_ridge are measured under
     * identical cache conditions. Without this the harvest runs cold,
     * train_ridge runs warm, and the difference can go negative. */
    spires_reservoir_reset(r);
    for (size_t t = 0; t < T; t++) {
        spires_step(r, &u[t]);
        spires_read_reservoir_state(r, &X[t * n]);
    }

    spires_reservoir_reset(r);
    double t0 = now();
    double c0 = cpu_now();
    for (size_t t = 0; t < T; t++) {
        spires_step(r, &u[t]);
        spires_read_reservoir_state(r, &X[t * n]);
    }
    double t_harvest = now() - t0;
    double cpu_harvest = cpu_now() - c0;

    /* second, untimed pass purely for spike statistics, so the reported
     * harvest time is not inflated by instrumentation */
    spires_reservoir_reset(r);
    double total_spikes = 0.0;
    for (size_t t = 0; t < T; t++) {
        spires_step(r, &u[t]);
        spires_read_spike_state(r, spkbuf);
        for (size_t i = 0; i < n; i++) { total_spikes += spkbuf[i]; spkcnt[i] += spkbuf[i]; }
    }
    size_t active = 0;
    for (size_t i = 0; i < n; i++) if (spkcnt[i] > 0) active++;
    double rate_hz = total_spikes / (double)n / ((double)T * 1e-3);

    /* --- phase 2: ridge solve, exactly as train_output_ridge_regression
     *     does it (steps 2-3): A = X'X + lambda*I, b = X'y, LU solve, on the
     *     training rows [WASH, TRAIN_END) as the library does with washout. --- */
    const size_t nf = TRAIN_END - WASH;
    const double *Xf = X + (size_t)WASH * n, *yf = y + WASH;
    double *X_T = malloc(n * nf * sizeof(double));
    double *A   = malloc(n * n * sizeof(double));
    double *b   = malloc(n * sizeof(double));
    double *w   = malloc(n * sizeof(double));
    if (!X_T || !A || !b || !w) { fprintf(stderr, "alloc fit failed\n"); return 1; }

    t0 = now();
    c0 = cpu_now();
    mat_transpose((double *)Xf, X_T, nf, n);
    mat_mat_mult(X_T, (double *)Xf, A, n, nf, n);
    for (size_t i = 0; i < n; i++) A[i * n + i] += ridge;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, nf, 1.0, X_T, nf, yf, 1, 0.0, b, 1);
    if (solve_linear_system_lud(A, b, w, n) != 0)
        fprintf(stderr, "warning: ridge solve failed\n");
    double t_fit = now() - t0;
    double cpu_fit = cpu_now() - c0;

    free(X_T); free(A); free(b); free(w);

    /* untimed: train natively on the training segment (spires_train_ridge
     * resets, harvests TRAIN_END steps and fits rows [WASH, TRAIN_END)) so
     * the scores reflect SPIRES's own readout */
    if (spires_train_ridge(r, u, y, TRAIN_END, ridge) != SPIRES_OK) {
        fprintf(stderr, "train_ridge failed\n"); return 1;
    }

    /* --- held-out scoring: run every row from rest, score each segment --- */
    spires_reservoir_reset(r);
    double *pred = spires_run(r, u, T);
    double nr_train = NAN, nr_val = NAN, nr_test = NAN;
    if (pred) {
        nr_train = nrmse_range(pred, y, WASH, TRAIN_END);
        nr_val   = nrmse_range(pred, y, TRAIN_END, VAL_END);
        nr_test  = nrmse_range(pred, y, VAL_END, T);
        free(pred);
    }

    printf("@@BENCH@@ {\"framework\": \"SPIRES\", \"group\": \"lsm\", "
           "\"n\": %zu, \"rep\": %d, \"topology\": \"%s\", \"task\": \"%s\", \"t_harvest\": %.9f, \"t_fit\": %.9f, "
           "\"cpu_harvest\": %.9f, \"cpu_fit\": %.9f, \"cpu_total\": %.9f, \"t_total\": %.9f, "
           "\"nrmse\": %.6f, \"nrmse_val\": %.6f, \"nrmse_train\": %.6f, "
           "\"split\": \"fit[%d:%d) val[%d:%d) test[%d:%zu)\", "
           "\"native_readout\": true, \"spike_rate_hz\": %.4f, \"ei_ratio\": %.3f, \"frac_active\": %.4f, \"total_spikes\": %.0f, "
           "\"notes\": \"discrete LIF, simple synapse, sparse backend; "
           "fit times SPIRES ridge primitives directly\"}\n",
           n, rep, topology, task, t_harvest, t_fit, cpu_harvest, cpu_fit, cpu_harvest + cpu_fit, t_harvest + t_fit,
           nr_test, nr_val, nr_train, WASH, TRAIN_END, TRAIN_END, VAL_END, VAL_END, T,
           rate_hz, ei, (double)active / (double)n, total_spikes);

    free(X); free(u); free(y);
    spires_reservoir_destroy(r);
    return 0;
}
