/* Does inhibitory STDP buy anything on a task?
 *
 * The hypothesis is not that plasticity raises peak accuracy -- a well-tuned
 * static reservoir is hard to beat -- but that it widens the range of
 * hyperparameters that work at all. The Vogels rule regulates firing rate, so
 * a reservoir initialised too hot or too cold should be pulled back into a
 * useful regime, flattening the performance curve against input gain.
 *
 * Protocol: plasticity runs unsupervised on the training inputs only, is then
 * frozen, and the readout is fitted to the resulting stationary reservoir.
 * Freezing works by zeroing the learning rates in place -- the reservoir holds
 * a pointer to the caller's parameter array, not a copy.
 *
 * Scoring uses a held-out tail of the series, so a reservoir cannot win by
 * memorising the fitting window.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <spires.h>

enum { PL_LAMBDA_I = 2, PL_ETA_I = 7 };

int main(int argc, char **argv)
{
    size_t n = 300; int rep = 0; unsigned seed = 0;
    double dt = 1.0, leak = 0.05, sr = 1.2, gain = 5.0, vth = 1.0, conn = 0.1;
    double ei = 0.8, rho_0 = 0.05, eta = 0.002, lambda = 0.0;
    long pretrain = 20000;
    int use_stdp = 0;
    const char *task = "narma10";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n")    && i+1 < argc) n    = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--rep")  && i+1 < argc) rep  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sr")   && i+1 < argc) sr   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gain") && i+1 < argc) gain = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ei")   && i+1 < argc) ei   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--task") && i+1 < argc) task = argv[++i];
        else if (!strcmp(argv[i], "--stdp") && i+1 < argc) use_stdp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rho0") && i+1 < argc) rho_0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eta")  && i+1 < argc) eta  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i+1 < argc) lambda = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pretrain") && i+1 < argc) pretrain = atol(argv[++i]);
    }

    char taskpath[512];
    snprintf(taskpath, sizeof taskpath, "data/%s.bin", task);
    FILE *f = fopen(taskpath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", taskpath); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    size_t T = (size_t)bytes / (2 * sizeof(double));
    double *pairs = malloc((size_t)bytes);
    if (fread(pairs, 1, (size_t)bytes, f) != (size_t)bytes) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    double *u = malloc(T * sizeof(double)), *y = malloc(T * sizeof(double));
    for (size_t t = 0; t < T; t++) { u[t] = pairs[2*t]; y[t] = pairs[2*t+1]; }
    free(pairs);

    size_t T_train = (size_t)(0.7 * (double)T);

    double np[] = { 0.0, vth, leak, 0.0 };
    double plp[10] = {
        20.0, 20.0, lambda, 1.05, 1.0, 1.0,
        20.0, eta, rho_0,
        50.0
    };

    spires_reservoir_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = sr; cfg.ei_ratio = ei;
    cfg.input_strength = gain; cfg.connectivity = conn; cfg.dt = dt;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    cfg.seed = (uint64_t)(seed + rep);
    if (use_stdp) {
        cfg.ei_mode = SPIRES_EI_PER_NEURON;      /* required for plasticity */
        cfg.plasticity_type = SPIRES_PLASTICITY_STDP;
        cfg.plasticity_params = plp;
    } else {
        cfg.ei_mode = SPIRES_EI_PER_NEURON;      /* same topology either way */
    }

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) {
        fprintf(stderr, "create failed\n"); return 1;
    }

    /* --- unsupervised pre-training on training inputs only --- */
    double w_mean_before = 0.0, w_mean_after = 0.0;
    double *W = malloc((size_t)n * n * sizeof(double));
    long cnt0 = 0; double s0 = 0.0;
    spires_read_weights(r, W);
    for (size_t k = 0; k < (size_t)n * n; k++) if (W[k] < 0.0) { s0 += -W[k]; cnt0++; }
    w_mean_before = cnt0 ? s0 / (double)cnt0 : 0.0;

    if (use_stdp) {
        for (long t = 0; t < pretrain; t++)
            spires_step(r, &u[(size_t)t % T_train]);
        /* freeze: the reservoir reads these through the stored pointer */
        plp[PL_LAMBDA_I] = 0.0;
        plp[PL_ETA_I]    = 0.0;
    }
    spires_read_weights(r, W);
    long cnt1 = 0; double s1 = 0.0;
    for (size_t k = 0; k < (size_t)n * n; k++) if (W[k] < 0.0) { s1 += -W[k]; cnt1++; }
    w_mean_after = cnt1 ? s1 / (double)cnt1 : 0.0;
    free(W);

    /* --- firing rate of the reservoir the readout will actually see --- */
    spires_reservoir_reset(r);
    double *spk = malloc(n * sizeof(double));
    long spikes = 0;
    for (size_t t = 0; t < T; t++) {
        spires_step(r, &u[t]);
        spires_read_spike_state(r, spk);
        for (size_t k = 0; k < n; k++) if (spk[k] != 0.0) spikes++;
    }
    double rate = (double)spikes / ((double)T * (double)n);
    free(spk);

    /* --- fit on the training window, score on the held-out tail --- */
    spires_reservoir_reset(r);
    if (spires_train_ridge(r, u, y, T_train, 1e-6) != SPIRES_OK) {
        fprintf(stderr, "train_ridge failed\n"); return 1;
    }
    spires_reservoir_reset(r);
    double *pred = spires_run(r, u, T);
    double nrmse = NAN;
    if (pred) {
        size_t a = T_train + 100, cnt = 0;   /* small washout into the tail */
        double se = 0, mean = 0, var = 0;
        for (size_t t = a; t < T; t++) { mean += y[t]; cnt++; }
        mean /= (double)cnt;
        for (size_t t = a; t < T; t++) {
            se  += (pred[t] - y[t]) * (pred[t] - y[t]);
            var += (y[t] - mean) * (y[t] - mean);
        }
        nrmse = sqrt(se / (double)cnt) / sqrt(var / (double)cnt);
        free(pred);
    }

    printf("@@STDP@@ {\"task\": \"%s\", \"n\": %zu, \"rep\": %d, \"seed\": %u, "
           "\"stdp\": %d, \"gain\": %.4f, \"sr\": %.3f, \"ei\": %.3f, "
           "\"rho_0\": %.4f, \"eta\": %.5f, \"lambda\": %.5f, \"pretrain\": %ld, "
           "\"rate\": %.5f, \"w_inh_before\": %.4f, \"w_inh_after\": %.4f, "
           "\"nrmse\": %.6f}\n",
           task, n, rep, seed, use_stdp, gain, sr, ei, rho_0, eta, lambda,
           use_stdp ? pretrain : 0L, rate, w_mean_before, w_mean_after, nrmse);

    free(u); free(y);
    spires_reservoir_destroy(r);
    return 0;
}
