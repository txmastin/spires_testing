/* Per-step latency for SPIRES.
 *
 * Real-time embedded control cares about worst-case latency and jitter, not
 * the aggregate throughput the main benchmark reports. This times each
 * individual spires_step + state read and dumps the raw per-step durations
 * so percentiles can be computed outside.
 *
 * Timing uses CLOCK_MONOTONIC per step. At these durations the clock call
 * itself is a non-trivial fraction of the measurement, so a calibration pass
 * measures the bare clock overhead and reports it alongside.
 *
 * The tail is only meaningful if the measurement is protected from the
 * scheduler. A 7 us step is short enough that one preemption or one page
 * fault dominates the maximum, which is why an unprotected run reports a
 * worst case whose standard deviation exceeds its mean. So the process pins
 * itself to one CPU, requests SCHED_FIFO, and locks its pages resident before
 * timing. Each is best-effort: SCHED_FIFO needs privileges this may not have,
 * and the run continues without it, recording in the output what it obtained
 * so the analysis can say whether the tail is trustworthy.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <spires.h>

/* Returns a bitmask of what was obtained: 1 affinity, 2 SCHED_FIFO, 4 mlock. */
static int isolate(void)
{
    int got = 0;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    if (sched_setaffinity(0, sizeof set, &set) == 0) got |= 1;
    struct sched_param sp; memset(&sp, 0, sizeof sp);
    sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) got |= 2;
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) got |= 4;
    return got;
}

static inline double now(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv)
{
    size_t n = 200, T = 4000; unsigned seed = 0;
    const char *out = "/tmp/spires_latency.bin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i+1 < argc) n = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--T") && i+1 < argc) T = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out = argv[++i];
    }
    srand(seed);
    int isolation = isolate();

    double np_[] = { 0.0, 1.0, 0.05, 0.0 };
    spires_reservoir_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = 1.2; cfg.ei_ratio = 0.5; cfg.input_strength = 5.0;
    cfg.connectivity = 0.1; cfg.dt = 1.0;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE; cfg.neuron_params = np_;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE; cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    /* SPIRES draws from its own stream now; srand() no longer applies. */
    cfg.seed = (uint64_t)seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) { fprintf(stderr, "create\n"); return 1; }

    double *st = malloc(n * sizeof(double));
    double *lat = malloc(T * sizeof(double));
    double *u = malloc(T * sizeof(double));
    if (!st || !lat || !u) { fprintf(stderr, "alloc\n"); return 1; }
    for (size_t t = 0; t < T; t++) u[t] = 0.5 * ((double)rand() / RAND_MAX);

    /* warmup so page faults and cache misses are not counted as jitter */
    for (size_t t = 0; t < T; t++) { spires_step(r, &u[t]); spires_read_reservoir_state(r, st); }

    spires_reservoir_reset(r);
    for (size_t t = 0; t < T; t++) {
        double a = now();
        spires_step(r, &u[t]);
        spires_read_reservoir_state(r, st);
        lat[t] = now() - a;
    }

    /* clock-call overhead, for context */
    double c0 = now();
    for (size_t t = 0; t < T; t++) { volatile double x = now(); (void)x; }
    double clock_overhead = (now() - c0) / (double)T;

    FILE *f = fopen(out, "wb");
    int hdr[2] = { (int)T, isolation };
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(&clock_overhead, sizeof(double), 1, f);
    fwrite(lat, sizeof(double), T, f);
    fclose(f);
    printf("wrote %s T=%zu clock_overhead=%.3g s isolation=%s%s%s\n",
           out, T, clock_overhead,
           (isolation & 1) ? "affinity " : "",
           (isolation & 2) ? "fifo "     : "",
           (isolation & 4) ? "mlock"     : "");
    spires_reservoir_destroy(r);
    return 0;
}
