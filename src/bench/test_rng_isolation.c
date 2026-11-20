/* Verify the reservoir RNG is decoupled from the global srand()/rand() pair.
 *
 * Four properties, each a separate failure mode the old global-rand design
 * allowed:
 *   1. same cfg.seed  -> identical network, whatever srand() was called with
 *   2. different cfg.seed -> different network
 *   3. building a reservoir consumes no draws from the caller's rand() stream
 *   4. two reservoirs built in sequence with the same seed are identical
 *      (no cross-contamination from the first build into the second)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <spires.h>
#include <omp.h>

/* FNV-1a over the weight matrix, so networks compare as a single number. */
static uint64_t hash_weights(spires_reservoir *r, size_t n)
{
    double *W = malloc(n * n * sizeof(double));
    if (!W || spires_read_weights(r, W) != SPIRES_OK) { free(W); return 0; }
    const unsigned char *p = (const unsigned char *)W;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n * n * sizeof(double); i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    free(W);
    return h;
}

static uint64_t build_and_hash(uint64_t seed, size_t n)
{
    double np[] = { 0.0, 1.0, 0.1, 0.0 };
    spires_reservoir_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.num_neurons = n; cfg.num_inputs = 1; cfg.num_outputs = 1;
    cfg.spectral_radius = 0.9; cfg.ei_ratio = 0.5; cfg.input_strength = 1.0;
    cfg.connectivity = 0.1; cfg.dt = 1.0;
    cfg.connectivity_type = SPIRES_CONN_RANDOM;
    cfg.neuron_type = SPIRES_NEURON_LIF_DISCRETE;
    cfg.neuron_params = np;
    cfg.synapse_type = SPIRES_SYNAPSE_SIMPLE;
    cfg.synapse_backend = SPIRES_SYNAPSE_SPARSE;
    cfg.seed = seed;

    spires_reservoir *r = NULL;
    if (spires_reservoir_create(&cfg, &r) != SPIRES_OK) return 0;
    uint64_t h = hash_weights(r, n);
    spires_reservoir_destroy(r);
    return h;
}

int main(void)
{
    const size_t n = 120;
    int fails = 0;

    /* 1. same seed, wildly different global srand() state */
    srand(1);        uint64_t a = build_and_hash(42, n);
    srand(999999);   uint64_t b = build_and_hash(42, n);
    for (int i = 0; i < 5000; i++) (void)rand();
    uint64_t c = build_and_hash(42, n);
    printf("1. same cfg.seed across srand(1)/srand(999999)/mid-stream: %s\n",
           (a == b && b == c) ? "IDENTICAL (pass)" : "DIFFER (FAIL)");
    if (!(a == b && b == c)) fails++;

    /* 2. different seed must give a different network */
    uint64_t d = build_and_hash(43, n);
    printf("2. cfg.seed 42 vs 43: %s\n",
           (a != d) ? "different (pass)" : "IDENTICAL (FAIL)");
    if (a == d) fails++;

    /* 3. reservoir construction must not consume the caller's rand() draws */
    srand(7);
    int x1 = rand();
    (void)build_and_hash(42, n);
    int x2 = rand();
    srand(7);
    int y1 = rand();
    int y2 = rand();
    printf("3. caller's rand() stream across a build: %s (%d,%d vs %d,%d)\n",
           (x1 == y1 && x2 == y2) ? "untouched (pass)" : "PERTURBED (FAIL)",
           x1, x2, y1, y2);
    if (!(x1 == y1 && x2 == y2)) fails++;

    /* 4. no cross-contamination between successive builds */
    uint64_t e = build_and_hash(7, n);
    (void)build_and_hash(12345, n);
    uint64_t f = build_and_hash(7, n);
    printf("4. seed 7 before and after an unrelated build: %s\n",
           (e == f) ? "IDENTICAL (pass)" : "DIFFER (FAIL)");
    if (e != f) fails++;

    /* 5. concurrent construction: each reservoir must match the network it
     *    would have got if built alone, regardless of interleaving */
    enum { NTH = 8 };
    uint64_t seq[NTH], par[NTH];
    for (int i = 0; i < NTH; i++)
        seq[i] = build_and_hash((uint64_t)(1000 + i), n);

    #pragma omp parallel for
    for (int i = 0; i < NTH; i++)
        par[i] = build_and_hash((uint64_t)(1000 + i), n);

    int cmatch = 1;
    for (int i = 0; i < NTH; i++)
        if (seq[i] != par[i]) cmatch = 0;
    printf("5. %d reservoirs built concurrently vs sequentially: %s\n",
           NTH, cmatch ? "IDENTICAL (pass)" : "DIFFER (FAIL)");
    if (!cmatch) fails++;

    printf("\n%s\n", fails ? "FAILURES PRESENT" : "all 5 properties hold");
    return fails ? 1 : 0;
}
