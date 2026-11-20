/* Single-neuron harness for verify_exact.py: drives one SPIRES fractional
 * neuron (compiled from the library's own src/neurons sources) with a
 * constant input and prints its trajectory, for comparison against the exact
 * Mittag-Leffler solution of the subthreshold equation.
 *
 * usage: verify_exact_harness model alpha dt T V_rest_or_Vl V0 input
 * model 0 = GL  [V_th, V_reset, V_rest, tau_m=2, alpha, T_mem, bias=0, t_ref=0]
 * model 1 = Caputo [C_m=1, g_l=0.5, V_l, V_th, V_0, V_peak, alpha, t_ref=0, T_mem]
 * The memory window covers the whole run (T_mem = T + 10 dt), so the test
 * measures the discretization, not truncation. Spiking is disabled (V_th
 * huge). Prints "t V" after every update. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "flif_gl.h"
#include "flif_caputo.h"
int main(int c, char **v) {
    int model = atoi(v[1]); double a = atof(v[2]), dt = atof(v[3]), T = atof(v[4]);
    double vr = atof(v[5]), v0 = atof(v[6]), I = atof(v[7]);
    long n = lround(T / dt);
    double Tm = T + 10 * dt;
    if (model == 0) {
        double p[8] = { 1e9, v0, vr, 2.0, a, Tm, 0.0, 0.0 };
        int L = (int)(Tm / dt); double *cf = malloc(L * sizeof *cf);
        cf[0] = 1.0; for (int k = 1; k < L; k++) cf[k] = cf[k-1] * (1.0 - (a + 1.0) / (float)k);
        struct flif_gl_neuron *nn = init_flif_gl(p, dt, cf);
        for (long i = 1; i <= n; i++) { update_flif_gl(nn, I, dt); printf("%.10g %.17g\n", i*dt, nn->V); }
    } else {
        double p[9] = { 1.0, 0.5, vr, 1e9, v0, 1e9, a, 0.0, Tm };
        struct flif_caputo_neuron *nn = init_flif_caputo(p, dt);
        for (long i = 1; i <= n; i++) { update_flif_caputo(nn, I, dt); printf("%.10g %.17g\n", i*dt, nn->V); }
    }
    return 0;
}
