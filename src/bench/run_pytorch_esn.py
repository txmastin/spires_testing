#!/usr/bin/env python
"""pytorch-esn runner: native ESN reservoir + Cholesky ridge readout.

UPSTREAM BUG (pytorch-esn 1.2.5): ESN.fit() calls

    torch.cholesky_solve(XTy, XTX + lambda*I)

but torch.cholesky_solve expects the *Cholesky factor* as its second
argument, not the matrix itself. The result is a systematically wrong
readout shrunk toward zero (NRMSE ~3.8 instead of ~0.5 on NARMA-10).

We monkeypatch fit() with the correct factorisation below. This is the
same algorithm the library intends, so the timing remains representative
of pytorch-esn's native Cholesky path -- it simply also produces the
right answer. Benchmarking the unpatched version would report a
misleadingly fast solve for a readout that does not work.

Note: pytorch-esn accumulates the normal equations inside forward(), so
t_harvest includes that accumulation; t_fit is the factor-and-solve.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def _patched_fit(self):
    import torch
    from torch import nn
    # Solve in float64: pytorch-esn harvests in float32, and at lambda=1e-6
    # the float32 XTX is not numerically positive-definite, so the Cholesky
    # factorisation fails. reservoirpy/SPIRES/RC.jl all solve in double, so
    # this keeps the solve comparable without changing lambda.
    # Two deviations from upstream, both forced:
    #  1. cholesky_solve was passed the matrix instead of its factor (bug).
    #  2. pytorch-esn accumulates the normal equations in float32, which
    #     leaves XTX slightly indefinite (min eig ~ -2.5e-3 against max
    #     ~2.1e5, cond ~1e8). lambda=1e-6 cannot lift that positive, so a
    #     Cholesky factorisation genuinely cannot run. We use an LU solve
    #     in float64 instead -- same O(N^3) dense-solve class, and the same
    #     lambda, so the timing stays representative.
    XTX = self.XTX.double(); XTy = self.XTy.double()
    A = XTX + self.lambda_reg * torch.eye(XTX.size(0), dtype=torch.float64, device=XTX.device)
    A = (A + A.t()) / 2
    W = torch.linalg.solve(A, XTy).t().to(self.XTy.dtype)
    self.XTX = None; self.XTy = None
    self.readout.bias = nn.Parameter(W[:, 0])
    self.readout.weight = nn.Parameter(W[:, 1:])

def main():
    a = common.argparser("pytorch-esn").parse_args()
    import torch
    from torchesn.nn import ESN
    from torchesn.utils import prepare_target
    ESN.fit = _patched_fit
    torch.manual_seed(a.seed + a.rep)

    u, y, wash = common.load_data(a.task)
    X = torch.tensor(u, dtype=torch.float32).view(-1, 1, 1)
    Y = torch.tensor(y, dtype=torch.float32).view(-1, 1, 1)
    washout = [wash]
    tgt = prepare_target(Y, [X.size(0)], washout)

    P = common.resolve(a, "pytorch-esn")
    def make():
        torch.manual_seed(a.seed + a.rep)      # same draw -> same reservoir
        return ESN(1, a.n, 1, spectral_radius=P["sr"], density=0.1,
                   leaking_rate=P["leak"], lambda_reg=P["ridge"], w_ih_scale=P["gain"],
                   output_steps='all', readout_training='cholesky')

    # Timed: the same full-series training pass as the submitted benchmark, so
    # the timing does the same work as every other framework's (harvest the
    # whole series, accumulate the normal equations, solve).
    m = make()
    with common.timer() as th:
        m(X, washout, None, tgt)
    with common.timer() as tf:
        m.fit()

    # Untimed, for held-out accuracy: pytorch-esn accumulates its normal
    # equations inside the forward pass, so the readout can only be fit to a
    # contiguous prefix. Rebuild the identical reservoir, fit on the training
    # prefix only, then predict every row from rest (washout 0 keeps them all).
    tr = common.TRAIN_END
    m2 = make()
    m2(X[:tr], washout, None, prepare_target(Y[:tr], [tr], washout))
    m2.fit()
    out, _ = m2(X, [0])
    common.emit(framework="pytorch-esn", group="esn", n=a.n, rep=a.rep, task=a.task,
                t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                **common.split_scores(out.detach().numpy(), y, wash),
                native_readout=True,
                notes="patched upstream cholesky_solve bug; "
                      "LU solve in f64; t_harvest includes normal-equation accumulation; "
                      "accuracy from an identical reservoir refit on the training prefix")

if __name__ == "__main__":
    main()
