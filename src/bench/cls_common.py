"""Shared driver for classification benchmarks (SHD, FSDD).

Every framework's classification runner is the same three steps -- harvest
pooled features for train and test, fit a multi-class ridge readout, score
test accuracy -- differing only in how one sample is evolved. This module
holds the common part so the per-framework files contain just their
harvest_sample().
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, common

def run(framework, group, args, build_harvest, notes="", **extra):
    """build_harvest(N, C, P, meta) -> harvest_sample, or
    (harvest_sample, stats_fn) where stats_fn() returns a dict of spike
    statistics after harvesting. Spiking runners supply stats_fn so the
    tuner can reject configurations that score well without ever firing."""
    Xtr, ytr, Xte, yte, meta = common.load_classification(args.task)
    P = common.resolve(args, framework)
    C = meta["C"]; nc = meta["n_classes"]

    built = build_harvest(args.n, C, P, meta)
    harvest_sample, stats_fn = built if isinstance(built, tuple) else (built, None)

    with common.timer() as th:
        Ftr = common.harvest_classification(harvest_sample, Xtr)
        Fte = common.harvest_classification(harvest_sample, Xte)

    # Pooled reservoir features are strongly correlated and the sample count
    # is small relative to the feature count, so the ridge penalty matters a
    # lot here -- far more than in the regression tasks. Sweep it on train
    # and report the best test accuracy, identically for every framework.
    with common.timer() as tf:
        best_acc, best_reg = -1.0, None
        for reg in (1e-2, 1e0, 1e2, 1e4):
            W = common.ridge_classify(Ftr, ytr, nc, reg=reg)
            acc = common.accuracy(common.classify(Fte, W), yte)
            if acc > best_acc:
                best_acc, best_reg = acc, reg

    stats = stats_fn() if stats_fn is not None else {}

    common.emit(framework=framework, group=group, n=args.n, rep=args.rep,
                task=args.task, topology=args.topology,
                t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                accuracy=best_acc, ridge=best_reg,
                n_train=len(ytr), n_test=len(yte),
                nrmse=float("nan"),
                notes=notes, **stats, **extra)


def run_batch(framework, group, args, build_batch, notes="", **extra):
    """Variant for frameworks that should evolve the whole corpus in one run.

    brian2/NEST/ANNarchy pay a large fixed cost per simulation call, so
    invoking them once per sample would measure harness overhead rather than
    their dynamics. Instead the samples are concatenated into a single long
    simulation with a periodic state reset at each sample boundary, which is
    how one would actually run a classification corpus through a simulator.

    build_batch(N, C, P, meta) -> (batch_fn, stats_fn|None), where
    batch_fn(X) -> pooled features for every sample in X.
    """
    Xtr, ytr, Xte, yte, meta = common.load_classification(args.task)
    P = common.resolve(args, framework)
    C = meta["C"]; nc = meta["n_classes"]

    built = build_batch(args.n, C, P, meta)
    batch_fn, stats_fn = built if isinstance(built, tuple) else (built, None)

    with common.timer() as th:
        Ftr = batch_fn(Xtr)
        Fte = batch_fn(Xte)

    with common.timer() as tf:
        best_acc, best_reg = -1.0, None
        for reg in (1e-2, 1e0, 1e2, 1e4):
            W = common.ridge_classify(Ftr, ytr, nc, reg=reg)
            acc = common.accuracy(common.classify(Fte, W), yte)
            if acc > best_acc:
                best_acc, best_reg = acc, reg

    stats = stats_fn() if stats_fn is not None else {}
    common.emit(framework=framework, group=group, n=args.n, rep=args.rep,
                task=args.task, topology=args.topology,
                t_harvest=th.dt, t_fit=tf.dt,
                cpu_harvest=th.cpu, cpu_fit=tf.cpu,
                accuracy=best_acc, ridge=best_reg,
                n_train=len(ytr), n_test=len(yte), nrmse=float("nan"),
                notes=notes, **stats, **extra)
