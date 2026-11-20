"""Shared utilities for the reservoir speed benchmark.

Every runner reports the same JSON schema so the aggregator can treat
frameworks uniformly:

  {framework, group, n, rep, t_harvest, t_fit, t_total,
   nrmse, native_readout, notes}

`native_readout` records whether the framework's own readout was used
(reservoirpy, pytorch-esn, RC.jl, SPIRES) or whether we supplied a shared
numpy ridge because the framework has none (brian2, nest, ANNarchy,
snnTorch, norse, rockpool). That distinction matters when reading t_fit.
"""
import resource, json, pathlib, time, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

REGRESSION_TASKS = ("narma10", "mackey_glass")
TASKS = ("narma10", "mackey_glass", "shd", "fsdd")

def load_data(task="narma10"):
    """Return (u, y) as float64 1-D arrays, plus washout length."""
    arr = np.load(DATA / f"{task}.npy")
    meta = json.loads((DATA / f"{task}.meta.json").read_text())
    return arr[0], arr[1], meta["washout"]

# ---------------------------------------------------------------------------
# Held-out evaluation (revision, R1 comment 3).
#
# The reservoir is run over the whole series from rest; the readout is fit on
# rows [washout, TRAIN_END) only. Rows [TRAIN_END, VAL_END) are a validation
# segment, used by tune.py to choose hyperparameters, and rows [VAL_END, T)
# are the test segment, whose NRMSE is the number reported. Every driver,
# including the SPIRES C runner and the Julia runner, uses these constants.
# ---------------------------------------------------------------------------
TRAIN_END = 2000
VAL_END = 3000

def split_scores(pred_all, y, wash):
    """NRMSE on the train, validation and test segments.

    `pred_all` holds the readout's prediction for every row of the series
    (row t predicts y[t]); rows before the washout are ignored. `nrmse` is the
    test score, so existing table code reads held-out accuracy."""
    pred_all = np.asarray(pred_all).ravel(); y = np.asarray(y).ravel()
    assert len(pred_all) == len(y) > VAL_END, (len(pred_all), len(y))
    return dict(nrmse_train=nrmse(pred_all[wash:TRAIN_END], y[wash:TRAIN_END]),
                nrmse_val=nrmse(pred_all[TRAIN_END:VAL_END], y[TRAIN_END:VAL_END]),
                nrmse=nrmse(pred_all[VAL_END:], y[VAL_END:]),
                split=f"fit[{wash}:{TRAIN_END}) val[{TRAIN_END}:{VAL_END}) test[{VAL_END}:{len(y)})")

class timer:
    """Wall and CPU time for the same region, in seconds.

    `dt` is wall time -- the user-facing quantity, and the only meaningful one
    for a multi-threaded comparison, since CPU time sums across threads. `cpu`
    is process CPU time over the identical region: for a single-threaded run it
    should track `dt` closely, and a cpu/wall ratio well below 1 means the
    process was descheduled, i.e. the measurement was taken against competing
    load and cannot be trusted.

    Pass `children=True` for a region whose work happens in a child process --
    Brian2's standalone device compiles and runs a separate binary, so the
    parent's process_time() would report nearly zero for it. The child figure
    comes from RUSAGE_CHILDREN and is only valid once the child has been
    reaped, which is true by the time the call returns.
    """
    def __init__(self, children=False):
        self.children = children
    def _cpu(self):
        if not self.children:
            return time.process_time()
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c0 = self._cpu()
        return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        self.cpu = self._cpu() - self.c0

def _ridge_penalty(d):
    """Diagonal penalty that leaves the bias term unpenalized.

    Frameworks disagreed here: reservoirpy centres the data and fits the bias
    separately, so its bias is never shrunk, while an augmented-column ridge
    penalizes the bias along with the weights. At the large lambda the
    classification tasks prefer (up to 1e4) that shrinks the bias toward zero
    and systematically disadvantages whichever framework uses the augmented
    form. Excluding the final (bias) column makes both conventions agree.
    """
    P = np.ones(d); P[-1] = 0.0
    return np.diag(P)

def ridge_fit(states, targets, reg=1e-6):
    """Shared ridge readout: states (T, N), targets (T,) -> Wout (N+1,).

    Bias column appended and left unpenalized, matching reservoirpy's
    centring formulation so every framework's readout is equivalent.
    """
    X = np.hstack([states, np.ones((states.shape[0], 1))])
    A = X.T @ X + reg * _ridge_penalty(X.shape[1])
    b = X.T @ targets
    return np.linalg.solve(A, b)

def ridge_predict(states, w):
    X = np.hstack([states, np.ones((states.shape[0], 1))])
    return X @ w

def nrmse(pred, true):
    """Normalised RMSE — a sanity check that the reservoir actually learned."""
    pred = np.asarray(pred).ravel(); true = np.asarray(true).ravel()
    s = true.std()
    if s == 0 or not np.all(np.isfinite(pred)):
        return float("nan")
    return float(np.sqrt(np.mean((pred - true) ** 2)) / s)

def emit(**kw):
    """Print one JSON record to stdout, prefixed so it survives noisy imports."""
    kw.setdefault("nrmse", float("nan"))
    kw.setdefault("notes", "")
    kw["t_total"] = kw.get("t_harvest", 0.0) + kw.get("t_fit", 0.0)
    if "cpu_harvest" in kw or "cpu_fit" in kw:
        kw["cpu_total"] = kw.get("cpu_harvest", 0.0) + kw.get("cpu_fit", 0.0)
    print("@@BENCH@@ " + json.dumps(kw), flush=True)

def argparser(desc):
    import argparse
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--n", type=int, required=True, help="reservoir size")
    p.add_argument("--rep", type=int, default=0, help="repeat index")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topology", default="random", choices=TOPOLOGIES)
    p.add_argument("--task", default="narma10", choices=TASKS)
    # tuning overrides; None means "use the tuned value, else the default"
    for k in ("gain", "sr", "leak", "vth", "ridge", "alpha", "beta", "wscale"):
        p.add_argument(f"--{k}", type=float, default=None)
    p.add_argument("--weights", choices=("shared", "spires"), default="shared",
                   help="'spires' runs every framework on the network SPIRES "
                        "actually generates, removing topology/distribution/"
                        "spectral-radius differences")
    p.add_argument("--no-tuned", action="store_true",
                   help="ignore config/tuned.json and use built-in defaults")
    p.add_argument("--one-step", action="store_true",
                   help="brian2/NEST only: integrate at 1 ms, i.e. one step per "
                        "input sample like SPIRES, instead of the idiomatic "
                        "0.1 ms (equal-work comparison, R1 comment 3)")
    return p


# ---------------------------------------------------------------------------
# Tuned hyperparameters.
#
# NARMA-10 and Mackey-Glass reward very different reservoir settings, and the
# spiking frameworks need different drive than the rate-coded ones to reach a
# comparable NRMSE. tune.py sweeps each (framework, task) pair and writes the
# winners to config/tuned.json; runners pick them up automatically so the
# speed comparison is between reservoirs that are each doing the task about
# as well as that framework can, rather than between one tuned and several
# detuned ones.
# ---------------------------------------------------------------------------
DEFAULTS = dict(gain=5.0, sr=1.2, leak=0.05, vth=1.0, ridge=1e-6,
                alpha=0.8, beta=0.5, wscale=1.0)
TUNED_PATH = ROOT / "config" / "tuned.json"

def tuned_table():
    if TUNED_PATH.exists():
        return json.loads(TUNED_PATH.read_text())
    return {}

def resolve(args, framework):
    """Merge built-in defaults <- tuned.json <- explicit CLI flags.

    Also sets the weight source, so `--weights spires` takes effect for any
    runner that goes through resolve(). Note this is only meaningful for the
    spiking frameworks: SPIRES's exported matrices carry a spectral radius
    above 1 (1.05-1.60 observed), which a spiking reservoir tolerates because
    the reset provides nonlinear contraction, but which violates the echo
    state property a tanh ESN depends on. The ESN libraries therefore keep
    their native initialisation.
    """
    P = dict(DEFAULTS)
    if not getattr(args, "no_tuned", False):
        P.update(tuned_table().get(f"{framework}|{args.task}", {}))
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            P[k] = v
    set_weight_source(getattr(args, "weights", "shared"),
                      conn=0.1, ei=P.get("ei", 0.5))
    return P


# ---------------------------------------------------------------------------
# Shared LSM topology.
#
# The spiking frameworks (snnTorch, norse, rockpool, brian2, NEST, ANNarchy)
# have no reservoir abstraction of their own, so we build the network. Using
# one generator for all of them means they differ only in how fast they
# integrate identical dynamics -- not in what network they happen to sample.
# SPIRES builds its own topology internally with matched parameters
# (connectivity 0.1, spectral radius 1.2, gain 5.0).
# ---------------------------------------------------------------------------
LSM = dict(density=0.1, spectral_radius=1.2, gain=5.0, leak=0.05, v_th=1.0)

TOPOLOGIES = ("random", "small_world", "scale_free")

def _mask(n, topology, density, rng):
    """Connectivity mask matched to `density` across the three topologies.

    random      -- Erdos-Renyi, each edge independent (directed).
    small_world -- Watts-Strogatz ring lattice, degree k = density*n, rewire 0.1.
    scale_free  -- Barabasi-Albert preferential attachment, m = density*n/2.

    Watts-Strogatz and Barabasi-Albert are undirected, so their masks are
    symmetric while the Erdos-Renyi mask is not. That is the same convention
    SPIRES uses internally, so the comparison stays consistent.
    """
    if topology == "random":
        return rng.random((n, n)) < density
    import networkx as nx
    seed = int(rng.integers(0, 2**31 - 1))
    if topology == "small_world":
        k = max(2, int(round(density * n)))
        if k % 2: k += 1
        k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
        g = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    elif topology == "scale_free":
        m = max(1, int(round(density * n / 2)))
        m = min(m, n - 1)
        g = nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        raise ValueError(f"unknown topology {topology!r}")
    return nx.to_numpy_array(g, dtype=bool)

def make_reservoir(n, seed=0, density=None, spectral_radius=None, gain=None,
                   topology="random"):
    """Return (W, Win): sparse recurrent matrix scaled to a spectral radius,
    plus an input vector. Dense float64 arrays; callers convert as needed."""
    density = LSM["density"] if density is None else density
    spectral_radius = LSM["spectral_radius"] if spectral_radius is None else spectral_radius
    gain = LSM["gain"] if gain is None else gain

    if _WEIGHT_CTX["source"] == "spires":
        # SPIRES already applies its own density, sign split and scaling, so
        # its matrix is taken as-is; only the input vector is rescaled by gain.
        W, Win = spires_weights(n, 1, seed=seed, sr=spectral_radius,
                                topology=topology)
        return W, Win[:, 0] * gain

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=(n, n))
    W *= _mask(n, topology, density, rng)
    # Scale to the requested spectral radius. For n this size a full eig is
    # cheap relative to the benchmark itself and is done outside the timed
    # region in every runner.
    ev = np.linalg.eigvals(W)
    rho = np.max(np.abs(ev))
    if rho > 0:
        W *= spectral_radius / rho
    Win = rng.uniform(-1.0, 1.0, size=(n,)) * gain
    return W, Win


# ---------------------------------------------------------------------------
# Classification tasks (SHD, FSDD).
#
# Structurally different from the regression tasks: instead of one long
# series, there are many short samples, the reservoir is reset between them,
# and the readout maps a pooled state vector to a class. The abstraction is
# at the *sample* level rather than the timestep level -- harvest_sample()
# takes a whole (T, C) sample and returns (T, N) states -- because Rockpool
# evolves a sequence in one vectorised call and brian2/NEST/ANNarchy run a
# whole simulation per call. A per-timestep callback would have forced those
# frameworks into a Python loop they do not otherwise need, which would be
# measuring our harness rather than them.
# ---------------------------------------------------------------------------
CLASSIFICATION_TASKS = ("shd", "fsdd")

def is_classification(task):
    return task in CLASSIFICATION_TASKS

def load_classification(task="shd"):
    """Return (X_train, y_train, X_test, y_test, meta).

    X arrays are (n_samples, T, C) uint8 spike/feature counts.
    """
    z = np.load(DATA / f"{task}.npz")
    meta = json.loads((DATA / f"{task}.meta.json").read_text())
    return (z["X_train"], z["y_train"], z["X_test"], z["y_test"], meta)

def pool_states(states):
    """Pool a (T, N) trajectory to one feature vector.

    Mean over time plus the final state: the mean captures overall
    population activity, the final state retains the trajectory endpoint.
    Concatenating both is standard for LSM readouts and costs nothing.
    """
    return np.concatenate([states.mean(axis=0), states[-1]])

def ridge_classify(F, y, n_classes, reg=1e-4):
    """One-vs-all ridge on one-hot targets -> W of shape (F+1, n_classes)."""
    X = np.hstack([F, np.ones((F.shape[0], 1))])
    Y = np.zeros((len(y), n_classes)); Y[np.arange(len(y)), y] = 1.0
    A = X.T @ X + reg * _ridge_penalty(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)

def classify(F, W):
    X = np.hstack([F, np.ones((F.shape[0], 1))])
    return np.argmax(X @ W, axis=1)

def accuracy(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())

def harvest_classification(harvest_sample, X):
    """Apply harvest_sample to every sample, returning pooled features."""
    feats = None
    for i in range(len(X)):
        s = harvest_sample(X[i])
        p = pool_states(np.asarray(s, dtype=np.float64))
        if feats is None:
            feats = np.empty((len(X), p.shape[0]), dtype=np.float64)
        feats[i] = p
    return feats


def make_input_weights(n, c, seed=0, gain=None):
    """Input matrix (n, c) for multi-channel tasks (SHD/FSDD).

    The regression tasks are single-input so `make_reservoir` returns a
    vector; classification input has C channels and needs a matrix. Kept
    separate so the recurrent topology generator stays identical across
    both task families.
    """
    gain = LSM["gain"] if gain is None else gain
    if _WEIGHT_CTX["source"] == "spires":
        _, Win = spires_weights(n, c, seed=seed)
        return Win * gain
    rng = np.random.default_rng(seed + 9973)
    return rng.normal(0.0, 1.0, size=(n, c)) * gain


# ---------------------------------------------------------------------------
# Spiking-regime reporting.
#
# A reservoir that never fires can still score well: its membrane potential
# becomes a leaky linear filter of the input, and on SHD that scored better
# than genuinely spiking configurations. Such a config is not an LSM, and
# tuning that selects one is measuring the wrong thing. Every spiking runner
# reports its firing statistics so the tuner can reject silent (and
# saturated) regimes, and so the published numbers can be shown to come from
# reservoirs that actually spike.
# ---------------------------------------------------------------------------
def spike_stats(total_spikes, n_neurons, n_steps, dt_seconds=1e-3,
                active_neurons=None):
    """Mean firing rate in Hz plus the fraction of neurons that ever fired."""
    duration = max(n_steps * dt_seconds, 1e-12)
    rate = float(total_spikes) / max(n_neurons, 1) / duration
    out = {"spike_rate_hz": rate, "total_spikes": int(total_spikes)}
    if active_neurons is not None:
        out["frac_active"] = float(active_neurons) / max(n_neurons, 1)
    return out


# ---------------------------------------------------------------------------
# Equalized weights: run every framework on the network SPIRES actually builds.
#
# Frameworks nominally running "the same LIF reservoir" diverged more than
# expected. Auditing found four controllable causes, all upstream of the
# neuron model: different topology realisations, different weight
# distributions (SPIRES draws uniform magnitudes, the shared generator drew
# normal), and -- most significantly -- different effective spectral radii,
# because SPIRES scales using power iteration, which only converges when the
# dominant eigenvalue is real. With sign-balanced weights it is not, and the
# realised radius overshoots the request by an amount that varies per draw
# (0.76x to 1.60x observed), so it cannot be predicted or corrected for.
#
# Rather than modify SPIRES, `--weights spires` exports the matrices SPIRES
# actually generates and hands them to every other framework. Whatever SPIRES
# builds becomes the common substrate, so the remaining differences are the
# neuron and synapse models themselves -- which is what we want to measure.
# ---------------------------------------------------------------------------
import subprocess, hashlib

_WEIGHT_CTX = {"source": "shared", "conn": 0.1, "ei": 0.5}
CACHE = ROOT / "data" / "weight_cache"

def set_weight_source(source, conn=0.1, ei=0.5):
    _WEIGHT_CTX.update(source=source, conn=conn, ei=ei)

def weight_source():
    return _WEIGHT_CTX["source"]

def spires_weights(n, c, seed=0, conn=None, ei=None, sr=1.2, gain=1.0,
                   topology="random"):
    """Export (or load from cache) the W and W_in that SPIRES generates."""
    conn = _WEIGHT_CTX["conn"] if conn is None else conn
    ei = _WEIGHT_CTX["ei"] if ei is None else ei
    key = f"{n}_{c}_{conn}_{ei}_{sr}_{gain}_{seed}_{topology}"
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".bin")
    if not path.exists():
        exe = ROOT / "src" / "bench" / "spires_export"
        subprocess.run([str(exe), "--n", str(n), "--c", str(c),
                        "--conn", str(conn), "--ei", str(ei), "--sr", str(sr),
                        "--gain", str(gain), "--seed", str(seed),
                        "--topology", topology, "--out", str(path)],
                       cwd=ROOT, check=True, capture_output=True)
    with open(path, "rb") as f:
        nn, cc = np.frombuffer(f.read(8), dtype="<i4")
        W = np.frombuffer(f.read(nn*nn*8), dtype="<f8").reshape(nn, nn).copy()
        Win = np.frombuffer(f.read(nn*cc*8), dtype="<f8").reshape(nn, cc).copy()
    return W, Win
