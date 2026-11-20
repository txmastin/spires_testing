#!/usr/bin/env python
"""Preprocess the Free Spoken Digit Dataset into a benchmark task.

Pipeline: 8 kHz waveform -> STFT -> 40-channel log-mel cochleagram ->
resampled to a fixed T frames -> per-utterance normalised -> threshold-
crossing (delta) spike coding.

Spike coding is the default because it is what makes this task a fair test
of a liquid state machine. In a prototype comparing a hand-written LIF
reservoir against a matched tanh reservoir -- identical topology, readout
and ridge sweep, differing only in the neuron -- the LIF won on spike-coded
input (0.79-0.85 vs 0.75-0.77 across three seeds) and tied on continuous
input (0.98 vs 0.98). The encoding, not the dataset, decides which
architecture has the advantage. `--encoding continuous` keeps the log-mel
features for comparison.

Output format is identical to prep_shd.py so the same runners consume both.
"""
import glob, json, pathlib, argparse
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft

SR = 8000

def mel_filterbank(nfft, sr, nmel, fmin=50.0, fmax=3800.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), nmel + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((nmel, nfft // 2 + 1))
    for i in range(nmel):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[i, l:c] = np.linspace(0, 1, c - l)
        fb[i, c:r] = np.linspace(1, 0, r - c)
    return fb

def cochleagram(path, fb, T, nfft=256):
    _, d = wav.read(path)
    d = d.astype(np.float64)
    d /= (np.abs(d).max() + 1e-9)
    _, _, Z = stft(d, fs=SR, nperseg=nfft, noverlap=nfft * 3 // 4)
    M = np.log10(fb @ (np.abs(Z) ** 2) + 1e-8).T
    idx = np.linspace(0, len(M) - 1, T).astype(int)
    M = M[idx]
    return (M - M.mean()) / (M.std() + 1e-9)

def delta_spikes(M, thr):
    """Threshold-crossing coding: a channel spikes when it has risen by `thr`
    since its last spike. The standard cochlea-to-spike conversion, and the
    reason this task suits an LSM -- information is carried by event timing
    rather than by amplitude."""
    S = np.zeros_like(M, dtype=np.uint8)
    ref = M[0].copy()
    for t in range(1, len(M)):
        up = (M[t] - ref) > thr
        S[t][up] = 1
        ref = np.where(up, M[t], ref)
    return S

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--nmel", type=int, default=40)
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--encoding", choices=("spikes", "continuous"), default="spikes")
    ap.add_argument("--seed", type=int, default=20260220)
    a = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    d = root / "data"
    files = sorted(glob.glob(str(d / "fsdd" / "recordings" / "*.wav")))
    if not files:
        raise SystemExit("no FSDD recordings found under data/fsdd/recordings")

    labels = np.array([int(pathlib.Path(f).name.split("_")[0]) for f in files])
    speakers = np.array([pathlib.Path(f).name.split("_")[1] for f in files])
    rng = np.random.default_rng(a.seed)

    # Stratified by digit, shuffled, then split. Speakers appear in both
    # splits: this is the speaker-dependent protocol, matching the prototype
    # that motivated the task. A speaker-independent split would be a harder
    # and separate experiment.
    order = np.concatenate([rng.permutation(np.where(labels == c)[0])
                            for c in range(10)])
    rng.shuffle(order)
    n_keep = min(len(order), a.n_train + a.n_test)
    order = order[:n_keep]
    tr, te = order[:a.n_train], order[a.n_train:]

    fb = mel_filterbank(256, SR, a.nmel)
    def encode(idx):
        X = np.empty((len(idx), a.T, a.nmel),
                     dtype=np.uint8 if a.encoding == "spikes" else np.float32)
        for row, i in enumerate(idx):
            M = cochleagram(files[i], fb, a.T)
            X[row] = delta_spikes(M, a.thr) if a.encoding == "spikes" else M
        return X

    Xtr, Xte = encode(tr), encode(te)
    ytr, yte = labels[tr].astype(np.int32), labels[te].astype(np.int32)

    np.savez_compressed(d / "fsdd.npz", X_train=Xtr, y_train=ytr,
                        X_test=Xte, y_test=yte)
    with open(d / "fsdd.bin", "wb") as fh:
        np.array([len(ytr), len(yte), a.T, a.nmel, 10], dtype=np.int32).tofile(fh)
        Xtr.astype(np.uint8).tofile(fh); ytr.tofile(fh)
        Xte.astype(np.uint8).tofile(fh); yte.tofile(fh)

    meta = {"task": "fsdd", "kind": "classification", "encoding": a.encoding,
            "n_train": len(ytr), "n_test": len(yte), "T": a.T, "C": a.nmel,
            "n_classes": 10, "thr": a.thr, "seed": a.seed,
            "speakers": sorted(set(speakers.tolist())),
            "split": "speaker-dependent, stratified by digit",
            "density": float(Xtr.mean())}
    (d / "fsdd.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
