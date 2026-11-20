#!/usr/bin/env python
"""Preprocess Spiking Heidelberg Digits into a framework-neutral cache.

SHD is natively event-coded: each sample is a list of (spike time, channel)
pairs over 700 cochlear channels. We bin it to a fixed grid so every
framework consumes identical input, and subsample the corpus because the
benchmark cost scales as n_samples * T reservoir steps -- the full 8156-
sample train split at T=100 would be 815k steps per framework per repeat,
which is hours on this CPU for the slower simulators.

Outputs:
  shd.npz  -- X_train/X_test uint8 (n, T, C), y_train/y_test int32
  shd.bin  -- same, flat little-endian for the SPIRES C runner:
              [int32 n_train, n_test, T, C, n_classes]
              [uint8 X_train][int32 y_train][uint8 X_test][int32 y_test]
"""
import h5py, numpy as np, pathlib, json, argparse

def bin_split(f, idx, T, C, t_max):
    times = f["spikes"]["times"]; units = f["spikes"]["units"]
    X = np.zeros((len(idx), T, C), dtype=np.uint8)
    for row, i in enumerate(idx):
        t = np.asarray(times[i]); u = np.asarray(units[i], dtype=np.int64)
        keep = (t < t_max) & (u < C)
        tb = np.clip((t[keep] / t_max * T).astype(np.int64), 0, T - 1)
        np.add.at(X[row], (tb, u[keep]), 1)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--C", type=int, default=700)
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260220)
    a = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    d = root / "data"
    rng = np.random.default_rng(a.seed)

    out = {}
    for split, n_want in (("train", a.n_train), ("test", a.n_test)):
        with h5py.File(d / "shd" / f"shd_{split}.h5", "r") as f:
            y = np.asarray(f["labels"][:], dtype=np.int64)
            # stratified subsample: equal per class, so a 20-class accuracy
            # is not distorted by an unbalanced draw
            per = max(1, n_want // len(np.unique(y)))
            idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                             size=min(per, (y == c).sum()),
                                             replace=False)
                                  for c in np.unique(y)])
            rng.shuffle(idx)
            X = bin_split(f, idx, a.T, a.C, a.t_max)
            out[f"X_{split}"] = X
            out[f"y_{split}"] = y[idx].astype(np.int32)

    np.savez_compressed(d / "shd.npz", **out)

    n_classes = int(out["y_train"].max() + 1)
    with open(d / "shd.bin", "wb") as fh:
        np.array([len(out["y_train"]), len(out["y_test"]), a.T, a.C, n_classes],
                 dtype=np.int32).tofile(fh)
        out["X_train"].tofile(fh); out["y_train"].tofile(fh)
        out["X_test"].tofile(fh);  out["y_test"].tofile(fh)

    meta = {"task": "shd", "n_train": len(out["y_train"]),
            "n_test": len(out["y_test"]), "T": a.T, "C": a.C,
            "n_classes": n_classes, "t_max": a.t_max, "seed": a.seed,
            "kind": "classification",
            "mean_spikes_per_sample": float(out["X_train"].sum(axis=(1, 2)).mean())}
    (d / "shd.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
