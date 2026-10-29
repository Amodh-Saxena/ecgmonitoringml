import json
import numpy as np
from pathlib import Path
from ecg.config import ExperimentConfig
from ecg.datasets.build import ECGArrayDataset, preprocess_config_from
from ecg.datasets.ptbxl import iter_ptbxl
from ecg.preprocess.signal import preprocess

cfg = ExperimentConfig.from_yaml("configs/baseline.yaml")
cache_dirs = [d for d in Path("data/cache").iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "manifest.json").exists()]
if not cache_dirs:
    print("No finished cache directory found.")
    exit(1)
cache = max(cache_dirs, key=lambda d: (d / "manifest.json").stat().st_mtime)
print(f"Checking cache: {cache}")

X = np.load(cache / "signals.npy", mmap_mode="r")
LM = np.load(cache / "lead_masks.npy")
print("shape", X.shape, X.dtype, "NaN/inf:", int((~np.isfinite(X)).sum()))
print("all 12 leads present in", int((LM.sum(1) == 12).sum()), "of", len(LM))
print("per-lead mean (abs max):", float(np.abs(X[:, :, :].mean(-1)).max()), " std range:",
      float(X.std(-1).min()), float(X.std(-1).max()))

# determinism on real records: reprocess 3 and compare to cache
pre = preprocess_config_from(cfg.data)
for i, rec in enumerate(iter_ptbxl(cfg.data.root, 100, limit=3)):
    a, ma = preprocess(rec.signal, rec.sampling_rate, rec.lead_names, pre)
    b, mb = preprocess(rec.signal, rec.sampling_rate, rec.lead_names, pre)
    print(rec.record_id, "same twice:", np.array_equal(a, b), " equals cache:", np.array_equal(a, X[i]), "mask=12:", ma.sum() == 12)

# labels in cache vs step-1 matrix
z = np.load("data/processed/ptbxl_labels_v1.npz")
Y, M = np.load(cache / "targets.npy"), np.load(cache / "target_masks.npy")
print("labels match step 1:", np.array_equal(Y, z["y"]), np.array_equal(M, z["mask"]))
ds = ECGArrayDataset(cache, "train")
print("train dataset", len(ds), "augmentation active:", ds.augmentation is not None,
      "| val augmentation:", ECGArrayDataset(cache, "validation", {"noise_std": 1}).augmentation)
