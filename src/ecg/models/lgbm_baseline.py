"""LightGBM baseline on handcrafted per-lead and global features (one binary model per class)."""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from ecg.eval.metrics import evaluate_arrays, patient_bootstrap
from ecg.eval.calibration import choose_thresholds


def extract_features(signal: np.ndarray, lead_mask: np.ndarray, fs: int = 100) -> np.ndarray:
    """signal [12,T] -> feature vector: per-lead stats, heart rate, RR/HRV stats, R amplitude."""
    feats = []
    for lead in range(signal.shape[0]):
        x = signal[lead]
        if lead_mask[lead] > 0 and x.std() > 0:
            feats += [float(x.mean()), float(x.std()), float(x.min()), float(x.max()),
                      float(np.ptp(x)), float(skew(x)), float(kurtosis(x)),
                      float(np.abs(np.diff(x)).mean())]
        else:
            feats += [np.nan] * 8
            
    ref = signal[1] if lead_mask[1] > 0 else signal[np.argmax(lead_mask)]  # lead II preferred
    ref_std = max(float(ref.std()), 1e-6)
    peaks, _ = find_peaks(ref, height=ref_std * 1.0, distance=int(0.25 * fs))
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        d = np.diff(rr)
        feats += [float(60 / rr.mean()), float(rr.mean()), float(rr.std()),
                  float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d) > 0.05)),
                  float(ref[peaks].mean()), float(len(peaks))]
    else:
        feats += [np.nan] * 5 + [np.nan, float(len(peaks))]
    return np.asarray(feats, dtype=np.float32)


def _extract_chunk(args):
    signals, lead_masks, fs = args
    return [extract_features(s, lm, fs) for s, lm in zip(signals, lead_masks)]


def extract_dataset_features(signals: np.ndarray, lead_masks: np.ndarray, fs: int = 100, n_workers: int = 4) -> np.ndarray:
    """Extract features for all records in parallel with progress reporting."""
    n = len(signals)
    chunk_size = max(100, n // (n_workers * 4))
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append((signals[i:i + chunk_size], lead_masks[i:i + chunk_size], fs))
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_res in executor.map(_extract_chunk, chunks):
            results.extend(chunk_res)
    return np.stack(results)


class LightGBMBaseline:
    def __init__(self, num_classes: int, seed: int = 42, **params):
        self.k, self.seed = num_classes, seed
        self.params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
                       "colsample_bytree": 0.8, "verbose": -1, "n_jobs": -1, **params}
        self.models = [None] * num_classes

    def fit(self, X, Y, M):
        from lightgbm import LGBMClassifier
        for k in range(self.k):
            valid = M[:, k] > 0
            if valid.sum() > 1 and len(np.unique(Y[valid, k])) == 2:
                self.models[k] = LGBMClassifier(random_state=self.seed, **self.params).fit(X[valid], Y[valid, k])
        return self

    def predict_proba(self, X):
        out = np.full((len(X), self.k), 0.5, dtype=np.float32)
        for k, m in enumerate(self.models):
            if m is not None:
                out[:, k] = m.predict_proba(X)[:, 1]
        return out


def train_eval_baseline(cache_path: str | Path, out_dir: str | Path = "runs/baseline", seed: int = 42):
    """Train LightGBM baseline on train split and evaluate on validation & test splits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_path)
    
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    splits = np.array([r["split"] for r in records])
    patient_ids = np.array([r["patient_id"] for r in records])
    
    signals = np.load(cache / "signals.npy", mmap_mode="r")
    lead_masks = np.load(cache / "lead_masks.npy")
    targets = np.load(cache / "targets.npy")
    target_masks = np.load(cache / "target_masks.npy")
    
    feat_cache = cache / "features_lgbm.npy"
    if feat_cache.exists():
        print(f"Loading cached baseline features from {feat_cache}...")
        X = np.load(feat_cache)
    else:
        print("Extracting handcrafted ECG features for baseline...")
        X = extract_dataset_features(signals, lead_masks, fs=100, n_workers=min(8, os.cpu_count() or 4))
        np.save(feat_cache, X)
        print(f"Saved extracted features: {X.shape}")
        
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "validation")[0]
    test_idx = np.where(splits == "test")[0]
    
    num_classes = targets.shape[1]
    model = LightGBMBaseline(num_classes=num_classes, seed=seed)
    print(f"Training LightGBM on {len(train_idx)} train records across {num_classes} classes...")
    model.fit(X[train_idx], targets[train_idx], target_masks[train_idx])
    
    # Predict validation
    p_val = model.predict_proba(X[val_idx])
    val_thresholds = choose_thresholds(targets[val_idx], p_val, target_masks[val_idx], target_sensitivity=0.9)
    val_metrics = evaluate_arrays(targets[val_idx], p_val, target_masks[val_idx], thresholds=val_thresholds)
    val_ci = patient_bootstrap(targets[val_idx], p_val, target_masks[val_idx], patient_ids[val_idx], n_boot=500, seed=seed)
    val_metrics["macro_auroc_ci95"] = val_ci
    
    # Predict test
    p_test = model.predict_proba(X[test_idx])
    test_metrics = evaluate_arrays(targets[test_idx], p_test, target_masks[test_idx], thresholds=val_thresholds)
    test_ci = patient_bootstrap(targets[test_idx], p_test, target_masks[test_idx], patient_ids[test_idx], n_boot=500, seed=seed)
    test_metrics["macro_auroc_ci95"] = test_ci
    
    summary = {
        "model": "lightgbm_baseline",
        "seed": seed,
        "validation": val_metrics,
        "test": test_metrics,
        "thresholds": val_thresholds
    }
    np.save(out_dir / "lgbm_val_probs.npy", p_val)
    np.save(out_dir / "lgbm_test_probs.npy", p_test)
    (out_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"--- Baseline Validation Macro AUROC: {val_metrics['macro_auroc']:.4f} (95% CI: {val_ci[0]:.4f} - {val_ci[1]:.4f}) ---")
    print(f"--- Baseline Test Macro AUROC:       {test_metrics['macro_auroc']:.4f} (95% CI: {test_ci[0]:.4f} - {test_ci[1]:.4f}) ---")
    return summary
