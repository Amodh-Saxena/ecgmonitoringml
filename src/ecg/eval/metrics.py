"""Masked multilabel metrics and patient-level bootstrap intervals."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix


def evaluate_arrays(y_true, probabilities, mask, thresholds=None):
    y, p, m = map(np.asarray, (y_true, probabilities, mask))
    thresholds = np.full(y.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    per_class, aucs, aps = [], [], []
    for k in range(y.shape[1]):
        valid = m[:, k] > 0
        yt, yp = y[valid, k], p[valid, k]
        auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else None
        ap = float(average_precision_score(yt, yp)) if len(yt) and yt.sum() else None
        pred = yp >= thresholds[k]
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel() if len(yt) else (0, 0, 0, 0)
        per_class.append({"auroc": auc, "auprc": ap, "prevalence": float(yt.mean()) if len(yt) else None,
                          "f1": float(f1_score(yt, pred, zero_division=0)) if len(yt) else None,
                          "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
                          "n": int(len(yt))})
        if auc is not None: aucs.append(auc)
        if ap is not None: aps.append(ap)
    return {"macro_auroc": float(np.mean(aucs)) if aucs else None,
            "macro_auprc": float(np.mean(aps)) if aps else None, "classes": per_class}


def patient_bootstrap(y, p, mask, patient_ids, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    pids = np.asarray(patient_ids)
    patients = np.unique(pids)
    # Pre-index patient to record positions for O(1) sampling speed
    from collections import defaultdict
    patient_to_idx = defaultdict(list)
    for idx, pid in enumerate(pids):
        patient_to_idx[pid].append(idx)
    patient_idx_arrays = [np.array(patient_to_idx[pid], dtype=np.int64) for pid in patients]
    
    y_arr, p_arr, m_arr = np.asarray(y), np.asarray(p), np.asarray(mask)
    values = []
    n_p = len(patients)
    for _ in range(n_boot):
        sample_choice = rng.integers(0, n_p, size=n_p)
        idx = np.concatenate([patient_idx_arrays[i] for i in sample_choice])
        values.append(evaluate_arrays(y_arr[idx], p_arr[idx], m_arr[idx])["macro_auroc"])
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [None, None]

