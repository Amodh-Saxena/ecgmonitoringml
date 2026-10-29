"""Calibration quality and abstention analysis."""
import numpy as np


def reliability_curve(y, p, mask=None, n_bins: int = 10):
    """Equal-mass bins. Returns (mean_confidence, observed_frequency, counts) arrays."""
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    if mask is not None:
        keep = np.asarray(mask).ravel() > 0
        y, p = y[keep], p[keep]
    if not len(p):
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(p)
    bins = np.array_split(order, min(n_bins, len(p)))
    conf = np.array([p[b].mean() for b in bins]); obs = np.array([y[b].mean() for b in bins])
    return conf, obs, np.array([len(b) for b in bins])


def expected_calibration_error(y, p, mask=None, n_bins: int = 10) -> float | None:
    conf, obs, counts = reliability_curve(y, p, mask, n_bins)
    if not len(counts):
        return None
    return float(np.sum(counts * np.abs(conf - obs)) / counts.sum())


def per_class_ece(y, p, mask, n_bins: int = 10):
    return [expected_calibration_error(y[:, k], p[:, k], mask[:, k], n_bins) for k in range(y.shape[1])]


def brier_per_class(y, p, mask):
    out = []
    for k in range(y.shape[1]):
        v = np.asarray(mask)[:, k] > 0
        out.append(float(np.mean((np.asarray(p)[v, k] - np.asarray(y)[v, k]) ** 2)) if v.any() else None)
    return out


def sample_correct(y, p, mask, thresholds):
    """A sample is correct when every known class decision matches the label."""
    pred = np.asarray(p) >= np.asarray(thresholds)
    wrong = (pred != (np.asarray(y) > 0.5)) & (np.asarray(mask) > 0)
    return ~wrong.any(axis=1)


def sample_confidence(p, thresholds):
    return np.min(np.abs(np.asarray(p) - np.asarray(thresholds)), axis=1)


def abstention_curve(confidence, correct, n_points: int = 20):
    """Accuracy against coverage as the confidence limit varies."""
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    limits = np.quantile(confidence, np.linspace(0, 0.95, n_points))
    rows = []
    for lim in limits:
        keep = confidence >= lim
        rows.append({"limit": float(lim), "coverage": float(keep.mean()),
                     "accuracy": float(correct[keep].mean()) if keep.any() else None})
    return rows


def choose_confidence_limit(confidence, correct, target_accuracy: float = 0.9, min_coverage: float = 0.5):
    """Smallest limit (max coverage) reaching target accuracy; fall back to the most accurate point."""
    curve = [r for r in abstention_curve(confidence, correct, 40) if r["accuracy"] is not None]
    ok = [r for r in curve if r["accuracy"] >= target_accuracy and r["coverage"] >= min_coverage]
    return (ok[0] if ok else max(curve, key=lambda r: r["accuracy"]))["limit"]
