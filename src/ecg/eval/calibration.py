"""Source-validation-only calibration and threshold selection."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def fit_temperature(logits, targets, mask, per_class=True):
    z, y, m = map(np.asarray, (logits, targets, mask))
    temps = []
    for k in range(z.shape[1]):
        valid = m[:, k] > 0
        if not valid.any():
            temps.append(1.0); continue
        objective = lambda log_t: np.mean(np.logaddexp(0, z[valid, k] / np.exp(log_t)) - y[valid, k] * z[valid, k] / np.exp(log_t))
        temps.append(float(np.exp(minimize_scalar(objective, bounds=(-4, 4), method="bounded").x)))
    return temps if per_class else [float(np.mean(temps))]


def calibrate(logits, temperatures):
    return expit(np.asarray(logits) / np.asarray(temperatures))


def fit_platt(logits, targets, mask):
    """Per-class Platt scaling p = sigmoid(a*z + b). Returns list of (a, b)."""
    from sklearn.linear_model import LogisticRegression
    z, y, m = map(np.asarray, (logits, targets, mask))
    params = []
    for k in range(z.shape[1]):
        v = m[:, k] > 0
        if v.sum() < 2 or len(np.unique(y[v, k])) < 2:
            params.append((1.0, 0.0)); continue
        lr = LogisticRegression(C=1e6).fit(z[v, k:k + 1], y[v, k])
        params.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
    return params


def apply_platt(logits, params):
    a, b = np.asarray(params)[:, 0], np.asarray(params)[:, 1]
    return expit(np.asarray(logits) * a + b)


def fit_isotonic(logits, targets, mask):
    from sklearn.isotonic import IsotonicRegression
    z, y, m = map(np.asarray, (logits, targets, mask))
    models = []
    for k in range(z.shape[1]):
        v = m[:, k] > 0
        models.append(IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(z[v, k], y[v, k]) if v.sum() > 1 else None)
    return models


def apply_isotonic(logits, models):
    z = np.asarray(logits)
    return np.stack([mod.predict(z[:, k]) if mod is not None else expit(z[:, k]) for k, mod in enumerate(models)], 1)


def select_calibrator(logits, targets, mask, ece_target: float = 0.05, n_bins: int = 10):
    """Keep the simplest method meeting the ECE target: global T < per-class T < Platt < isotonic.

    Only temperature-type calibrators are serializable into calibration.json; Platt is exported as
    (a, b) pairs. Isotonic is reported for comparison but not selected for the bundle.
    """
    from ecg.eval.uncertainty import per_class_ece
    candidates = {
        "temperature_global": (lambda: fit_temperature(logits, targets, mask, False),
                               lambda p: calibrate(logits, p)),
        "temperature_per_class": (lambda: fit_temperature(logits, targets, mask, True),
                                  lambda p: calibrate(logits, p)),
        "platt": (lambda: fit_platt(logits, targets, mask), lambda p: apply_platt(logits, p)),
    }
    report, best = {}, None
    for name, (fit, apply) in candidates.items():
        params = fit()
        eces = [e for e in per_class_ece(np.asarray(targets), apply(params), np.asarray(mask), n_bins) if e is not None]
        report[name] = float(np.mean(eces)) if eces else None
        if best is None and report[name] is not None and report[name] <= ece_target:
            best = (name, params)
    if best is None:  # none met the target: take the lowest ECE
        name = min((n for n in report if report[n] is not None), key=lambda n: report[n])
        best = (name, candidates[name][0]())
    iso = fit_isotonic(logits, targets, mask)
    eces = [e for e in per_class_ece(np.asarray(targets), apply_isotonic(logits, iso), np.asarray(mask), n_bins) if e is not None]
    report["isotonic (comparison only)"] = float(np.mean(eces)) if eces else None
    return {"method": best[0], "parameters": best[1], "mean_ece_by_method": report}


def choose_thresholds(y, probabilities, mask, target_sensitivity=0.9):
    y, p, m = map(np.asarray, (y, probabilities, mask))
    result = []
    for k in range(y.shape[1]):
        valid = (m[:, k] > 0) & (y[:, k] == 1)
        result.append(float(np.quantile(p[valid, k], 1 - target_sensitivity)) if valid.any() else 0.5)
    return result
