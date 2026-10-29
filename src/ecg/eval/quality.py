"""Transparent signal-quality features for abstention decisions."""
import numpy as np


def quality_score(signal, lead_mask):
    """Return a bounded heuristic score using flatness, clipping, noise level and lead coverage.

    Noise is the share of signal energy in the first difference (high-frequency content) of
    z-scored leads. This is an engineering baseline, not a clinically validated quality classifier.
    """
    x, present = np.asarray(signal), np.asarray(lead_mask) > 0
    if x.ndim != 2 or present.shape != (x.shape[0],):
        raise ValueError("Expected signal [leads, time] and lead_mask [leads]")
    active = x[present]
    if not active.size:
        return 0.0
    std = np.std(active, axis=-1)
    flat_fraction = float(np.mean(std < 1e-5))
    clipping_fraction = float(np.mean(np.abs(active) > 10.0))
    coverage = float(present.mean())
    usable = active[std >= 1e-5]
    if usable.size:
        z = (usable - usable.mean(-1, keepdims=True)) / usable.std(-1, keepdims=True)
        noise = float(np.clip(np.mean(np.diff(z, axis=-1) ** 2) / 2.0, 0, 1))  # ~0 smooth, ~1 white noise
    else:
        noise = 1.0
    return float(np.clip(coverage * (1 - flat_fraction) * (1 - clipping_fraction) * (1 - noise ** 2), 0, 1))


def confidence_margin(probabilities, thresholds) -> float:
    """Predictive confidence: smallest distance of a calibrated probability from its class threshold."""
    return float(np.min(np.abs(np.asarray(probabilities) - np.asarray(thresholds))))


def prediction_status(probabilities, thresholds, quality, quality_floor=0.25, confidence_floor=0.05,
                      disagreement=None, disagreement_limit=None):
    """rejected if quality below floor; uncertain if confidence or disagreement crosses its limit."""
    if quality < quality_floor:
        return "rejected"
    if confidence_margin(probabilities, thresholds) < confidence_floor:
        return "uncertain"
    if disagreement is not None and disagreement_limit is not None and np.max(disagreement) > disagreement_limit:
        return "uncertain"
    return "confident"
