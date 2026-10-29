"""Evaluation harness producing metrics, bootstrap intervals, calibration, and abstention curves."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg.eval.metrics import evaluate_arrays, patient_bootstrap
from ecg.eval.calibration import calibrate, fit_temperature
from ecg.eval.uncertainty import per_class_ece, brier_per_class, abstention_curve, sample_correct, sample_confidence


def predict_logits(model, dataset, device, batch_size: int = 64):
    """Run model inference over dataset, returning (logits, targets, masks) as float32 numpy arrays."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_list, targets_list, masks_list = [], [], []
    with torch.no_grad():
        for x, lm, y, m in loader:
            x, lm = x.to(device), lm.to(device)
            out = model(x, lm)
            # Support models returning (logits, features) tuple or just logits
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits_list.append(logits.detach().cpu().numpy())
            targets_list.append(y.numpy())
            masks_list.append(m.numpy())
    return (np.concatenate(logits_list, axis=0),
            np.concatenate(targets_list, axis=0),
            np.concatenate(masks_list, axis=0))


def evaluate(model, dataset, config=None, thresholds=None, temperatures=None, device=None, n_boot: int = 1000, seed: int = 42):
    """Single comprehensive evaluation function matching the PRD specification."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 64
    if config and hasattr(config, "train") and hasattr(config.train, "batch_size"):
        batch_size = config.train.batch_size
    if config and hasattr(config, "eval") and hasattr(config.eval, "n_boot"):
        n_boot = config.eval.n_boot

    logits, targets, masks = predict_logits(model, dataset, device, batch_size=batch_size)
    
    # Uncalibrated probabilities
    probs_uncal = 1.0 / (1.0 + np.exp(-logits))
    
    # Calibrated probabilities
    if temperatures is not None:
        probs = calibrate(logits, temperatures)
    else:
        probs = probs_uncal
        
    num_classes = targets.shape[1]
    if thresholds is None:
        thresholds = [0.5] * num_classes

    # Standard metrics
    metrics = evaluate_arrays(targets, probs, masks, thresholds=thresholds)
    
    # ECE and Brier score
    ece_uncal = per_class_ece(targets, probs_uncal, masks)
    ece_cal = per_class_ece(targets, probs, masks)
    brier = brier_per_class(targets, probs, masks)
    
    valid_eces = [e for e in ece_cal if e is not None]
    metrics["macro_ece"] = float(np.mean(valid_eces)) if valid_eces else None
    
    for k in range(num_classes):
        metrics["classes"][k]["ece_uncalibrated"] = ece_uncal[k]
        metrics["classes"][k]["ece_calibrated"] = ece_cal[k]
        metrics["classes"][k]["brier_score"] = brier[k]
        metrics["classes"][k]["threshold"] = thresholds[k]

    # Patient-level bootstrap CI for macro AUROC
    if hasattr(dataset, "patient_ids"):
        pids = dataset.patient_ids()
        ci = patient_bootstrap(targets, probs, masks, pids, n_boot=n_boot, seed=seed)
        metrics["macro_auroc_ci95"] = ci
    else:
        metrics["macro_auroc_ci95"] = [None, None]

    # Abstention curve
    correct = sample_correct(targets, probs, masks, thresholds)
    conf = sample_confidence(probs, thresholds)
    metrics["abstention_curve"] = abstention_curve(conf, correct)

    return metrics
