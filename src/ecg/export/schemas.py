"""Runtime checks for inference contract and version compatibility."""
import json
from pathlib import Path


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def validate_inference_request(signal, lead_mask, sampling_rate, duration_seconds, n_classes):
    if signal.dtype.name != "float32" or signal.ndim != 3 or signal.shape[1] != 12:
        raise ValueError("signal must be float32 [B,12,T]")
    if lead_mask.shape != signal.shape[:2] or lead_mask.dtype.name != "float32":
        raise ValueError("lead_mask must be float32 [B,12]")
    if signal.shape[-1] != sampling_rate * duration_seconds:
        raise ValueError("Signal time dimension disagrees with preprocessing contract")
    if n_classes < 1:
        raise ValueError("At least one output class is required")
