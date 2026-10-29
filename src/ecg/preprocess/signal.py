"""Deterministic ECG preprocessing and seeded training-only augmentation."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, iirnotch, filtfilt
from math import gcd

CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class PreprocessConfig:
    sampling_rate: int = 100
    duration_seconds: int = 10
    low_hz: float = 0.5
    high_hz: float = 40.0
    notch_hz: float | None = None
    normalization: str = "record_zscore"
    epsilon: float = 1e-6


def preprocess(signal: np.ndarray, source_rate: int, lead_names: list[str] | tuple[str, ...],
               config: PreprocessConfig = PreprocessConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Convert [samples, leads] input to float32 [12, T] and a [12] lead mask."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or source_rate <= 0 or len(lead_names) != x.shape[1]:
        raise ValueError("Expected [samples, leads] signal and matching lead names")
    length = config.sampling_rate * config.duration_seconds
    out = np.zeros((12, length), dtype=np.float32)
    mask = np.zeros(12, dtype=np.float32)
    available = {str(n).strip().upper(): i for i, n in enumerate(lead_names)}
    for j, lead in enumerate(CANONICAL_LEADS):
        if lead.upper() not in available:
            continue
        values = x[:, available[lead.upper()]]
        if source_rate != config.sampling_rate:
            common = gcd(source_rate, config.sampling_rate)
            values = resample_poly(values, config.sampling_rate // common, source_rate // common)
        nyquist = config.sampling_rate / 2
        hi = min(config.high_hz, nyquist * 0.95)
        if config.low_hz > 0 and config.low_hz < hi:
            values = sosfiltfilt(butter(2, [config.low_hz, hi], btype="bandpass", fs=config.sampling_rate, output="sos"), values)
        if config.notch_hz and config.notch_hz < nyquist:
            b, a = iirnotch(config.notch_hz, 30, fs=config.sampling_rate)
            values = filtfilt(b, a, values)
        values = values[:length]
        if config.normalization == "record_zscore" and values.size:
            values = (values - values.mean()) / max(values.std(), config.epsilon)
        elif config.normalization != "none":
            raise ValueError(f"Unsupported normalization: {config.normalization}")
        out[j, :len(values)] = values.astype(np.float32)
        mask[j] = 1.0
    return out, mask


