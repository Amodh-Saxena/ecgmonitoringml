"""Seeded, training-only augmentation. Every transform has a probability and range set in config.

Callers must restrict this to training data (ECGArrayDataset only enables it for the train split).
"""
import numpy as np


def _fire(rng, p):
    return rng.random() < p


def augment(signal, lead_mask, rng: np.random.Generator, fs: int = 100,
            noise_std: float = 0.01, noise_p: float = 1.0,
            wander_amp: float = 0.0, wander_p: float = 0.0,
            powerline_amp: float = 0.0, powerline_hz: float = 50.0, powerline_p: float = 0.0,
            scale_range: tuple[float, float] = (0.9, 1.1), scale_p: float = 1.0,
            max_shift: int = 20, shift_p: float = 1.0,
            crop_min_fraction: float = 1.0, crop_p: float = 0.0,
            lead_dropout_p: float = 0.0,
            jitter: float = 0.0, jitter_p: float = 0.0,
            time_mask_len: int = 0, time_mask_p: float = 0.0):
    x = np.array(signal, dtype=np.float32, copy=True)
    mask = np.array(lead_mask, dtype=np.float32, copy=True)
    n = x.shape[-1]
    t = np.arange(n, dtype=np.float32) / fs
    if jitter > 0 and _fire(rng, jitter_p):  # small resampling jitter (time-axis stretch)
        factor = 1.0 + rng.uniform(-jitter, jitter)
        src = np.clip(np.arange(n) * factor, 0, n - 1)
        x = np.stack([np.interp(src, np.arange(n), row) for row in x]).astype(np.float32)
    if crop_p > 0 and _fire(rng, crop_p):  # random crop, stretched back to length
        keep = int(n * rng.uniform(crop_min_fraction, 1.0))
        start = int(rng.integers(0, n - keep + 1))
        src = np.linspace(start, start + keep - 1, n)
        x = np.stack([np.interp(src, np.arange(n), row) for row in x]).astype(np.float32)
    if noise_std > 0 and _fire(rng, noise_p):
        x += rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    if wander_amp > 0 and _fire(rng, wander_p):  # low-frequency sinusoid
        f, ph = rng.uniform(0.05, 0.5), rng.uniform(0, 2 * np.pi)
        x += (wander_amp * rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * f * t + ph)).astype(np.float32)[None, :]
    if powerline_amp > 0 and _fire(rng, powerline_p) and powerline_hz < fs / 2:
        x += (powerline_amp * np.sin(2 * np.pi * powerline_hz * t + rng.uniform(0, 2 * np.pi))).astype(np.float32)[None, :]
    if _fire(rng, scale_p):
        x *= rng.uniform(*scale_range, size=(x.shape[0], 1)).astype(np.float32)
    if max_shift and _fire(rng, shift_p):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        x = np.roll(x, shift, axis=-1)
        if shift > 0:
            x[:, :shift] = 0
        elif shift < 0:
            x[:, shift:] = 0
    if time_mask_len > 0 and _fire(rng, time_mask_p):
        start = int(rng.integers(0, max(n - time_mask_len, 1)))
        x[:, start:start + time_mask_len] = 0
    dropped = (rng.random(mask.shape) < lead_dropout_p) & (mask > 0)
    if dropped.all() or (mask > 0).sum() - dropped.sum() < 1:  # always keep at least one lead
        dropped[np.argmax(mask)] = False
    mask[dropped] = 0
    x[mask == 0] = 0
    return x, mask
