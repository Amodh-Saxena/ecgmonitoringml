"""Inference pipeline: preprocess with existing model code, run ONNX session, calibrate, and compute CAM."""
from typing import List, Tuple
import numpy as np

from ecg.preprocess.signal import preprocess, PreprocessConfig
from ecg.eval.quality import quality_score
from .model_loader import ModelBundle
from .schemas import PredictResponse, ClassResult, SignalResult, AttentionResult


def run_pipeline(
    raw_signal: np.ndarray,
    lead_names: List[str],
    source_fs: int,
    units: str,
    bundle: ModelBundle
) -> PredictResponse:
    """Execute complete end-to-end inference using model bundle artifacts."""
    
    # 1. Preprocess using existing model library code (P1 requirement)
    prep_cfg = PreprocessConfig(
        sampling_rate=bundle.target_fs,
        duration_seconds=bundle.target_duration,
        low_hz=bundle.prep_config.get("low_hz", 0.5),
        high_hz=bundle.prep_config.get("high_hz", 40.0),
        notch_hz=bundle.prep_config.get("notch_hz", None),
        normalization=bundle.prep_config.get("normalization", "record_zscore"),
        epsilon=bundle.prep_config.get("epsilon", 1e-6),
    )
    
    x_pre, lead_mask = preprocess(raw_signal, source_fs, lead_names, prep_cfg)
    leads_used = int(np.sum(lead_mask > 0))

    # 2. Quality scoring
    q_score = float(quality_score(x_pre, lead_mask))

    # Signal payload for UI plotting (filtered, canonical leads, original scale units)
    # The preprocessed signal is 100 Hz, shape [12, 1000]
    signal_result = SignalResult(
        fs=bundle.target_fs,
        unit=units if units else "mV",
        leads=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        samples=[[round(float(v), 4) for v in trace] for trace in x_pre]
    )

    # 3. Quality floor rejection check
    reasons: List[str] = []
    if q_score < bundle.quality_floor:
        reasons.append(
            f"Signal quality score ({q_score:.2f}) is below acceptable floor ({bundle.quality_floor:.2f}) due to flatline or severe noise"
        )
        return PredictResponse(
            schema_version="1",
            model_version=bundle.model_version,
            manifest_hash=bundle.manifest_hash,
            status="rejected",
            reasons=reasons,
            quality_score=round(q_score, 3),
            leads_used=leads_used,
            classes=[],
            signal=signal_result,
            attention=None
        )

    # 4. ONNX Runtime inference
    feed = {
        bundle.input_names[0]: x_pre[None, :, :].astype(np.float32),
        bundle.input_names[1]: lead_mask[None, :].astype(np.float32)
    }
    outputs = bundle.session.run(None, feed)
    logits = outputs[0][0]  # [5]
    features = outputs[1][0] if len(outputs) > 1 else None  # [512, 32]

    # 5. Temperature scaling calibration
    T = bundle.temperatures
    scaled_logits = logits / np.maximum(T, 1e-4)
    probs = 1.0 / (1.0 + np.exp(-scaled_logits))

    # 6. Confidence margin & Status
    margins = np.abs(probs - bundle.thresholds)
    min_margin = float(np.min(margins))
    
    if min_margin < bundle.confidence_floor:
        pred_status = "uncertain"
        reasons.append("Prediction confidence is near the decision threshold; manual clinician review is recommended")
    else:
        pred_status = "confident"

    # 7. Class results with threshold comparison
    class_results: List[ClassResult] = []
    for k in range(len(bundle.classes)):
        p_val = float(probs[k])
        thr_val = float(bundle.thresholds[k])
        flagged = bool(p_val >= thr_val)
        class_results.append(
            ClassResult(
                code=bundle.class_codes[k],
                name=bundle.class_names[k],
                probability=round(p_val, 4),
                threshold=round(thr_val, 4),
                flagged=flagged
            )
        )

    # 8. Gradient-free Class Activation Mapping (CAM) attention
    attention_result = None
    if features is not None:
        # features shape: [512, 32], head_weights shape: [5, 512]
        cam_per_class: List[List[float]] = []
        target_len = x_pre.shape[1]  # 1000
        time_x_orig = np.arange(features.shape[1])
        time_x_target = np.linspace(0, features.shape[1] - 1, target_len)

        for k in range(len(bundle.classes)):
            w = bundle.head_weights[k]  # [512]
            # Linear projection: sum over feature channels
            cam_1d = np.maximum((w[:, None] * features).sum(axis=0), 0.0)  # [32]
            # Interpolate to signal length
            cam_interp = np.interp(time_x_target, time_x_orig, cam_1d)
            peak = float(cam_interp.max())
            if peak > 1e-6:
                cam_norm = cam_interp / peak
            else:
                cam_norm = cam_interp
            cam_per_class.append([round(float(v), 3) for v in cam_norm])

        attention_result = AttentionResult(
            method="cam",
            per_class=cam_per_class
        )

    return PredictResponse(
        schema_version="1",
        model_version=bundle.model_version,
        manifest_hash=bundle.manifest_hash,
        status=pred_status,
        reasons=reasons,
        quality_score=round(q_score, 3),
        leads_used=leads_used,
        classes=class_results,
        signal=signal_result,
        attention=attention_result
    )
