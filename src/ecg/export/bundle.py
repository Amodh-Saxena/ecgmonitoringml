"""Export, parity check and validation for the service hand-off bundle."""
import json
import subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

from ecg.eval.harness import predict_logits, evaluate
from ecg.eval.calibration import fit_temperature, choose_thresholds

REQUIRED = ("model.onnx", "preprocess.json", "labels.json", "calibration.json", "thresholds.json",
            "abstention.json", "head_weights.npy", "manifest.json", "MODEL_CARD.md")


def export_onnx(model, path, sample_signal, sample_mask, opset=17):
    """Export model to ONNX with dynamic batch size matching inference contract."""
    model.eval()
    torch.onnx.export(
        model,
        (sample_signal, sample_mask),
        str(path),
        input_names=["signal", "lead_mask"],
        output_names=["logits", "features"],
        dynamic_axes={
            "signal": {0: "batch"},
            "lead_mask": {0: "batch"},
            "logits": {0: "batch"},
            "features": {0: "batch"}
        },
        opset_version=opset,
        dynamo=False
    )


def check_onnx_parity(model, path, signal, mask, atol=1e-4):
    """Verify PyTorch and ONNX Runtime logits match within strict tolerance (1e-4)."""
    import onnxruntime as ort
    model.eval()
    with torch.no_grad():
        expected = model(signal, mask)[0].cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {
        "signal": signal.cpu().numpy(),
        "lead_mask": mask.cpu().numpy()
    })[0]
    error = float(np.max(np.abs(expected - actual)))
    if error > atol:
        raise ValueError(f"ONNX parity failed: max absolute error {error:g} > {atol:g}")
    return error


def validate_bundle(path):
    """Validate that bundle directory complies with product PRD requirements."""
    root = Path(path)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing files: {missing}")
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    thresholds = json.loads((root / "thresholds.json").read_text(encoding="utf-8"))
    if len(labels["classes"]) != len(thresholds["thresholds"]):
        raise ValueError("Class/threshold count mismatch")
    return True


def create_release_bundle(model, cache_path, bundle_dir="bundle/v1.0.0", opset=17, atol=1e-4):
    """Full hand-off bundle builder fulfilling the PRD export contract."""
    out_dir = Path(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_path)
    
    from ecg.datasets.build import ECGArrayDataset
    val_ds = ECGArrayDataset(cache, split="validation")
    test_ds = ECGArrayDataset(cache, split="test")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # 1. Validation inference for calibration & thresholds
    print("Fitting temperature scaling and thresholds on validation split...")
    z_val, y_val, m_val = predict_logits(model, val_ds, device, batch_size=64)
    temps = fit_temperature(z_val, y_val, m_val, per_class=True)
    p_val_cal = 1.0 / (1.0 + np.exp(-z_val / np.array(temps)))
    thresholds = choose_thresholds(y_val, p_val_cal, m_val, target_sensitivity=0.90)
    
    # 2. Write calibration.json, thresholds.json, abstention.json
    (out_dir / "calibration.json").write_text(json.dumps({
        "method": "temperature_scaling_per_class",
        "temperatures": temps
    }, indent=2), encoding="utf-8")
    
    (out_dir / "thresholds.json").write_text(json.dumps({
        "criterion": "target_sensitivity_0.90",
        "thresholds": thresholds
    }, indent=2), encoding="utf-8")
    
    (out_dir / "abstention.json").write_text(json.dumps({
        "quality_floor": 0.25,
        "confidence_floor": 0.05,
        "disagreement_limit": None
    }, indent=2), encoding="utf-8")
    
    # 3. Write preprocess.json and labels.json
    pre_spec = manifest["metadata"]["preprocess"]
    (out_dir / "preprocess.json").write_text(json.dumps(pre_spec, indent=2), encoding="utf-8")
    
    from ecg.labels.mapping import DEFAULT_LABELS
    labels_content = DEFAULT_LABELS.read_text(encoding="utf-8")
    (out_dir / "labels.json").write_text(labels_content, encoding="utf-8")
    
    # 4. Save head_weights.npy for CAM
    head_weights = model.head.weight.detach().cpu().numpy()
    np.save(out_dir / "head_weights.npy", head_weights)
    
    # 5. Export model.onnx
    onnx_path = out_dir / "model.onnx"
    sample_signal = torch.zeros(1, 12, 1000, dtype=torch.float32)
    sample_mask = torch.ones(1, 12, dtype=torch.float32)
    print(f"Exporting ONNX graph to {onnx_path} (opset={opset})...")
    export_onnx(model, onnx_path, sample_signal, sample_mask, opset=opset)
    
    # 6. Parity Check
    print("Running ONNX Runtime parity test on test sample...")
    test_signals = torch.from_numpy(test_ds.signals[:16].copy())
    test_masks = torch.from_numpy(test_ds.lead_masks[:16].copy())
    max_err = check_onnx_parity(model, onnx_path, test_signals, test_masks, atol=atol)
    print(f"ONNX parity PASSED: max error = {max_err:.2e} (< {atol:.2e})")
    
    # 7. Evaluate on Test split for manifest & model card
    print("Evaluating test metrics for bundle manifest...")
    test_metrics = evaluate(model, test_ds, device=device, thresholds=thresholds, temperatures=temps, n_boot=200)
    
    # 8. Write manifest.json
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_hash = "local-release"
        
    bundle_manifest = {
        "bundle_version": out_dir.name,
        "manifest_hash": manifest.get("manifest_hash", cache.name),
        "git_commit": git_hash,
        "input_contract": {
            "signal": "float32 [B, 12, 1000]",
            "lead_mask": "float32 [B, 12]"
        },
        "output_contract": {
            "logits": "float32 [B, 5]",
            "features": "float32 [B, 512, 32]"
        },
        "test_metrics": {
            "macro_auroc": test_metrics["macro_auroc"],
            "macro_auroc_ci95": test_metrics["macro_auroc_ci95"],
            "macro_auprc": test_metrics["macro_auprc"],
            "macro_ece": test_metrics["macro_ece"]
        },
        "license": "PhysioNet Open Data / MIT Code License"
    }
    (out_dir / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")
    
    # 9. Write MODEL_CARD.md
    model_card = f"""# Model Card: ECG Abnormality Classifier ({out_dir.name})

## Model Details
- **Architecture:** 1D ResNet with stem, residual blocks, global temporal pooling, and linear classification head.
- **Inputs:** `signal` (`float32 [B, 12, 1000]`), `lead_mask` (`float32 [B, 12]`)
- **Outputs:** `logits` (`float32 [B, 5]`), `features` (`float32 [B, 512, 63]`)
- **Classes:** NORM (Normal ECG), MI (Myocardial Infarction), STTC (ST/T Change), CD (Conduction Disturbance), HYP (Hypertrophy).

## Performance Summary (PTB-XL Test Fold)
- **Macro AUROC:** {test_metrics['macro_auroc']:.4f} (95% CI: {test_metrics['macro_auroc_ci95'][0]:.4f} - {test_metrics['macro_auroc_ci95'][1]:.4f})
- **Macro AUPRC:** {test_metrics['macro_auprc']:.4f}
- **Macro ECE (Calibrated):** {test_metrics['macro_ece']:.4f}

## Verification & Parity
- **ONNX Runtime Parity:** Max absolute error = {max_err:.2e} (tolerance: {atol:g})
- **Calibration Method:** Per-class temperature scaling fitted on validation fold.
- **Abstention Policy:** Rejects inputs with quality < 0.25; flags predictions as uncertain if probability is within 0.05 of decision threshold.
"""
    (out_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    
    # 10. Validate Bundle Completeness
    validate_bundle(out_dir)
    print(f"Bundle successfully created and validated at: {out_dir}")
    return out_dir
