# Model Card: ECG Abnormality Classifier (v1.0.0)

## Model Details
- **Architecture:** 1D ResNet with stem, residual blocks, global temporal pooling, and linear classification head.
- **Inputs:** `signal` (`float32 [B, 12, 1000]`), `lead_mask` (`float32 [B, 12]`)
- **Outputs:** `logits` (`float32 [B, 5]`), `features` (`float32 [B, 512, 32]`)
- **Classes:** NORM (Normal ECG), MI (Myocardial Infarction), STTC (ST/T Change), CD (Conduction Disturbance), HYP (Hypertrophy).

## Performance Summary (PTB-XL Test Fold)
- **Macro AUROC:** 0.9112 (95% CI: 0.9045 - 0.9194)
- **Macro AUPRC:** 0.7768
- **Macro ECE (Calibrated):** 0.0225

## Verification & Parity
- **ONNX Runtime Parity:** Max absolute error = 4.77e-06 (tolerance: 0.0001)
- **Calibration Method:** Per-class temperature scaling fitted on validation fold.
- **Abstention Policy:** Rejects inputs with quality < 0.25; flags predictions as uncertain if probability is within 0.05 of decision threshold.
