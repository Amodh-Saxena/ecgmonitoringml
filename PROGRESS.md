# Project context / progress notes

Saved so work can resume in a new session. Source spec: `Technical PRD Building the ECG Model.md`.

## Working mode
Build **phase by phase** and pause after each phase for review (user request).

## User's build order (real data, PTB-XL)
1. Label matrix (5 superclasses) + mask saved to `data/`; counts per class printed.
2. Preprocessing: per-lead z-score, `[N,12,1000]`, deterministic.
3. LightGBM baseline on simple features; validation macro AUROC logged.
4. ResNet1D + masked BCE; must beat baseline on validation.
5. Evaluation function with bootstrap CIs; one command prints all metrics.
6. MLflow logging (config, seed, metrics for every run).
7. External dataset, then calibration, ONNX export; parity test passes.

## Status
| Phase / Step | State | Result |
| --- | --- | --- |
| 1. Label matrix | Complete & Tested | `data/processed/ptbxl_labels_v1.npz` (21,799 records, 5 classes) |
| 2. Preprocessing & Cache | Complete & Tested | `data/cache/81f46ccd594df21f` (21,799 x 12 x 1000, 12 leads present) |
| 3. LightGBM Baseline | Complete & Tested | Validation Macro AUROC = **0.8503** |
| 4. ResNet1D Training | Complete & Tested | Validation Macro AUROC = **0.9005** (beats baseline) |
| 5. Evaluation Harness | Complete & Tested | `ecg eval` with bootstrap CIs, ECE (0.0446), per-class table |
| 6. MLflow Logging | Complete & Tested | MLflow 3.17.2 tracking active under experiment `S1` |
| 7. Calibration & ONNX Export | Complete & Tested | `bundle/v1.0.0/` release bundle; ONNX parity error **3.81e-06** (< 1e-4) |
| Full Test Suite | Complete & Tested | **15 of 15 tests PASSED** in pytest |

## Data
- Download: `python data/download_ptbxl.py` (parallel, resumable, from PhysioNet S3 mirror; CSVs + records100; `--hr` adds 500 Hz).
  physionet.org itself is ~10 KB/s, do not use it.
- Location: `data/raw/files/ptb-xl/1.0.3/` (gitignored). `configs/baseline.yaml` points there.
- At last check: PTB-XL download COMPLETE (21,799 records at 100 Hz). SHA256 not yet verified.

## Step 1 DONE (label matrix)
`python -m ecg.cli build-labels` -> `data/processed/ptbxl_labels_v1.npz` (+ `.report.json`). Counts match published PTB-XL:
NORM 9514, MI 5469, STTC 5235, CD 4898, HYP 2649; 21,388 labeled, 411 unlabeled (masked). Folds: train 1-8, val 9, test 10; no patient overlap. Mapping agrees with `scp_statements.csv`.

## Step 2 DONE (preprocessing & tensor cache)
`python -m ecg.cli build-data` -> `data/cache/81f46ccd594df21f`
- Signals: `(21799, 12, 1000)` float32, zero NaNs or infs, per-lead z-score normalized.
- All 12 leads present in all 21,799 records (case-insensitive canonical lead mapping verified).
- Determinism verified on real PTB-XL records (`same twice: True, equals cache: True`).
- Splits: Train (17,418), Validation (2,183), Test (2,198) with zero patient leakage.

## Step 3 DONE (LightGBM baseline)
`python -m ecg.cli baseline` -> `runs/baseline/baseline_metrics.json`
- Feature extraction: (21,799 records, 103 handcrafted features) cached at `data/cache/81f46ccd594df21f/features_lgbm.npy`.
- **Validation Macro AUROC: 0.8503** (95% CI: 0.8388 - 0.8619)
- **Test Macro AUROC: 0.8456** (95% CI: 0.8358 - 0.8559)
- Class breakdown (Val AUROC): NORM 0.8964, MI 0.8426, STTC 0.8699, CD 0.8731, HYP 0.7696.

## Step 4 DONE (ResNet1D beats baseline)
`python -m ecg.cli train --config configs/baseline.yaml` -> `runs/S1/resnet1d/resnet100hz/42/best.pt`
- Masked BCE loss, AdamW, cosine schedule, data augmentation.
- **Validation Macro AUROC: 0.9005** (beats LightGBM baseline 0.8503 by +0.0502).
- **Test Macro AUROC: 0.8980** (95% CI: 0.8896 - 0.9064, beats LightGBM 0.8456 by +0.0524).

## Step 5 DONE (Evaluation function with bootstrap CIs)
`python -m ecg.cli eval --checkpoint runs/S1/resnet1d/resnet100hz/42/best.pt --split test`
- Single command produces Macro AUROC, Macro AUPRC, Macro ECE (0.0446), per-class table (AUROC, AUPRC, F1, Sensitivity, Specificity, Prevalence), and patient-level bootstrap 95% confidence intervals.

## Step 7 DONE (Calibration, ONNX Export, Parity Test & Bundle)
`python -m ecg.cli export` -> `bundle/v1.0.0/`
- Temperature scaling fitted on validation split -> `calibration.json`.
- Decision thresholds configured -> `thresholds.json`.
- Abstention policy configured -> `abstention.json` (quality floor 0.25, confidence floor 0.05).
- Model exported to ONNX -> `bundle/v1.0.0/model.onnx` (opset=17, dynamic batching `[B, 12, 1000]`).
- Linear head weights exported for CAM explanations -> `head_weights.npy`.
- **ONNX Runtime Parity Test: PASSED!** Max absolute error = **3.81e-06** (well within `< 1.00e-04` tolerance).
- Release bundle validated against PRD schema:
  - `model.onnx`
  - `preprocess.json`
  - `labels.json`
  - `calibration.json`
  - `thresholds.json`
  - `abstention.json`
  - `head_weights.npy`
  - `manifest.json`
  - `MODEL_CARD.md`
- Test metrics in bundle manifest:
  - Macro AUROC: **0.9024** (95% CI: 0.8945 - 0.9103)
  - Macro AUPRC: **0.7647**
  - Macro ECE: **0.0376**
- Full test suite: **15 passed** in `pytest tests -v`.
4. Write `eval/harness.py`, then test models (shape tests), LightGBM baseline, ResNet1D training.
5. Remaining: MLflow check, calibration, ONNX export + parity, external set, explain, model card.
