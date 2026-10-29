# Production & Verification Report: ECG Abnormality Classifier

**Date:** October 2026  
**Pipeline Version:** `v1.0.0`  
**Dataset:** PTB-XL (PhysioNet v1.0.3, 21,799 records)  
**Hand-off Release Bundle:** `bundle/v1.0.0/`  

---

## 1. Executive Summary

This document serves as the comprehensive audit, verification, training log record, and replication guide for the 12-lead ECG Abnormality Classifier built strictly according to the **Technical PRD: Building the ECG Model**.

All 7 core stages have been completed, verified against real ECG recordings, and validated through unit and parity test suites:
1. **Data Acquisition & Ingestion:** 43,606 raw files (all 21,799 100 Hz records) downloaded from the open PhysioNet S3 mirror using a 16-worker parallel downloader.
2. **Label Harmonization & Matrix:** 5 diagnostic superclasses mapped, masked unknown labels, zero patient leakage across folds.
3. **Deterministic Preprocessing:** 21,799 12-lead records converted to `[21799, 12, 1000]` float32 tensors with per-lead z-score normalization and 100% 12-lead verification.
4. **Classical Baseline (LightGBM):** 103 clinical/statistical features extracted per record; achieved **0.8503 Validation Macro AUROC** (95% CI: 0.8388 - 0.8619).
5. **Deep Architecture (ResNet1D):** Trained with masked BCE loss, AdamW, and cosine decay; reached **0.9109 Validation Macro AUROC** (beating the baseline by **+0.0606**).
6. **Evaluation Harness:** Evaluated on holdout in-domain test fold (fold 10, 2,198 records); achieved **0.9112 Macro AUROC** (95% Bootstrap CI: **0.9045 - 0.9194**), **0.7768 Macro AUPRC**, and calibrated ECE = **0.0225** (well below the 0.05 target).
7. **Experiment Tracking:** MLflow tracking initialized under experiment `S1`, recording provenance hashes, seeds, and metrics.
8. **Release Bundle & ONNX Parity:** Model exported to ONNX (opset 17, dynamic batching). ONNX Runtime numerical parity test achieved max absolute error of **4.77e-06** (well below the `< 1e-4` threshold). Full release bundle assembled and validated in `bundle/v1.0.0/`.
9. **Test Suite:** **15 of 15 tests PASSED** in pytest.

---

## 2. Stage 1: Data Acquisition & Download Process

### 2.1 Download Strategy & Challenge
- **Source Dataset:** PhysioNet PTB-XL dataset (version 1.0.3, DOI: 10.13026/kfzx-aw45).
- **Challenge:** Downloading directly from `https://physionet.org/files/ptb-xl/1.0.3/` or `get-zip` crawled at ~10 KB/s, which would take over 45 hours.
- **Solution:** Designed and executed a custom multithreaded downloader (`data/download_ptbxl.py`) targeting the public AWS S3 mirror hosted by PhysioNet:
  `https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/`
- **Architecture of Downloader:**
  - 16 parallel worker threads using `concurrent.futures.ThreadPoolExecutor`.
  - HTTP Range requests for partial content and byte-resume capability.
  - Automatic directory creation and streaming in 64 KB chunks.
  - Automated extraction and validation of both metadata CSVs and all record files.

### 2.2 Download Execution Log Summary
```
Scraping index from PhysioNet S3 mirror: https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/
Identified target files:
  - ptbxl_database.csv (metadata, demographics, folds, scp statements)
  - scp_statements.csv (diagnostic taxonomy and superclass mappings)
  - records100/00000/ to records100/21000/ (.hea headers and .dat binary traces)
Total files identified: 43,606 files (~0.51 GB)
Starting parallel download with 16 threads...
[100%] 43,606 / 43,606 files downloaded successfully.
All files verified on disk at: data/raw/files/ptb-xl/1.0.3/
```

### 2.3 Storage Layout
```
data/raw/files/ptb-xl/1.0.3/
├── ptbxl_database.csv           (Metadata, patient IDs, diagnostics, folds 1-10)
├── scp_statements.csv           (Diagnostic codes, superclasses, weights)
└── records100/                  (21,799 WFDB 100 Hz records)
    ├── 00000/
    │   ├── 00001_lr.hea & 00001_lr.dat
    │   └── ...
    └── 21000/
        └── 21837_lr.hea & 21837_lr.dat
```

---

## 3. Stage 2: Label Harmonization & Label Matrix

### 3.1 Mapping Rules & Taxonomy
Class set v1 maps source diagnostic statements to the 5 canonical PTB-XL superclasses:
- `NORM`: Normal ECG
- `MI`: Myocardial Infarction
- `STTC`: ST/T Change
- `CD`: Conduction Disturbance
- `HYP`: Hypertrophy

**Masking Rule:** Target labels are stored as pairs of vectors $(y, m)$ of length $K=5$:
- If a diagnosis is present and mapped: $y_k = 1$, $m_k = 1$.
- If covered by taxonomy but not present: $y_k = 0$, $m_k = 1$.
- If a record has unmapped statements or no statements: $m_k = 0$ for all classes (masked out of loss). Unmapped classes are never treated as silent negatives.

### 3.2 Label Matrix Generation & Leakage Audit
Command:
```powershell
python -m ecg.cli build-labels
```

Output Artifacts: `data/processed/ptbxl_labels_v1.npz` and `ptbxl_labels_v1.report.json`

**Audit Log:**
```
Mapping vs dataset statement table: OK (zero discrepancies)
Records: 21,799
  Labeled records:       21,388 (98.1%)
  Unlabeled (masked):       411 (1.9%)
  Multi-label records:    5,144 (23.6%)

Positives Per Class:
  NORM:  9,514  (train=7596, validation=955, test=963)
  MI:    5,469  (train=4379, validation=540, test=550)
  STTC:  5,235  (train=4186, validation=528, test=521)
  CD:    4,898  (train=3907, validation=495, test=496)
  HYP:   2,649  (train=2119, validation=268, test=262)

Split Sizes:
  Train (Folds 1-8):      17,418 records
  Validation (Fold 9):     2,183 records
  Test (Fold 10):          2,198 records

Patient Overlap Audit:
  train & validation:  0
  train & test:        0
  validation & test:   0
Zero patient leakage verified across all splits.
```

---

## 4. Stage 3: Deterministic Preprocessing & Tensor Cache

### 4.1 Signal Processing Specification
- **Input:** Raw signal `[T_raw, 12]` from WFDB reader (`wfdb.rdrecord`).
- **Canonical Lead Order:** `["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]`.
- **Case-Insensitive Match:** Handled case variations (e.g., WFDB `AVR`, `AVL`, `AVF` accurately mapped to `aVR`, `aVL`, `aVF`).
- **Resampling:** Polyphase rational resampling to 100 Hz (`scipy.signal.resample_poly`).
- **Bandpass Filtering:** 4th-order Butterworth bandpass filter from 0.5 Hz to 40.0 Hz (`scipy.signal.sosfiltfilt`).
- **Fixed Windowing:** 10.0 seconds (1000 samples); zero-padded or centrally cropped.
- **Lead Normalization:** Per-lead record z-score: $x_{norm} = \frac{x - \mu}{\max(\sigma, 10^{-6})}$.
- **Lead Mask:** `float32 [12]` indicator tracking valid leads vs zero-padded missing leads.

### 4.2 Cache Build & Verification Log
Command:
```powershell
python -m ecg.cli build-data
```

Cache Location: `data/cache/81f46ccd594df21f/`

**Verification Report:**
```
Cache directory: data/cache/81f46ccd594df21f/
- signals.npy:       (21799, 12, 1000) float32  [104.6 MB]
- lead_masks.npy:    (21799, 12) float32        [1.0 MB]
- targets.npy:       (21799, 5) float32         [436 KB]
- target_masks.npy:  (21799, 5) float32         [436 KB]

Sanity Checks:
- NaN / inf count:   0
- 12-lead coverage:  21,799 of 21,799 records (100.0%)
- Per-lead mean:     ~0.0 (max abs = 3.05e-8)
- Per-lead std:      1.0
- Determinism test:  Identical on repeated processing (same twice = True, equals cache = True)
```

---

## 5. Stage 4: Classical Baseline (LightGBM)

### 5.1 Handcrafted Feature Extraction
Command:
```powershell
python -m ecg.cli baseline
```

Output: `runs/baseline/baseline_metrics.json` and cached `features_lgbm.npy` `(21799, 103)`

**Extracted 103 Features:**
- **Per-Lead Features (8 × 12 = 96 features):** Mean, standard deviation, min, max, peak-to-peak amplitude, skewness, kurtosis, mean absolute first difference.
- **Rhythm & Global Features (7 features):** Heart rate (BPM), mean RR interval, RR interval standard deviation, RMSSD, pNN50, mean R-peak amplitude (Lead II), detected peak count.

### 5.2 Baseline Performance Metrics
Trained 5 independent binary `LGBMClassifier` models (300 estimators, lr=0.05, num_leaves=31).

```
--- LightGBM Baseline Validation Metrics (Fold 9, 2183 records) ---
Macro AUROC: 0.8503  (95% Bootstrap CI: 0.8388 - 0.8619)
Macro AUPRC: 0.6699

Per-Class Validation Metrics:
- NORM:  AUROC = 0.8964 | AUPRC = 0.8546 | F1 = 0.8054 | Sens = 0.8995 | Spec = 0.7322
- MI:    AUROC = 0.8426 | AUPRC = 0.6747 | F1 = 0.5599 | Sens = 0.9000 | Spec = 0.5579
- STTC:  AUROC = 0.8699 | AUPRC = 0.6827 | F1 = 0.6161 | Sens = 0.8996 | Spec = 0.6669
- CD:    AUROC = 0.8731 | AUPRC = 0.7708 | F1 = 0.5358 | Sens = 0.8990 | Spec = 0.5633
- HYP:   AUROC = 0.7696 | AUPRC = 0.3665 | F1 = 0.3013 | Sens = 0.8993 | Spec = 0.4191

--- LightGBM Baseline Test Metrics (Fold 10, 2198 records) ---
Macro AUROC: 0.8456  (95% Bootstrap CI: 0.8358 - 0.8559)
Macro AUPRC: 0.6568
```

**Benchmark Requirement:** Deep learning architecture must exceed **0.8503 Validation Macro AUROC**.

---

## 6. Stage 5: ResNet1D Training & Checkpoints

### 6.1 Model Architecture & Configuration
- **Input Channels:** 24 channels (12 signal leads concatenated with 12 lead mask indicators).
- **Stem:** 1D Conv (kernel 15, stride 2, pad 7) -> BatchNorm1d -> ReLU -> MaxPool1d (kernel 3, stride 2, pad 1).
- **Residual Backbone:** 4 stages with channel widths `[64, 128, 256, 512]` and 2 residual blocks per stage. Each block uses kernel size 7 with residual skip connections.
- **Temporal Downsampling:** Downsampled by a factor of 32 ($\lfloor 1000 / 32 \rfloor = 32$). Feature map output is `float32 [B, 512, 32]`.
- **Classification Head:** Global Average Pooling over temporal dimension -> Dropout (0.1) -> Linear (512 -> 5).
- **Loss:** Masked Binary Cross-Entropy with Logits:
  $$\mathcal{L} = \frac{\sum_{b,k} m_{b,k} \cdot \text{BCE}(z_{b,k}, y_{b,k})}{\sum_{b,k} m_{b,k}}$$
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Schedule:** Cosine annealing with 2 warmup epochs.
- **Batch Size:** 64.
- **Augmentation:** Gaussian noise ($\sigma=0.01$), random lead dropout ($p=0.10$).

### 6.2 Training Progression Log
Command:
```powershell
python -m ecg.cli train --config configs/baseline.yaml
```

**Epoch-by-Epoch Log:**
```
Epoch  1/15 | Val Macro AUROC: 0.8764 [BEST]  (Surpassed LightGBM 0.8503 on epoch 1!)
Epoch  2/15 | Val Macro AUROC: 0.8579
Epoch  3/15 | Val Macro AUROC: 0.8886 [BEST]
Epoch  4/15 | Val Macro AUROC: 0.8946 [BEST]
Epoch  5/15 | Val Macro AUROC: 0.9005 [BEST]  (Crossed 0.90 AUROC threshold!)
Epoch  6/15 | Val Macro AUROC: 0.8979
Epoch  7/15 | Val Macro AUROC: 0.9044 [BEST]
Epoch  8/15 | Val Macro AUROC: 0.9056 [BEST]
Epoch  9/15 | Val Macro AUROC: 0.9080 [BEST]
Epoch 10/15 | Val Macro AUROC: 0.9070
Epoch 11/15 | Val Macro AUROC: 0.9107 [BEST]
Epoch 12/15 | Val Macro AUROC: 0.9109 [BEST]  (Final Best Checkpoint Saved)
```

**Checkpoint Location:** `runs/S1/resnet1d/resnet100hz/42/best.pt`

---

## 7. Stage 6: Holdout Test Evaluation & Benchmark Results

### 7.1 Single-Command Benchmark Execution
Command:
```powershell
python -m ecg.cli eval --checkpoint runs/S1/resnet1d/resnet100hz/42/best.pt --split test
```

### 7.2 Benchmark Results on Holdout Test Split (Fold 10, 2,198 Records)
- **Macro AUROC:** **0.9112** (95% Bootstrap CI: **0.9045 - 0.9194**)
- **Macro AUPRC:** **0.7768**
- **Calibrated Expected Calibration Error (ECE):** **0.0225** (well below the < 0.05 target)

### 7.3 Per-Class Benchmark Breakdown

#### A. At Production Calibrated Decision Thresholds (Target ~90% Sensitivity)
*Decision thresholds from `thresholds.json`: NORM: 0.507, MI: 0.150, STTC: 0.230, CD: 0.118, HYP: 0.065.*

| Class | Diagnosis | AUROC | AUPRC | Sensitivity @ Threshold | Specificity @ Threshold | Prevalence |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **NORM** | Normal ECG | **0.9433** | 0.9226 | **0.905** | 0.823 | 44.6% |
| **MI** | Myocardial Infarction | **0.9218** | 0.8188 | **0.907** | 0.803 | 25.5% |
| **STTC** | ST/T Changes | **0.9333** | 0.8246 | **0.898** | 0.841 | 24.1% |
| **CD** | Conduction Disturbance | **0.9288** | 0.8520 | **0.901** | 0.831 | 23.0% |
| **HYP** | Hypertrophy | **0.8290** | 0.4659 | **0.897** | 0.613 | 12.1% |

*(Note: The raw 0.2786 sensitivity for HYP in previous logs was an artifact of an uncalibrated default 0.50 cutoff; at the calibrated bundle threshold of 0.065, HYP sensitivity is 0.897 with 0.613 specificity, fulfilling the clinical 90% sensitivity objective).*

### 7.4 Head-to-Head Comparison: ResNet1D vs LightGBM Baseline

| Metric | LightGBM Baseline | ResNet1D | Absolute Gain |
|:---|:---:|:---:|:---:|
| **Validation Macro AUROC** | 0.8503 | **0.9109** | **+0.0606** |
| **Test Macro AUROC** | 0.8456 | **0.9112** | **+0.0656** |
| **Test Macro AUPRC** | 0.6568 | **0.7768** | **+0.1200** |
| **NORM AUROC / Specificity @ 90% Sens** | 0.8901 / 0.711 | **0.9433 / 0.823** | +0.0532 / **+0.112 Spec** |
| **MI AUROC / Specificity @ 90% Sens** | 0.8145 / 0.552 | **0.9218 / 0.803** | +0.1073 / **+0.251 Spec** |
| **STTC AUROC / Specificity @ 90% Sens** | 0.8754 / 0.676 | **0.9333 / 0.841** | +0.0579 / **+0.165 Spec** |
| **CD AUROC / Specificity @ 90% Sens** | 0.8578 / 0.572 | **0.9288 / 0.831** | +0.0710 / **+0.259 Spec** |
| **HYP AUROC / Specificity @ 90% Sens** | 0.7902 / 0.433 | **0.8290 / 0.613** | +0.0388 / **+0.180 Spec** |

---

## 8. Stage 7: Experiment Tracking (MLflow)

- **MLflow Version:** 3.17.2
- **Tracking Database:** Local SQLite / MLflow store under `mlruns/`
- **Experiment Name:** `S1`
- **Run Identifier:** `S1/resnet1d/resnet100hz/42`
- **Logged Metadata & Parameters:**
  - `config_hash`: `e7c0a63fcc752901`
  - `manifest_hash`: `81f46ccd594df21f`
  - `git_commit`: `local-release`
  - `seed`: `42`
  - Optimizer, learning rate, warmup epochs, schedule, channel widths.
- **Logged Epoch Metrics:**
  - `val_macro_auroc` at each epoch.
  - `best_val_macro_auroc`.
- **Browsing Runs:**
  ```powershell
  python -m mlflow ui
  # Navigate to http://127.0.0.1:5000
  ```

---

## 9. Stage 8: ONNX Export, Parity Check & Release Bundle

### 9.1 Export Command & Verification
Command:
```powershell
python -m ecg.cli export
```

Target Directory: `bundle/v1.0.0/`

**Execution Log:**
```
Loading checkpoint for export: runs\S1\resnet1d\resnet100hz\42\best.pt
Fitting temperature scaling and thresholds on validation split...
Exporting ONNX graph to bundle\v1.0.0\model.onnx (opset=17)...
Running ONNX Runtime parity test on test sample...
PyTorch output shape: (16, 5), ONNX output shape: (16, 5)
Max absolute error between PyTorch and ONNX Runtime: 4.76837158203125e-06
Parity check PASSED: error 4.77e-06 <= 1.00e-04
Saved calibration.json
Saved thresholds.json
Saved abstention.json
Saved linear head weights: bundle\v1.0.0\head_weights.npy
Saved manifest: bundle\v1.0.0\manifest.json
Wrote bundle\v1.0.0\MODEL_CARD.md
Export bundle ready at: bundle\v1.0.0
```

### 9.2 Release Bundle Artifact Inventory

| File | Size | Purpose |
|:---|:---:|:---|
| `model.onnx` | 34.99 MB | Pinned ONNX neural network with dynamic batching |
| `calibration.json` | 200 B | Per-class temperature scaling parameters fitted on validation fold |
| `thresholds.json` | 200 B | Per-class decision thresholds (calibrated for 90% target sensitivity) |
| `abstention.json` | 89 B | Quality score floor (`0.25`) & confidence margin (`0.05`) rules |
| `head_weights.npy` | 10.3 KB | Linear head weight matrix `[5, 512]` for gradient-free CAM explanations |
| `labels.json` | 22.2 KB | Class identifiers, display names, and source mapping rules |
| `preprocess.json` | 172 B | Sampling rate (100 Hz), length (10 s), filter & normalization spec |
| `manifest.json` | 651 B | Provenance hashes, input/output contracts, test metrics |
| `MODEL_CARD.md` | 981 B | Clinical intended use, performance summary, and limitations |

---

## 10. Stage 9: Test Suite Execution Report

Command:
```powershell
python -m pytest tests -v
```

**Full Pytest Execution Log:**
```
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.2.0, pluggy-1.7.0 -- C:\Python314\python.exe
cachedir: .pytest_cache
rootdir: D:\Projects\mlprjresume
configfile: pyproject.toml
plugins: anyio-4.13.0, langsmith-0.8.17
collecting ... collected 15 items

tests/test_bundle.py::test_bundle_structure_and_schemas PASSED           [  6%]
tests/test_bundle.py::test_onnx_runtime_parity PASSED                    [ 13%]
tests/test_bundle.py::test_exported_head_weights_cam PASSED              [ 20%]
tests/test_phase1_3.py::test_every_mapping_row_resolves PASSED           [ 26%]
tests/test_phase1_3.py::test_unmapped_code_is_never_a_silent_negative PASSED [ 33%]
tests/test_phase1_3.py::test_mapped_record_marks_covered_classes_known PASSED [ 40%]
tests/test_phase1_3.py::test_v2_class_missing_from_source_is_masked PASSED [ 46%]
tests/test_phase1_3.py::test_split_integrity_no_patient_overlap PASSED   [ 53%]
tests/test_phase1_3.py::test_splits_are_deterministic PASSED             [ 60%]
tests/test_phase1_3.py::test_preprocess_shape_and_determinism PASSED     [ 66%]
tests/test_phase1_3.py::test_missing_leads_zero_filled_with_mask PASSED  [ 73%]
tests/test_phase1_3.py::test_ptbxl_uppercase_lead_names_are_matched PASSED [ 80%]
tests/test_phase1_3.py::test_short_record_is_zero_padded PASSED          [ 86%]
tests/test_phase1_3.py::test_augmentation_is_seeded_and_keeps_a_lead PASSED [ 93%]
tests/test_phase1_3.py::test_config_roundtrip_and_unknown_keys PASSED    [100%]

======================== 15 passed, 1 warning in 8.56s ========================
```

---

## 11. Command Reference for Replication

To reproduce every stage end-to-end:

```powershell
# 1. Download real PTB-XL records from S3 mirror
python data/download_ptbxl.py

# 2. Build harmonized labels matrix (Step 1)
python -m ecg.cli build-labels

# 3. Preprocess and cache 12-lead tensors (Step 2)
python -m ecg.cli build-data

# 4. Train & evaluate LightGBM baseline (Step 3)
python -m ecg.cli baseline

# 5. Train ResNet1D deep model (Step 4)
python -m ecg.cli train --config configs/baseline.yaml

# 6. Run standalone evaluation with bootstrap CIs (Step 5)
python -m ecg.cli eval --checkpoint runs/S1/resnet1d/resnet100hz/42/best.pt --split test

# 7. Export release bundle & run ONNX Runtime parity check (Step 7)
python -m ecg.cli export

# 8. Run full test suite
python -m pytest tests -v
```
