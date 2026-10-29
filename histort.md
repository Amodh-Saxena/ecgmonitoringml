# Project History & Complete Context Log: ECG Abnormality Classifier

**Last Updated:** October 2026  
**Repository:** `d:/Projects/mlprjresume`  
**Current Pipeline Version:** `v1.0.0`  
**Target Standard:** Technical PRD: Building the ECG Model  

---

## 1. Project Background & Objective

The goal of this project is to build an end-to-end, production-grade 12-lead ECG abnormality classifier and release bundle for clinical machine learning service integration (`ml_service`). 

### Core Requirements
- **Inputs:** Raw 12-lead ECG records from PhysioNet PTB-XL (21,799 records, 100 Hz / 500 Hz).
- **Target Diagnoses (v1 Superclasses):**
  1. `NORM`: Normal ECG
  2. `MI`: Myocardial Infarction
  3. `STTC`: ST/T Changes
  4. `CD`: Conduction Disturbance
  5. `HYP`: Hypertrophy
- **Architectures:**
  - Classical Baseline: LightGBM on 103 tabular/morphological features.
  - Deep Learning Model: ResNet1D operating directly on raw 12-lead signals with masked BCE loss.
- **Verification Criterion:** Deep learning model must beat the LightGBM validation macro AUROC.
- **Export Contract:** Self-contained bundle in `bundle/v1.0.0/` with ONNX graph, dynamic batching, numerical parity error $< 10^{-4}$, calibration parameters, optimal decision thresholds, CAM weights, and clinical model card.

---

## 2. Chronological Milestones & Stage Execution

### Milestone 1: Real Data Acquisition (`data/download_ptbxl.py`)
- **Initial Request:** Download real PTB-XL data via `wget` / `curl`.
- **Encountered Bottleneck:** Direct HTTP download from `physionet.org` throttled at ~10 KB/s (~45 hours ETA).
- **Solution:** Implemented `data/download_ptbxl.py` with 16 parallel threads targeting PhysioNet's open AWS S3 mirror (`https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/`).
- **Result:** Downloaded all **43,606 files** (~0.51 GB) into `data/raw/files/ptb-xl/1.0.3/`, including `ptbxl_database.csv`, `scp_statements.csv`, and all 21,799 WFDB recordings (`.hea` and `.dat`) across directories `records100/00000/` through `records100/21000/`.

---

### Milestone 2: Label Harmonization & Matrix (`src/ecg/datasets/label_matrix.py`)
- **Action:** Created mapping rules from SCP diagnostic statements into the 5 target superclasses.
- **Masking Mechanism:** Handled unmapped statements with loss masking ($m_k = 0$) rather than treating them as negative labels.
- **Execution:** `python -m ecg.cli build-labels`
- **Output:** `data/processed/ptbxl_labels_v1.npz` and `ptbxl_labels_v1.report.json`.
- **Audit Findings:**
  - 21,799 total records: 21,388 labeled (98.1%), 411 masked (1.9%), 5,144 multi-label (23.6%).
  - Positives: NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, HYP 2,649.
  - Stratified splits: Train (Folds 1-8): 17,418; Validation (Fold 9): 2,183; Test (Fold 10): 2,198.
  - Zero patient overlap verified across all splits (`train & val = 0`, `train & test = 0`, `val & test = 0`).

---

### Milestone 3: Deterministic Preprocessing Pipeline (`src/ecg/preprocess/signal.py`)
- **Pipeline Stages:**
  1. Reorder leads to canonical order: `["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]`.
  2. Polyphase rational resampling to 100 Hz (`scipy.signal.resample_poly`).
  3. Butterworth bandpass filtering (0.5 Hz – 40.0 Hz).
  4. Fixed 10.0-second length adjustment (1000 samples).
  5. Per-lead record z-score normalization: $x_{norm} = \frac{x - \mu}{\max(\sigma, 10^{-6})}$.
  6. Lead mask tracking: `float32 [12]` tracking available vs zero-padded missing leads.
- **Bug Fix:** Fixed case-sensitivity bug where WFDB headers with uppercase `AVR`, `AVL`, `AVF` failed to match canonical `aVR`, `aVL`, `aVF`. Re-indexed lead mapping case-insensitively, reaching 100% 12-lead coverage.
- **Cache Generation:** `python -m ecg.cli build-data` cached tensors in `data/cache/81f46ccd594df21f/`:
  - `signals.npy`: `(21799, 12, 1000)` float32
  - `lead_masks.npy`: `(21799, 12)` float32
  - `targets.npy`: `(21799, 5)` float32
  - `target_masks.npy`: `(21799, 5)` float32
  - Confirmed 0 NaNs/Infs and exact determinism on repeated runs.

---

### Milestone 4: Classical Baseline (`src/ecg/models/lgbm_baseline.py`)
- **Feature Extraction:** Extracted 103 handcrafted features per record (8 morphological features per lead $\times$ 12 leads = 96 features, plus 7 global rhythm features including Heart Rate, RR intervals, RMSSD, pNN50). Cached in `features_lgbm.npy`.
- **Model Training:** 5 independent binary `LGBMClassifier` models (300 estimators, lr=0.05, num_leaves=31).
- **Execution:** `python -m ecg.cli baseline`
- **Baseline Results:**
  - Validation Macro AUROC: **0.8503** (95% CI: 0.8388 – 0.8619)
  - Validation Macro AUPRC: **0.6699**
  - Test Macro AUROC: **0.8456** (95% CI: 0.8358 – 0.8559)
  - Per-class Validation AUROC: NORM 0.8964, MI 0.8426, STTC 0.8699, CD 0.8731, HYP 0.7696.

---

### Milestone 5: Deep Learning Model Training (`src/ecg/models/resnet1d.py`)
- **Model Architecture:**
  - 24 input channels (12 raw signal leads concatenated with 12 lead presence indicator masks).
  - Stem: 1D Conv (k=15, s=2, p=7) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ MaxPool1d (k=3, s=2, p=1).
  - 4 Residual stages with channel widths `[64, 128, 256, 512]` and 2 residual blocks per stage (k=7).
  - Temporal downsampling by 32: outputs convolutional feature map of shape `float32 [B, 512, 32]`.
  - Linear classification head mapping global average pooled features `[B, 512]` to `[B, 5]`.
- **Optimization Strategy:**
  - Masked BCE loss: $\mathcal{L} = \frac{\sum m_{b,k} \text{BCE}(z_{b,k}, y_{b,k})}{\sum m_{b,k}}$.
  - AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
  - Cosine annealing schedule with 2 warmup epochs.
  - Augmentation: Gaussian noise ($\sigma=0.01$) and random lead dropout ($p=0.10$).
- **Execution:** `python -m ecg.cli train --config configs/baseline.yaml`
- **Training Progression:**
  - Epoch 1: Val Macro AUROC = **0.8764** *(Beats LightGBM baseline 0.8503 immediately!)*
  - Epoch 3: Val Macro AUROC = **0.8886**
  - Epoch 4: Val Macro AUROC = **0.8946**
  - Epoch 5: Val Macro AUROC = **0.9005** *(Crossed 0.90 AUROC)*
  - Epoch 8: Val Macro AUROC = **0.9056**
  - Epoch 11: Val Macro AUROC = **0.9107**
  - Epoch 12: Val Macro AUROC = **0.9109** *(Final Best Checkpoint)*
- **Saved Best Model:** `runs/S1/resnet1d/resnet100hz/42/best.pt`

---

### Milestone 6: Holdout Test Evaluation & Benchmark Harness (`src/ecg/eval/harness.py`)
- **Execution:** `python -m ecg.cli eval --checkpoint runs/S1/resnet1d/resnet100hz/42/best.pt --split test`
- **Optimization for Speed:** Vectorized patient lookup in bootstrap loop (reduced 1000-iteration bootstrap time from minutes to $< 1$ second).
- **Test Metrics (Fold 10, 2,198 records):**
  - **Macro AUROC:** **0.9112** (95% Bootstrap CI: **0.9045 – 0.9194**)
  - **Macro AUPRC:** **0.7768**
  - **Calibrated Expected Calibration Error (ECE):** **0.0225** (well below the 0.05 PRD requirement)
- **Per-Class Breakdown:**
  - `NORM`: AUROC **0.9433** | AUPRC 0.9226 | F1 0.8518 | Sens 0.9045 | Spec 0.8234
  - `MI`: AUROC **0.9218** | AUPRC 0.8188 | F1 0.7007 | Sens 0.6236 | Spec 0.9465
  - `STTC`: AUROC **0.9333** | AUPRC 0.8246 | F1 0.7485 | Sens 0.7370 | Spec 0.9261
  - `CD`: AUROC **0.9288** | AUPRC 0.8520 | F1 0.7400 | Sens 0.6512 | Spec 0.9675
  - `HYP`: AUROC **0.8290** | AUPRC 0.4659 | F1 0.3862 | Sens 0.2786 | Spec 0.9773

---

### Milestone 7: MLflow Experiment Tracking (`src/ecg/train/tracking.py`)
- Integrated MLflow 3.17.2 tracking under experiment `S1`.
- Logged config hash (`e7c0a63fcc752901`), manifest hash (`81f46ccd594df21f`), seed (`42`), training hyperparameters, and epoch metrics.
- Browseable locally via `python -m mlflow ui`.

---

### Milestone 8: ONNX Export & Parity Verification (`src/ecg/export/bundle.py`)
- **Export Configuration:** TorchScript ONNX exporter (opset 17) with dynamic batching.
- **Export Artifacts in `bundle/v1.0.0/`:**
  - `model.onnx` (34.99 MB): Graph accepting `signal [B, 12, 1000]` and `lead_mask [B, 12]`, returning `logits [B, 5]` and `features [B, 512, 32]`.
  - `calibration.json`: Per-class temperature scaling parameters.
  - `thresholds.json`: Per-class decision thresholds calibrated for 90% target sensitivity.
  - `abstention.json`: Signal quality floor (0.25) and confidence margin (0.05).
  - `head_weights.npy` (10.3 KB): `[5, 512]` linear head weight matrix for gradient-free CAM heatmaps.
  - `labels.json`: Diagnostic class definitions and mapping metadata.
  - `preprocess.json`: Preprocessing contract specification.
  - `manifest.json`: Hashes, input/output tensor contracts, and benchmark metrics.
  - `MODEL_CARD.md`: Complete clinical model card.
- **Parity Verification:** ONNX Runtime parity test on real ECG records yielded a maximum absolute difference of **$4.77 \times 10^{-6}$** ($\ll 1.00 \times 10^{-4}$).

---

### Milestone 9: Test Suite Validation
- **Execution:** `python -m pytest tests -v`
- **Result:** **15 of 15 tests PASSED** in 8.56s across unit tests (`test_phase1_3.py`) and export bundle integration tests (`test_bundle.py`).

---

### Milestone 10: Production Reports
- Compiled full production audit and training logs into [prod.md](file:///d:/Projects/mlprjresume/prod.md) and [prood.md](file:///d:/Projects/mlprjresume/prood.md).

---

## 3. Key Bugs Encountered & Solutions Applied

| # | Issue / Error | Root Cause | Solution |
|:---|:---|:---|:---|
| 1 | `ModuleNotFoundError: No module named 'onnx'` | Minimal Python environment missing dependencies | Installed `onnx`, `onnxruntime`, `wfdb`, `lightgbm`, `pytest`, `mlflow`, `matplotlib` |
| 2 | Slow PhysioNet download (~10 KB/s) | Direct HTTP rate limiting on PhysioNet servers | Wrote multi-threaded S3 downloader fetching from `physionet-open.s3.amazonaws.com` with 16 workers |
| 3 | 0 of 21,799 records had all 12 leads detected | WFDB header lead names (`AVR`, `AVL`, `AVF`) didn't match canonical lowercase (`aVR`, `aVL`, `aVF`) | Updated `src/ecg/preprocess/signal.py` to match lead names case-insensitively; rebuilt cache with 100% lead coverage |
| 4 | Windows console crash during ONNX export | PyTorch Dynamo exporter attempted to print unicode checkmark `\u2705` on `cp1252` Windows terminal | Set `dynamo=False` in `torch.onnx.export` and reconfigured `sys.stdout` to UTF-8 |
| 5 | Test shape assertion mismatch `(2, 512, 63)` | Calculation error in initial test expectation ($\lfloor 1000/32 \rfloor = 32$, not 63) | Corrected temporal feature map dimensions across test files, manifest, and model card |
| 6 | Slow bootstrap CI computation | Iterating over all patient IDs in Python loop inside 1000 bootstrap iterations | Pre-computed patient record indices array for $O(1)$ batch lookup; reduced runtime from 3 minutes to $< 1$ second |

---

## 4. Current Repository Structure

```
d:/Projects/mlprjresume/
├── bundle/
│   └── v1.0.0/
│       ├── abstention.json
│       ├── calibration.json
│       ├── head_weights.npy
│       ├── labels.json
│       ├── manifest.json
│       ├── model.onnx
│       ├── MODEL_CARD.md
│       ├── preprocess.json
│       └── thresholds.json
├── configs/
│   └── baseline.yaml
├── data/
│   ├── build_label_tables.py
│   ├── cache/81f46ccd594df21f/   (signals.npy, lead_masks.npy, targets.npy, target_masks.npy)
│   ├── download_ptbxl.py
│   ├── processed/ptbxl_labels_v1.npz
│   └── raw/files/ptb-xl/1.0.3/   (43,606 files)
├── runs/
│   ├── baseline/baseline_metrics.json
│   └── S1/resnet1d/resnet100hz/42/ (best.pt, last.pt, run.json)
├── src/
│   └── ecg/
│       ├── cli.py
│       ├── datasets/ (ptbxl.py, label_matrix.py, build.py, cinc.py, chapman.py)
│       ├── eval/ (harness.py, metrics.py, calibration.py, uncertainty.py, quality.py)
│       ├── export/ (bundle.py)
│       ├── models/ (resnet1d.py, lgbm_baseline.py)
│       ├── preprocess/ (signal.py, augment.py)
│       └── train/ (trainer.py, losses.py, tracking.py, shift.py)
├── tests/
│   ├── test_bundle.py
│   └── test_phase1_3.py
├── PROGRESS.md
├── prod.md
├── prood.md
├── history.md
├── histort.md
├── README.md
└── pyproject.toml
```

---

## 5. Next Steps & Future Work

1. **Study Protocols S2–S7 (PRD Section 8):**
   - S2: Preprocessing ablations (sampling rate 250/500 Hz, filter on/off).
   - S3: Augmentation ablation (noise, lead dropout, mixup).
   - S4: Source fingerprint classifier.
   - S5: Lead reduction degradation curves (12, 6, 4, 2, 1 lead inputs).
   - S6: Self-supervised pretraining label-efficiency curves (1%, 10%, 100%).
   - S7: Domain shift recovery on CinC 2020 / Chapman datasets via test-time BatchNorm adaptation.
2. **Clinical Explainability (PRD Section 10):**
   - Implement Grad-CAM 1D in PyTorch (`src/ecg/explain/gradcam.py`).
   - Run Faithfulness test (random window occlusion vs attribution window drop).
   - Integrate NeuroKit2 for P/QRS/T wave attribution mass delineation.
