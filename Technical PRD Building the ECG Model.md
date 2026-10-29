# Technical PRD: Building the ECG Model

Oct 28, 2026 · @Amodh

## Purpose and scope

This spec defines how to build, train, evaluate and export the ECG abnormality model so it can later drop into the product's `ml_service` without rework. It is the build companion to the product PRD, which holds goals, users and release gates.

**In scope**

- Data pipeline, label mapping, preprocessing and augmentation.
- Model code, training configs, experiment tracking and evaluation harness.
- Calibration, abstention, explanations, ONNX export and the inference contract.

**Out of scope here**

- Frontend, backend, auth, report generation and deployment infrastructure.
- ECG image digitization (specified separately; this model consumes signals).

**Hand-off artifact.** One versioned bundle: `model.onnx`, `calibration.json`, `thresholds.json`, `labels.json`, `preprocess.json` and a model card. Integration depends only on this bundle and the contract in the export section.

## Repository layout and environment

```text
ecg-model/
  configs/            YAML per experiment (data, model, train, eval)
  data/               Download and manifest scripts only; raw data is not committed
  src/ecg/
    datasets/         Loaders per source (ptbxl.py, cinc.py, chapman.py)
    labels/           Mapping tables and class definitions
    preprocess/       Resample, filter, normalize, augment
    models/           resnet1d.py, inception1d.py, cnn_transformer.py
    train/            Trainer, losses, SSL pretraining
    eval/             Metrics, bootstrap, calibration, shift tests
    explain/          Grad-CAM, Integrated Gradients, deletion test
    export/           ONNX export, parity test, bundle writer
  tests/              Unit and parity tests
  runs/               Tracked outputs (gitignored)
```

| Item | Choice |
| --- | --- |
| Language and framework | Python 3.10+, PyTorch |
| Signal I/O | `wfdb`; `neurokit2` for delineation and features |
| Classical baseline | LightGBM |
| Config | YAML files loaded into typed dataclasses; one config fully determines a run |
| Tracking | MLflow or Weights & Biases: params, metrics, artifacts, git hash, data manifest hash |
| Reproducibility | Fixed seeds, pinned `requirements.txt`, deterministic flags where feasible |
| Hardware | Single free-tier GPU is enough for 100 Hz PTB-XL; CPU for tests |

## Data pipeline

The pipeline turns raw datasets into versioned, split, preprocessed tensors, and records everything needed to rebuild them. It lives in `data/` and `src/ecg/datasets/`.

&#91;embedded content: data pipeline · 6 stages\]

Read the top row left to right, then the bottom row right to left; label mapping is the stage that most affects later results.

**Requirements**

1. Each loader returns the signal, sampling rate, lead names, record id, patient id, source and raw label codes.
2. Patient id is mandatory. A record without one is flagged and given a unique id, or excluded.
3. Label mapping runs before preprocessing, and the audit report is written at this step.
4. Splits are assigned once per dataset version, stored, and never regenerated silently.
5. The manifest stores file hashes, label version, preprocessing config, split assignment and counts; its hash identifies the dataset version.
6. Cached tensors are stored as memory-mapped arrays keyed by the manifest hash.

## Label harmonization spec

One `labels.json` defines the class set, and every dataset loader maps into it. A class a source gives no information about is masked, never treated as a silent negative.

**Rules**

1. Class set v1: the five PTB-XL superclasses (NORM, MI, STTC, CD, HYP). v2 adds atrial fibrillation (AF), left bundle branch block (LBBB), right bundle branch block (RBBB), and any other class present in every source.
2. PTB-XL: map SCP statements to superclasses with the dataset's own statement table, and record whether each label comes from a human-validated record.
3. CinC, Chapman and Ningbo: map SNOMED-CT codes through an explicit table, including the challenge's equivalent-class groups.
4. Targets are a binary vector `y` and a mask vector `m`, both of length K. `m = 0` means the source says nothing about that class, and the loss ignores it.
5. NORM is its own class where the source labels it; it is not derived as "no other label".
6. Every mapping row stores the source code, description, target class, rule and reviewer.

| Field | Example |
| --- | --- |
| `source` | `ptbxl` |
| `source_code` | An SCP statement or SNOMED code |
| `target_class` | A class from the set, or `unmapped` |
| `rule` | `exact`, `parent` or `equivalent` |
| `notes` | Free text for ambiguous cases |

**Audit output, required before any training:** per-source, per-class record counts; the unmapped-code list with counts; and a patient-overlap check across splits.

## Preprocessing and augmentation specification

Preprocessing is defined once in `preprocess.json` and reused unchanged at inference, so training and serving cannot drift apart.

| Step | Spec | Why |
| --- | --- | --- |
| Sampling rate | Resample every record to one rate (start at 100 Hz; ablate 250 to 500 Hz) | Removes the simplest dataset fingerprint |
| Length | Fixed 10 s window; crop or zero-pad | Fixed tensor shape for ONNX |
| Lead order | Canonical 12-lead order; missing leads zero-filled with a lead mask | Reduced-lead support |
| Filtering | Band-pass (default about 0.5 to 40 Hz), optional mains notch; ablate on and off | Tests filter-driven shortcuts |
| Normalization | Per-lead z-score per record, or fixed training-set statistics; choose by ablation | Gain and amplitude differences across devices |
| Output | `float32 [B, 12, T]` plus `lead_mask [B, 12]` | The contract with export |

**Augmentations (training only; each has a probability and range in config)**

- Gaussian noise, baseline wander (low-frequency sinusoid), powerline interference.
- Per-lead amplitude scaling, time shift, random crop.
- Lead dropout (random lead masking) for reduced-lead robustness.
- Small resampling jitter.
- Optional regularizers: mixup, time masking.

**Rules**

1. Augmentation never touches validation or test data.
2. All randomness uses a seeded generator.
3. The exact transform list is stored in the run config.

## Model architectures and training configuration

Every neural model shares one interface: inputs `signal [B, 12, T]` and `lead_mask [B, 12]`, output `logits [B, K]` (multi-label, sigmoid).

| Model | Core design | Notes |
| --- | --- | --- |
| LightGBM baseline | Handcrafted per-lead and global features (heart rate, intervals, amplitudes, HRV statistics) | One binary model per class; must be beaten or the gap explained |
| ResNet1D | Stem convolution, residual 1D blocks, global pooling, linear head | Main supervised baseline |
| xresnet1d, InceptionTime | Alternative CNN families | Compared under identical data and budget |
| CNN + Transformer | Convolutional front end that downsamples, Transformer encoder over time, pooled head | Tests attention under shift |
| Self-supervised encoder | Contrastive or masked-reconstruction pretraining, then fine-tuning | Label-efficiency study |

**Training defaults (a starting point; tune with an equal budget per model)**

| Setting | Default |
| --- | --- |
| Loss | Binary cross-entropy with logits, multiplied by the label mask; optional class weights or focal loss |
| Optimizer | AdamW with weight decay |
| Schedule | One-cycle or cosine with warmup; peak learning rate from a short sweep |
| Batch size | 64 to 128, as GPU memory allows |
| Stopping | Early stopping on validation macro AUROC, with patience |
| Precision | Mixed precision (AMP) |
| Seeds | 3 to 5 per config |
| Checkpoints | Best by validation macro AUROC, with optimizer state for resume |

**Fair comparison rule:** every model gets the same data, the same augmentation list and the same hyperparameter-search budget, for example the same number of trials.

## Experiment protocol

Each experiment is one YAML config that fully determines a run. Runs are grouped into studies, and each study answers one question.

| Study | Varies | Fixed | Output |
| --- | --- | --- | --- |
| S1 Baselines | Model family | Data, preprocessing, augmentation | In-domain and external scores per model |
| S2 Preprocessing | Sampling rate, filtering, normalization | Best model from S1 | Effect on the external gap |
| S3 Augmentation | Augmentation set, lead dropout | Best model and preprocessing | Robustness and shift curves |
| S4 Dataset fingerprint | Source classifier vs diagnostic model | Same inputs | Source-prediction accuracy; gap change after mitigation |
| S5 Lead reduction | 12, 6, 4, 2 and 1 leads | Best config | Degradation curve per class |
| S6 Pretraining | Self-supervised on or off; label fraction 1%, 10%, 100% | Best architecture | Label-efficiency curve |
| S7 Shift recovery | None, BatchNorm adaptation, small-sample fine-tuning | Final model | Recovery per external set |

**Rules**

1. Tune only on source validation data. Evaluate external sets after configs are frozen.
2. Log to the tracker: config hash, git commit, data manifest hash, seed, metrics, calibration plots and confusion tables.
3. Name runs `study/model/variant/seed`.
4. Re-running a config with the same seed reproduces its metrics within a documented tolerance.
5. A run that fails a leakage check (patient overlap, external data in training) is marked invalid in the tracker.

## Evaluation harness

One function, `evaluate(model, dataset, config)`, produces every reported number. Nothing is computed by hand in notebooks.

**Metrics**

| Metric | Implementation note |
| --- | --- |
| AUROC, per class and macro | Skip classes with no positives in the split, and report how many were skipped |
| AUPRC, per class and macro | Report class prevalence next to each value |
| F1, sensitivity, specificity | At thresholds chosen on source validation data |
| ECE and reliability curve | Equal-mass bins; before and after calibration |
| Brier score | Per class |
| Abstention curve | Accuracy against coverage as the uncertainty limit varies |

**Statistics**

- Bootstrap 95% confidence intervals, resampling patients rather than records.
- 3 to 5 seeds per config; report the mean and standard deviation across seeds.
- Paired model comparisons reuse the same bootstrap samples.

**Evaluation sets**

| Set | Use |
| --- | --- |
| PTB-XL validation fold | Model selection, thresholds, calibration |
| PTB-XL test fold | In-domain report |
| Each external set, separately | Generalization report; never used for tuning |
| Subgroups (age band, sex) | Fairness report |

**Outputs per run:** `metrics.json`, a per-class CSV, reliability plots, confusion tables, and a report page that compares runs side by side.

## Calibration, abstention and uncertainty implementation

**Calibration**

- Fit temperature scaling (one scalar per class, or one global) on the source validation set by minimizing negative log-likelihood. Store it in `calibration.json`.
- Compare against Platt scaling and isotonic regression, and keep the simplest method that meets the ECE target.
- Report ECE on each external set with the source-fit calibrator, and again after recalibrating on a small target sample.

**Thresholds**

- Set per-class decision thresholds on source validation data to a stated criterion, for example a target sensitivity. Store them in `thresholds.json`.

**Uncertainty and abstention**

| Signal | How it is computed |
| --- | --- |
| Predictive confidence | Distance of the calibrated probability from the class threshold |
| Ensemble disagreement | Variance across seeds or snapshots, if cost allows |
| Input quality | Signal quality score from noise level, flat segments and missing leads |

- Status rule: `rejected` if quality is below its floor, `uncertain` if confidence or disagreement crosses its limit, otherwise `confident`.
- Choose the limits from the abstention curve on source validation data, then check them on external sets.
- Optional: conformal prediction for set-valued outputs with a coverage target, with coverage evaluated under shift.

## Explainability implementation and faithfulness tests

**Methods**

| Method | Implementation note | Where it runs |
| --- | --- | --- |
| CAM from exported features | For a global-pooling CNN with a linear head, attribution for class c is the head weights times the last feature map, summed over channels. No gradients needed. | ONNX service path |
| Grad-CAM (1D) | Gradient of the class logit with respect to the last convolutional feature map, upsampled to input length | PyTorch research path |
| Integrated Gradients | Zero or per-lead-mean baseline, 50 or more interpolation steps; compare baselines | PyTorch research path |
| Occlusion | Slide a masked window and record the probability drop | Either path; slower |

Output: an attribution array `[12, T]` per predicted class, normalized to 0..1.

**Faithfulness test**

1. For each test ECG and predicted class, mask the top-attributed windows and record the drop in class probability.
2. Repeat with random windows of the same total length.
3. Report the mean drop for attributed and random masks, with bootstrap intervals.

**Clinical plausibility**

- Detect R peaks and delineate P, QRS, ST and T segments with `neurokit2`.
- Report the share of attribution mass in each segment, per class.

**Stability**

- Compute rank correlation of attributions across seeds and under small input noise.
- A method ships only if it beats the random baseline and is stable across seeds.

## Export and inference contract

Product integration depends only on this contract.

**Bundle**

```text
bundle/<model_version>/
  model.onnx          Network, opset pinned
  preprocess.json     Sampling rate, length, filter and normalization parameters
  labels.json         Ordered class list and display names
  calibration.json    Temperature or calibrator per class
  thresholds.json     Decision thresholds per class
  abstention.json     Quality floor, confidence and disagreement limits
  head_weights.npy    Linear-head weights for CAM explanations
  manifest.json       Data manifest hash, git commit, metrics summary, licence notes
  MODEL_CARD.md
```

**ONNX tensors**

| Tensor | Shape and type | Meaning |
| --- | --- | --- |
| `signal` (input) | `float32 [B, 12, T]` | Preprocessed ECG, canonical lead order |
| `lead_mask` (input) | `float32 [B, 12]` | 1 where the lead is present |
| `logits` (output) | `float32 [B, K]` | Uncalibrated class scores |
| `features` (output) | `float32 [B, D, T']` | Last convolutional feature map, for CAM |

The batch dimension is dynamic, and T is fixed by `preprocess.json`. Calibration, thresholds and abstention are applied in the service from the JSON files, not inside the graph, so they can change without re-exporting the network.

**Service response from `ml_service`**

```json
{
  "model_version": "string",
  "status": "confident | uncertain | rejected",
  "quality_score": 0.0,
  "classes": [{"name": "string", "probability": 0.0, "above_threshold": false}],
  "explanation": {"method": "cam", "attribution": "[12][T] floats in 0..1"},
  "manifest_hash": "string"
}
```

**Compatibility rules**

1. Any change to K, class order, T, sampling rate or lead order is a major version bump.
2. A new bundle must pass the parity test before release.
3. Keep the previous bundle available for rollback; the service picks the version from config.
4. This layer receives signals and opaque request ids only, never patient identifiers.

## Testing, acceptance and build checklist

**Tests**

| Test | Check |
| --- | --- |
| Label mapping | Every mapping row resolves; counts match the audit; no unmapped code becomes a silent negative |
| Split integrity | No patient appears in more than one split; external sets are absent from training |
| Preprocess determinism | Same input gives an identical tensor; training and inference share one code path |
| Model shape | The forward pass returns `[B, K]` for every supported lead mask |
| Metric sanity | Random scores give near-chance AUROC; perfect scores give 1 |
| ONNX parity | PyTorch and ONNX Runtime logits agree within a set tolerance (proposed: max absolute difference 1e-4) on a fixed sample |
| CAM consistency | CAM from exported features matches CAM computed in PyTorch |
| Latency | One ECG on CPU meets the product target |
| Contract | Bundle files validate against JSON schemas; version-bump rules are enforced in CI |

**Acceptance:** a bundle is releasable when the product PRD's release gates pass and every test above is green.

**Build checklist**

- [ ] Repo, environment and experiment tracking set up
- [ ] Dataset loaders and manifests for PTB-XL and one external set
- [ ] Label mapping v1 and audit report
- [ ] Preprocessing and augmentation modules with tests
- [ ] LightGBM baseline and ResNet1D with source validation scores
- [ ] Evaluation harness with bootstrap intervals
- [ ] External evaluation and dataset-fingerprint study
- [ ] Calibration, thresholds and abstention files
- [ ] Explanation module and faithfulness test
- [ ] ONNX export, parity test and bundle writer
- [ ] Model card and release bundle v1
