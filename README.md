# ECG Model

Reproducible research pipeline for multi-label ECG abnormality classification. The repository follows the technical PRD in `Technical PRD Building the ECG Model.md`; raw ECG data and trained weights are intentionally not stored in Git.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,data,export]"
python -m ecg.cli audit --labels src/ecg/labels/labels.json
```

Use `python -m ecg.cli --help` for available commands. Dataset acquisition is deliberately explicit: configure access and licenses for each source before downloading it.

## Repository map

- `configs/`: typed experiment configuration examples.
- `data/`: download/manifest entry points; raw data is ignored.
- `src/ecg/`: labels, preprocessing, model, training, evaluation, explanations and export.
- `tests/`: contract-focused unit tests.
- `runs/`: generated outputs (gitignored).

## Initial implementation status

Implemented: masked-label loss, deterministic signal preprocessing and training augmentation, patient-group splits/manifests, ResNet1D, calibration/threshold helpers, patient bootstrap metrics, quality scoring, CAM, ONNX bundle export and bundle validation. Source-specific PTB-XL/CinC/Chapman readers and trained model bundles require the licensed datasets and are not fabricated here.
