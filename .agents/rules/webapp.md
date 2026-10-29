# Standing Instructions: ECG Triage Web App (Part 1)

1. **Target Directory:** Build the web app described in `docs/prd-web-app.md` strictly inside `webapp/` of the repository (`webapp/backend/` and `webapp/frontend/`).
2. **Model Immutability:** The ECG model is finished and released. Treat `bundle/v1.0.0` and `src/ecg` as read-only inputs.
3. **No Retraining or Modifying:** Do NOT train, export, tune, or modify the model, its configs, data, or bundle files.
4. **No Preprocessing Reimplementation:** Import the existing preprocessing code from `src/ecg/preprocess/signal.py`; do NOT reimplement it in the website.
5. **Single Source of Truth:** Read all model values (classes, thresholds, temperatures, abstention floors, version) from `bundle/v1.0.0/` at startup; never hard-code them.
6. **No PyTorch in Web App:** Runtime must use ONNX Runtime only. Do NOT add PyTorch, LightGBM, or MLflow to `webapp/backend/requirements.txt`.
7. **Copy Rules:** Follow clinical copy guidelines strictly:
   - Use "Flagged for review" — NEVER use "detected", "diagnosed", or "you have".
   - Use "The model's attention" — NEVER use "the cause" or "the reason".
   - Visible research prototype disclaimer on every page.
8. **Phase-by-Phase Discipline:**
   - Phase 1: Backend only (`webapp/backend`), API schemas, endpoints, and golden parity test on 20 test records.
   - Phase 2: Frontend foundation (`webapp/frontend`) with React, Vite, TypeScript, Tailwind tokens, and Upload page.
   - Phase 3: Results page with 12-lead ECG paper viewer, attention overlay, and calibrated probabilities.
   - Phase 4: About page, accessibility audit, error banners.
   - Phase 5: Dockerfile, docker-compose, and sample ECG packaging.
