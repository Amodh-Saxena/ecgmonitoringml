# PRD: ECG Triage Web App, Part 1

Oct 29, 2026 · @Amodh

## Overview and scope

Part 1 is a professional, clinical-grade web app where a user uploads an ECG signal file and receives the model's results: class probabilities, a status (`confident`, `uncertain` or `rejected`), and a waveform view with the model's attention. It is an add-on to the finished ECG model: its backend loads the `bundle/v1.0.0` release and never changes it.

**Goals**

1. Upload a signal file and get a clear, correctly labelled result within a few seconds.
2. Look and read like a credible clinical tool: calm, precise and uncluttered.
3. Never overstate: every result carries its uncertainty, its limits and a visible disclaimer.
4. Match the model bundle exactly, with no re-implemented preprocessing in the website.

**In scope (Part 1)**

| Area | Included |
| --- | --- |
| Input | WFDB zip (`.hea` + `.dat`), CSV and `.npy` signal files, plus a built-in sample ECG |
| Output | Status, class probabilities with review thresholds, 12-lead viewer, attention overlay, JSON download |
| Pages | Upload, Results, About the model |
| Platform | Desktop-first responsive web, no accounts, no storage of uploads |

**Out of scope (later parts)**

- PDF or photo ECG digitization, PDF reports, user accounts and history, clinician feedback capture, batch upload, and hardening for real patient data.

## Relationship to the existing model project

This web app is an add-on. The ECG model is finished and lives in the `ecg-model` repository as release bundle `bundle/v1.0.0`. Everything in this PRD is new work under a `webapp/` folder. Read this section first.

&#91;embedded content: project boundary · model project vs web app\]

The web app loads the released bundle and imports the model's preprocessing code. It never trains, exports or edits anything in the model project.

**Terms used in these documents**

| Term | Meaning |
| --- | --- |
| Model project | The existing `ecg-model` repo: `src/ecg`, `configs`, `runs`, `data`, `tests` and `bundle`. Finished for v1.0.0. |
| Bundle | `bundle/v1.0.0/`: `model.onnx`, `preprocess.json`, `labels.json`, `calibration.json`, `thresholds.json`, `abstention.json`, `head_weights.npy`, `manifest.json`. The only model artifact the web app uses. |
| Web app | The `webapp/` folder (`backend/` and `frontend/`), built from this PRD |
| `ml_service` | The inference layer named in the other PRDs. In this web app it is `backend/app/model_loader.py` and `inference.py`; there is no separate `ml_service` process. |
| Technical PRD | "Technical PRD: Building the ECG Model". Reference for how the bundle was built and what its contract is. |
| Product PRD | "PRD: ECG Abnormality Detection Model". Goals and release gates. Not an instruction to build or retrain anything. |

**Repository layout**

```text
ecg-model/                      Existing repo (model project): do not change
  src/ecg/                      Model code and preprocessing
  configs/  runs/  data/  tests/
  bundle/v1.0.0/                Release bundle: read-only input
  webapp/                       NEW: everything this PRD builds
    backend/
      app/
        main.py                 FastAPI app, routes, CORS, startup
        config.py               Paths, env vars, size and time limits
        model_loader.py         Loads the bundle once at startup (singleton)
        schemas.py              Pydantic models matching the API contract
        parsing.py              Reads and validates zip, csv and npy uploads
        inference.py            Preprocess, ONNX, calibrate, status, CAM
      tests/                    Golden parity and bad-input tests
      requirements.txt          Runtime only: fastapi, uvicorn, onnxruntime, numpy, scipy, wfdb, pydantic
      Dockerfile
    frontend/
      src/
        components/             UploadForm, StatusBadge, ProbabilityBars, EcgViewer, Disclaimer
        pages/                  Upload, Results, About
        api/client.ts           Calls to the backend
        App.tsx
      public/                   Sample ECG data
      package.json
      vite.config.ts
    docker-compose.yml
```

File names follow the sketch the project owner provided, with TypeScript in the frontend (`.ts` and `.tsx` in place of `.js` and `.jsx`). The model project already plays the role of the sketch's `training/` folder, so do not create a new one. `config.py` reads `BUNDLE_DIR` (default `../../bundle/v1.0.0` in development); the Docker image copies the bundle in at build time and never writes to it.

**Rules for whoever implements this (person or coding agent)**

**Allowed**

- Create and edit files only inside `webapp/`.
- Read files in `bundle/v1.0.0/`, import the preprocessing code from `src/ecg`, and read PTB-XL test records under `data/` for tests.
- If importing that code pulls in training libraries (PyTorch, LightGBM, MLflow), move the preprocessing into a small dependency-light module inside `src/ecg`, once. Keep behaviour identical, move the code rather than copy it, and keep the existing preprocessing tests passing.

**Not allowed**

- Training, tuning, exporting or regenerating the bundle. Do not run the `ecg.cli` commands `train`, `export`, `baseline`, `build-labels` or `build-data`.
- Editing model code (`src/ecg/models`, `train`, `eval`, `export`), configs, runs, data or bundle files.
- Re-implementing preprocessing, calibration, thresholds, abstention or class order in the web app, or hard-coding values that exist in the bundle.
- Adding PyTorch, LightGBM or MLflow to the web app's runtime dependencies.
- Changing the model's metrics or claims. The About page quotes numbers from `manifest.json` and `MODEL_CARD.md`.

**Single source of truth.** The backend reads class order, sampling rate, window length, filter settings, temperature, thresholds and abstention limits from the bundle at startup. If a value it needs is missing from the bundle, stop and report it; do not invent one.

**If something is missing or broken.** If `bundle/v1.0.0` is absent, the ONNX parity test fails, or a model-project test fails, stop and report it. Do not retrain or patch the model to make the web app work.

**When documents disagree.** This PRD wins for the website and the `/predict` response; the bundle files win for model values; the Technical PRD explains how the bundle was built. Attention is one time profile per class (`[K][T]`), because the model's first convolution mixes all leads.

**Agent brief (paste-ready)**

```text
Build the web app described in "PRD: ECG Triage Web App, Part 1" inside webapp/ of the existing ecg-model repo.
The ECG model is finished. Treat bundle/v1.0.0 and src/ecg as read-only inputs.
Do not train, export, tune or modify the model, its configs, data or bundle.
Import the existing preprocessing code; do not reimplement it.
Read all model values from the bundle at startup; do not hard-code them.
If anything in the bundle is missing or a model test fails, stop and report instead of fixing the model.
Start with the backend and its golden parity test, then the frontend.
```

## Users and key flows

| User | Goal | What they need from the page |
| --- | --- | --- |
| Clinician or technician (primary) | A quick second look at an ECG | A fast result in plain words, with the evidence on screen |
| ML researcher or reviewer | Inspect model behaviour | Exact probabilities, thresholds, model version, JSON export |
| Demo reviewer | Judge the quality of the product | A polished flow, with a one-click sample ECG |

Primary flow: open the site, drop a file or choose the sample, wait for processing, read the result, then analyze another file. The diagram below shows where a file can be stopped.

&#91;embedded content: user flow · 6 steps, 1 decision\]

A file that fails quality checks never reaches the model; the user gets reasons and returns to the upload page.

## Functional requirements

**Upload**

| ID | Requirement |
| --- | --- |
| U1 | Drag-and-drop area plus a file picker; one file per analysis |
| U2 | Accepted: `.zip` (WFDB `.hea` + `.dat`), `.csv`, `.npy`; size limit set in config (proposed 20 MB) |
| U3 | CSV and `.npy` uploads ask for the sampling rate when it cannot be read from the file |
| U4 | Client-side checks for type and size before upload, with an inline message |
| U5 | A "Try a sample ECG" button that runs a bundled public PTB-XL record |

**Validation and processing (server)**

| ID | Requirement |
| --- | --- |
| P1 | Reuse the model's own preprocessing code; the website never reimplements it |
| P2 | Reject files with an unreadable format, NaN values, a flat signal, under 10 s of data, or an unknown sampling rate |
| P3 | Reduced-lead files are processed with a lead mask, and the result states how many leads were used |
| P4 | Strip patient-identifying header fields (name, age, sex, dates) before use; never display or log them |
| P5 | Show a progress state while processing, and a timeout message after a set limit |

**Results**

| ID | Requirement |
| --- | --- |
| R1 | Show the status (`confident`, `uncertain` or `rejected`) with a plain-language line |
| R2 | Show a calibrated probability for each of the 5 classes, with the review threshold marked |
| R3 | Show the 12-lead waveform; the attention overlay is optional and off by default |
| R4 | Show the quality score, model version and number of leads used |
| R5 | Offer a JSON download of the result |
| R6 | `uncertain` results show a review banner; `rejected` results show reasons and how to fix them |

**Errors:** every error has a title, a one-sentence cause and a next step. No stack traces or raw codes are shown to users.

## Results page specification

The page answers three questions in order: how reliable is this result, what did the model flag, and where in the ECG did it look. The layout below shows the zones and what each one carries.

&#91;embedded content: results page layout · 5 zones\]

On a phone, the zones stack top to bottom in the order the page answers its questions.

**Behaviour**

- The status badge comes first and pairs colour with an icon and a text label.
- Probabilities are calibrated values, and the threshold tick comes from `thresholds.json`. A class above its tick is labelled "Flagged for review", never "detected" or "diagnosed".
- The waveform plots the filtered signal in its original scale. The model's input is normalized per lead, so never draw model-input values with millivolt labels; if units are unknown, label the axis "arbitrary units".
- The attention overlay is one time band shared by all leads, because the model's first convolution mixes leads. Do not draw per-lead attribution.
- Hover or focus on any trace or bar shows exact values, and the same values are available as text for screen readers.
- On small screens the panels stack in this order: status, probabilities, viewer, notes.

## Visual design system

**Design principles**

1. Calm and precise: generous whitespace, a restrained palette, no decorative gradients or heavy shadows.
2. Data first: the probabilities and the waveform are the visual centre, and the surrounding chrome stays quiet.
3. Colour is never the only signal: every status pairs colour with an icon and a label.
4. Consistent: one spacing scale, one radius scale, one type scale.

**Colour tokens (proposed; verify contrast for every text and background pair)**

| Token | Value | Use |
| --- | --- | --- |
| `bg` | `#F6F8FA` | Page background |
| `surface` | `#FFFFFF` | Cards and panels |
| `border` | `#D8DEE4` | Hairlines and dividers |
| `ink` | `#0F1B2D` | Primary text |
| `muted` | `#5A6878` | Secondary text |
| `primary` | `#0B5C8F` | Buttons, links, selected states, ECG trace |
| `status-confident` | `#136B4B` | Confident badge |
| `status-uncertain` | `#8A5300` | Uncertain badge |
| `status-rejected` | `#A32A20` | Rejected badge |
| `ecg-grid-minor` and `ecg-grid-major` | `#F4DCDC` and `#E9B8B8` | Waveform background grid, like ECG paper |
| `attention` | `primary` at 14% opacity | Attention band over the waveform |

**Typography**

| Role | Choice |
| --- | --- |
| UI text | Inter, with the system UI font stack as fallback |
| Numbers, IDs, versions | A monospace such as JetBrains Mono, with tabular figures |
| Size scale | 12, 14, 16, 20, 28 and 40 px; line height 1.5 for body and 1.2 for headings |
| Weights | 400 body, 500 labels, 600 headings |

**Layout and components**

- 8 px spacing grid; content up to 1200 px wide; 12-column grid on desktop.
- Radius 8 px for controls and 12 px for cards; 1 px borders instead of shadows.
- Components: file dropzone, status badge, probability bar with threshold tick, ECG viewer, notice banner, primary and secondary buttons, skeleton loader, toast for non-blocking messages.
- Motion: short fades and a progress bar; no bouncing or auto-playing animation; honour reduced-motion settings.
- ECG viewer: grid from the `ecg-grid` tokens, trace 1.5 px in `primary`, lead names in the monospace, and a time scale bar.
- Dark theme: optional later. Tokens are named by role, so a dark set can be added without a redesign.

## Content and copy rules

**Voice:** plain, neutral and precise. Short sentences, no hype, no alarm.

| Do | Do not |
| --- | --- |
| "Flagged for review" | "Detected", "diagnosed", "you have" |
| "The model's attention" | "The reason" or "the cause" |
| "Not confident; manual review recommended" | "Maybe normal" |
| "Research prototype, not a medical device" | Any claim of clinical accuracy or approval |

**Required text**

- A disclaimer on every page: research prototype, not a medical device, not for diagnosis or treatment decisions.
- On the About page: intended use, training data (PTB-XL), an evaluation summary with confidence intervals, known limitations, and the model version. Current limitations include in-domain evaluation only, 12-lead input at 100 Hz, and the five superclasses as the only classes.
- Class names appear as the full label with its code, for example "Conduction disturbance (CD)".
- Every error message says what happened, why, and what to do next.

## Technical architecture and API contract

| Layer | Proposed choice |
| --- | --- |
| Frontend | React with Vite and TypeScript; Tailwind CSS with a small in-house component set |
| Charts | ECharts or Plotly.js for the viewer, with a custom SVG layer for the ECG-paper grid |
| Backend | FastAPI app in `webapp/backend` that loads the model bundle once at startup; no separate `ml_service` process |
| Inference | ONNX Runtime on CPU, using only the bundle's `model.onnx` |
| Packaging | Docker; one image serves the API and the built frontend, with the bundle copied in read-only |

**Endpoints**

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Multipart file upload with an optional `fs` field; returns the result below |
| `GET /sample` | Returns the result for the bundled sample ECG |
| `GET /health` | Liveness |
| `GET /model-info` | Model version, manifest hash, class list, thresholds, evaluation summary |

**`/predict` response**

```json
{
  "schema_version": "1",
  "model_version": "v1.0.0",
  "manifest_hash": "string",
  "status": "confident | uncertain | rejected",
  "reasons": ["string"],
  "quality_score": 0.0,
  "leads_used": 12,
  "classes": [
    {"code": "CD", "name": "Conduction disturbance", "probability": 0.0, "threshold": 0.0, "flagged": false}
  ],
  "signal": {"fs": 100, "unit": "mV | arbitrary", "leads": ["I", "II"], "samples": "[12][T] floats, filtered, original scale"},
  "attention": {"method": "cam", "per_class": "[5][T] floats in 0..1, one time profile per class"}
}
```

For `rejected` results, `classes` and `attention` are empty and `reasons` explains why.

**Errors**

| HTTP status | When | Body |
| --- | --- | --- |
| 400 | Unreadable or unsupported file | `{"code": "bad_file", "message": "..."}` |
| 413 | File over the size limit | `{"code": "too_large", "message": "..."}` |
| 422 | Missing field, such as sampling rate | `{"code": "missing_field", "field": "fs"}` |
| 504 | Processing exceeded the time limit | `{"code": "timeout", "message": "..."}` |

The attention output changed from the model PRD: the exported CAM gives one time profile per class, not a per-lead array, because the first convolution mixes all leads.

## Non-functional requirements

| Area | Requirement (proposed) |
| --- | --- |
| Performance | A result within a few seconds for a 10 s ECG on CPU; first meaningful paint under 2 s on a typical connection. Set final figures after the first measurement. |
| Reliability | A clear timeout and retry path; the API never returns a partial result |
| Security | HTTPS only; file type checked by content, not extension; size and rate limits; uploaded content is never executed; dependency scanning |
| Privacy | Process in memory and do not persist uploads; strip header identifiers; no analytics on file contents; request logs hold no signal data |
| Public demo rule | Only public PTB-XL sample records are used on any public deployment; real patient ECGs are never uploaded to it |
| Accessibility | Aim for WCAG 2.2 AA: keyboard-operable dropzone, visible focus, text alternatives and a data table for the waveform and bars, contrast checked for every token pair, reduced-motion support |
| Responsiveness | Desktop first; tablet and phone layouts stack the panels; the viewer scrolls horizontally rather than shrinking text |
| Browsers | Current versions of Chrome, Edge, Firefox and Safari |
| Observability | Structured logs with request id, timing, status and model version, and no patient data |

## Acceptance criteria and test plan

| Test | Pass criterion |
| --- | --- |
| Golden parity | For 20 PTB-XL test records, API probabilities match the offline bundle evaluation within about 1e-5 |
| Bad inputs | Empty file, wrong type, 8-lead file, flat line, NaN data, oversized file and unknown sampling rate each return a clear error and never crash |
| Status logic | A known low-quality file returns `rejected`; a known borderline file returns `uncertain` with the banner |
| Visual QA | Pages match the design tokens, with no layout break at 1440, 1024, 768 and 390 px widths |
| Accessibility | A keyboard-only run completes the full flow; an automated audit finds no serious issues; contrast is verified for every token pair |
| Privacy | No upload is persisted on disk; logs contain no signal values or header identifiers |
| Copy review | No result wording uses "diagnosis", "detected" or "you have" |
| Performance | Latency is recorded for 10 consecutive runs and compared with the proposed target |

**Definition of done for Part 1:** every test passes, the sample ECG works end to end, the About page is complete, and the disclaimer appears on every page.

## Milestones, risks and open questions

**Build order (rough estimates; revise after the first build)**

| Step | Deliverable | Estimate |
| --- | --- | --- |
| 1 | API wrapper around the bundle, plus the golden parity test | 2 to 3 days |
| 2 | Design tokens, component set and the upload page | 3 to 4 days |
| 3 | Results page and ECG viewer | 4 to 5 days |
| 4 | About page, error states and accessibility pass | 2 to 3 days |
| 5 | Docker image and a public demo with sample data only | 1 to 2 days |

**Risks**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Serving preprocessing drifts from training | Wrong probabilities | Reuse the training code and keep the golden parity test in CI |
| Waveform units mislead | Users misread amplitudes | Plot the original-scale signal and label units, or "arbitrary units" |
| Results over-read as diagnoses | Safety and credibility | Copy rules, visible disclaimer, status and threshold wording |
| Real-world files vary widely | Frequent rejections | Strict validation, clear errors, start with three formats |
| Private data reaches a public demo | Privacy breach | Public samples only, no persistence, header stripping |

**Open questions**

- [ ] Accepted file formats beyond WFDB zip, CSV and `.npy`
- [ ] Hosting target for the demo
- [ ] Product name and logo for the header
- [ ] Whether the JSON download gets a PDF version in Part 2
- [ ] Light theme only, or add a dark theme in Part 1
