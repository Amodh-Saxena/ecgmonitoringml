"""FastAPI backend application for ECG Triage Web App."""
import asyncio
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import numpy as np

from .config import MAX_UPLOAD_BYTES, TIMEOUT_SECONDS, REPO_ROOT
from .model_loader import get_model_bundle, ModelBundle
from .schemas import PredictResponse, HealthResponse, ModelInfoResponse, ErrorResponse
from .parsing import (
    parse_wfdb_zip,
    parse_csv_signal,
    parse_npy_signal,
    validate_raw_signal
)
from .inference import run_pipeline

app = FastAPI(
    title="ECG Triage API",
    description="Production-grade clinical ECG abnormality triage API",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load model bundle once into memory at startup."""
    get_model_bundle()


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTP errors to match PRD error contract."""
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": "error", "message": str(exc.detail)}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle missing fields such as fs with PRD 422 schema."""
    errors = exc.errors()
    field = errors[0]["loc"][-1] if errors and "loc" in errors[0] else "unknown"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"code": "missing_field", "field": str(field), "message": f"Required field '{field}' is missing"}
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness probe returning server status and model version."""
    bundle = get_model_bundle()
    return HealthResponse(status="ok", model_version=bundle.model_version)


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Return model version, manifest hash, class list, thresholds, and evaluation summary."""
    bundle = get_model_bundle()
    return ModelInfoResponse(
        model_version=bundle.model_version,
        manifest_hash=bundle.manifest_hash,
        classes=bundle.classes,
        thresholds=[round(float(t), 4) for t in bundle.thresholds],
        temperatures=[round(float(t), 4) for t in bundle.temperatures],
        test_metrics=bundle.test_metrics,
        license=bundle.license,
        intended_use=(
            "Research prototype for 12-lead ECG abnormality triage. "
            "Flagged classes are recommendations for clinical review, not automated diagnoses."
        ),
        limitations=[
            "Evaluated strictly on in-domain PTB-XL holdout fold; performance under external clinical domain shift may vary.",
            "Standard input is 12-lead ECG sampled at 100 Hz over a 10-second duration.",
            "Detects 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP); arrhythmias outside these groups require manual review."
        ]
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_ecg(
    file: UploadFile = File(...),
    fs: Optional[int] = Form(None)
):
    """Upload an ECG file (WFDB zip, CSV, or NPY) and receive abnormality triage analysis."""
    bundle = get_model_bundle()
    
    # Read upload contents with size check
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "too_large", "message": f"Uploaded file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"}
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": "Uploaded file is empty (0 bytes)"}
        )

    filename = (file.filename or "").lower()
    
    # Execute parsing within timeout limit
    try:
        if filename.endswith(".zip"):
            raw_signal, lead_names, detected_fs, units = parse_wfdb_zip(contents)
        elif filename.endswith(".csv"):
            raw_signal, lead_names, detected_fs, units = parse_csv_signal(contents, fs)
        elif filename.endswith(".npy"):
            raw_signal, lead_names, detected_fs, units = parse_npy_signal(contents, fs)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "bad_file", "message": f"Unsupported file extension in '{file.filename}'. Allowed: .zip (WFDB), .csv, .npy"}
            )
            
        validate_raw_signal(raw_signal, detected_fs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": f"Failed to parse file: {str(e)}"}
        )

    # Run inference pipeline
    try:
        result = run_pipeline(raw_signal, lead_names, detected_fs, units, bundle)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "inference_error", "message": f"Inference processing failed: {str(e)}"}
        )


@app.get("/sample", response_model=PredictResponse)
def get_sample_ecg():
    """Return analysis of the bundled public PTB-XL sample ECG record."""
    bundle = get_model_bundle()
    
    # Search for record 00001_lr in raw dataset or cache
    sample_hea = REPO_ROOT / "data" / "raw" / "files" / "ptb-xl" / "1.0.3" / "records100" / "00000" / "00001_lr.hea"
    sample_prefix = str(sample_hea)[:-4]
    
    if sample_hea.exists():
        import wfdb
        record = wfdb.rdrecord(sample_prefix)
        raw_signal = np.asarray(record.p_signal, dtype=np.float64)
        lead_names = [str(n).strip() for n in record.sig_name]
        source_fs = int(record.fs)
        units = str(record.units[0]) if record.units else "mV"
    else:
        # Fallback to cached signal if raw WFDB is not at default path
        cache_signals = REPO_ROOT / "data" / "cache" / "81f46ccd594df21f" / "signals.npy"
        if cache_signals.exists():
            sig = np.load(cache_signals, mmap_mode="r")[0]  # [12, 1000]
            raw_signal = sig.T.astype(np.float64)
            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            source_fs = 100
            units = "mV"
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "sample_not_found", "message": "Bundled sample ECG record not found"}
            )

    return run_pipeline(raw_signal, lead_names, source_fs, units, bundle)
