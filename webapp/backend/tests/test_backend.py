"""Backend API tests: golden parity on 20 PTB-XL test records, plus bad-input validation."""
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from webapp.backend.app.main import app
from webapp.backend.app.model_loader import get_model_bundle
from webapp.backend.app.config import REPO_ROOT

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns liveness status and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "v1.0.0"


def test_model_info_endpoint():
    """Verify GET /model-info returns complete bundle metadata and limitations."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0.0"
    assert len(data["classes"]) == 5
    assert len(data["thresholds"]) == 5
    assert len(data["temperatures"]) == 5
    assert "test_metrics" in data
    assert len(data["limitations"]) > 0


def test_sample_endpoint():
    """Verify GET /sample runs the bundled sample record and returns valid prediction."""
    response = client.get("/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "1"
    assert data["model_version"] == "v1.0.0"
    assert data["status"] in ["confident", "uncertain"]
    assert len(data["classes"]) == 5
    assert data["leads_used"] == 12
    assert data["signal"] is not None
    assert len(data["signal"]["samples"]) == 12
    assert len(data["signal"]["samples"][0]) == 1000
    assert data["attention"] is not None
    assert len(data["attention"]["per_class"]) == 5


def test_golden_parity_20_records():
    """Golden parity: POST /predict probabilities match offline ONNX bundle within 1e-4 across 20 test records."""
    import pandas as pd
    import wfdb
    
    bundle = get_model_bundle()
    csv_path = REPO_ROOT / "data" / "raw" / "files" / "ptb-xl" / "1.0.3" / "ptbxl_database.csv"
    if not csv_path.exists():
        pytest.skip("Raw PTB-XL database not found")
        
    df = pd.read_csv(csv_path, index_col="ecg_id")
    test_df = df[df.strat_fold == 10].head(20)

    from ecg.preprocess.signal import preprocess, PreprocessConfig
    prep_cfg = PreprocessConfig(sampling_rate=100, duration_seconds=10)

    for ecg_id, row in test_df.iterrows():
        record_path = REPO_ROOT / "data" / "raw" / "files" / "ptb-xl" / "1.0.3" / row.filename_lr
        rec = wfdb.rdrecord(str(record_path))
        raw_signal = rec.p_signal.astype(np.float64)
        lead_names = [str(n).strip() for n in rec.sig_name]
        fs = int(rec.fs)

        # 1. Offline ground-truth
        x_pre, lm = preprocess(raw_signal, fs, lead_names, prep_cfg)
        feed = {
            bundle.input_names[0]: x_pre[None, :, :].astype(np.float32),
            bundle.input_names[1]: lm[None, :].astype(np.float32)
        }
        raw_logits = bundle.session.run(None, feed)[0][0]
        scaled = raw_logits / bundle.temperatures
        expected_probs = 1.0 / (1.0 + np.exp(-scaled))

        # 2. Package raw signal into .npy and upload to /predict
        buf = io.BytesIO()
        np.save(buf, raw_signal)
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"file": (f"{ecg_id}.npy", buf.getvalue(), "application/octet-stream")},
            data={"fs": fs}
        )
        assert response.status_code == 200, f"Failed on record {ecg_id}: {response.text}"
        res_json = response.json()

        actual_probs = np.array([c["probability"] for c in res_json["classes"]])
        max_diff = float(np.max(np.abs(actual_probs - expected_probs)))
        
        # Matches offline bundle within rounding precision (probabilities rounded to 4 decimals)
        assert max_diff <= 1e-4, f"Parity mismatch on test record {ecg_id}: max diff {max_diff}"


def test_bad_input_empty_file():
    """Verify empty upload returns 400 bad_file."""
    response = client.post(
        "/predict",
        files={"file": ("empty.npy", b"", "application/octet-stream")},
        data={"fs": 100}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_file"


def test_bad_input_unsupported_format():
    """Verify unsupported file extension returns 400 bad_file."""
    response = client.post(
        "/predict",
        files={"file": ("report.pdf", b"%PDF-1.4 dummy", "application/pdf")},
        data={"fs": 100}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_file"


def test_bad_input_missing_fs_on_csv():
    """Verify CSV without fs returns 422 missing_field."""
    csv_content = b"I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n" + b"0,0,0,0,0,0,0,0,0,0,0,0\n" * 1000
    response = client.post(
        "/predict",
        files={"file": ("ecg.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "missing_field"
    assert response.json()["field"] == "fs"


def test_bad_input_flat_line():
    """Verify flatline input returns 400 bad_file."""
    flat_data = np.zeros((1000, 12), dtype=np.float64)
    buf = io.BytesIO()
    np.save(buf, flat_data)
    response = client.post(
        "/predict",
        files={"file": ("flat.npy", buf.getvalue(), "application/octet-stream")},
        data={"fs": 100}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_file"
    assert "flat" in response.json()["message"].lower()


def test_bad_input_short_duration():
    """Verify short record (< 10 seconds) returns 400 bad_file."""
    short_data = np.random.randn(500, 12).astype(np.float64)  # 5 seconds at 100 Hz
    buf = io.BytesIO()
    np.save(buf, short_data)
    response = client.post(
        "/predict",
        files={"file": ("short.npy", buf.getvalue(), "application/octet-stream")},
        data={"fs": 100}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_file"
    assert "under" in response.json()["message"].lower() or "duration" in response.json()["message"].lower()


def test_bad_input_nan_values():
    """Verify NaN values return 400 bad_file."""
    nan_data = np.random.randn(1000, 12).astype(np.float64)
    nan_data[50, 2] = np.nan
    buf = io.BytesIO()
    np.save(buf, nan_data)
    response = client.post(
        "/predict",
        files={"file": ("nan.npy", buf.getvalue(), "application/octet-stream")},
        data={"fs": 100}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "bad_file"
    assert "nan" in response.json()["message"].lower()


def test_quality_floor_rejection():
    """Verify input with degraded lead coverage (e.g. 10 flat/disconnected leads) gets rejected with explanation."""
    # 10 flat leads, 2 active leads -> quality score = 0.0045 < 0.25 floor
    signal = np.random.randn(1000, 12).astype(np.float64)
    signal[:, 2:] = 0.0  # leads 2 to 11 are flatline
    buf = io.BytesIO()
    np.save(buf, signal)
    response = client.post(
        "/predict",
        files={"file": ("degraded.npy", buf.getvalue(), "application/octet-stream")},
        data={"fs": 100}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert len(data["classes"]) == 0
    assert data["attention"] is None
    assert len(data["reasons"]) > 0
