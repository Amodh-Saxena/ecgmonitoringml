"""File parsing and validation for WFDB zip, CSV, and NPY uploads."""
import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import wfdb
from fastapi import HTTPException, status

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def sanitize_header_text(header_text: str) -> str:
    """Strip patient-identifying fields (name, age, sex, date, comments) from WFDB header string."""
    cleaned_lines = []
    for line in header_text.splitlines():
        # Keep channel lines and basic record header line, strip identifying comments
        if line.startswith("#"):
            # Strip lines containing age, sex, date, name or general patient metadata
            lower = line.lower()
            if any(k in lower for k in ["age", "sex", "patient", "date", "name", "id:", "hospital"]):
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def parse_wfdb_zip(file_bytes: bytes) -> Tuple[np.ndarray, List[str], int, str]:
    """Extract and parse WFDB zip archive containing .hea and .dat files.
    
    Returns:
        (signal [samples, leads], lead_names, fs, unit)
    """
    try:
        z = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": f"Invalid zip archive: {str(e)}"}
        )

    namelist = z.namelist()
    hea_files = [f for f in namelist if f.endswith(".hea")]
    if not hea_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": "WFDB zip must contain a .hea header file"}
        )

    # Use the first .hea found
    target_hea = hea_files[0]
    base_stem = target_hea[:-4]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        z.extractall(tmp_path)
        
        record_path = tmp_path / base_stem
        try:
            record = wfdb.rdrecord(str(record_path))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "bad_file", "message": f"Failed to parse WFDB record: {str(e)}"}
            )
            
        if record.p_signal is None or record.p_signal.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "bad_file", "message": "WFDB record contains no physical signal data"}
            )
            
        signal = np.asarray(record.p_signal, dtype=np.float64)
        lead_names = [str(name).strip() for name in record.sig_name] if record.sig_name else CANONICAL_LEADS[:signal.shape[1]]
        fs = int(record.fs) if record.fs else 100
        units = str(record.units[0]) if record.units and len(record.units) > 0 else "mV"
        
        return signal, lead_names, fs, units


def parse_csv_signal(file_bytes: bytes, user_fs: Optional[int] = None) -> Tuple[np.ndarray, List[str], int, str]:
    """Parse CSV signal data.
    
    Returns:
        (signal [samples, leads], lead_names, fs, unit)
    """
    if user_fs is None or user_fs <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "missing_field", "field": "fs", "message": "Sampling rate (fs) is required for CSV signal files"}
        )

    try:
        # Try reading header
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": f"Unable to parse CSV file: {str(e)}"}
        )

    if df.empty or df.shape[1] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": "CSV file contains no tabular data"}
        )

    # Check if first column is time/index
    first_col_name = str(df.columns[0]).strip().lower()
    if first_col_name in ["time", "t", "timestamp", "sample", "index"]:
        df = df.drop(columns=[df.columns[0]])

    # Lead names from columns or fallback
    lead_names = [str(c).strip() for c in df.columns]
    signal = df.to_numpy(dtype=np.float64)
    
    return signal, lead_names, int(user_fs), "mV"


def parse_npy_signal(file_bytes: bytes, user_fs: Optional[int] = None) -> Tuple[np.ndarray, List[str], int, str]:
    """Parse .npy signal data.
    
    Returns:
        (signal [samples, leads], lead_names, fs, unit)
    """
    if user_fs is None or user_fs <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "missing_field", "field": "fs", "message": "Sampling rate (fs) is required for NPY signal files"}
        )

    try:
        arr = np.load(io.BytesIO(file_bytes), allow_pickle=False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": f"Unable to load .npy array: {str(e)}"}
        )

    if arr.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": f"Expected 2D array [samples, leads] or [leads, samples], got ndim={arr.ndim}"}
        )

    # If [12, samples] where samples > 12, transpose to [samples, 12]
    if arr.shape[0] <= 12 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    signal = arr.astype(np.float64)
    leads_count = signal.shape[1]
    lead_names = CANONICAL_LEADS[:leads_count] if leads_count <= len(CANONICAL_LEADS) else [f"Lead_{i}" for i in range(leads_count)]

    return signal, lead_names, int(user_fs), "arbitrary"


def validate_raw_signal(signal: np.ndarray, fs: int) -> None:
    """Validate signal bounds, length, and sanity per PRD requirement P2."""
    if np.isnan(signal).any() or np.isinf(signal).any():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": "Signal contains invalid NaN or infinite values"}
        )

    samples, leads = signal.shape
    duration = samples / float(fs)
    if duration < 9.5:  # Require at least 10s of data
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": f"Signal duration ({duration:.2f}s) is under the required 10 seconds"}
        )

    # Check for flat line across all leads
    lead_stds = np.std(signal, axis=0)
    if (lead_stds < 1e-6).all():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_file", "message": "Signal is completely flat (no physiological electrical activity detected)"}
        )
