"""Common record representation used by source-specific readers."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import numpy as np


@dataclass
class ECGRecord:
    signal: np.ndarray
    sampling_rate: int
    lead_names: list[str]
    record_id: str
    patient_id: str
    source: str
    raw_label_codes: list[str]
    meta: dict = field(default_factory=dict)  # fold, age, sex, files (for hashing)


class ECGDataset(Protocol):
    def __iter__(self): ...


def require_patient_id(patient_id: str | None, record_id: str) -> str:
    """Use a record-unique fallback if patient linkage is absent; surface it for audit."""
    return str(patient_id).strip() if patient_id and str(patient_id).strip() else f"missing-patient:{record_id}"


def source_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}. Acquire the dataset and configure its path first.")
    return root
