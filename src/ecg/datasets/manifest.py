"""Stable dataset manifests and patient-group split assignment."""
import hashlib, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patient_splits(patient_ids: list[str], seed: int = 42,
                   fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> list[str]:
    if not np.isclose(sum(fractions), 1) or not patient_ids or any(not str(p).strip() for p in patient_ids):
        raise ValueError("Provide a patient id for each record and split fractions summing to one")
    groups = np.asarray(patient_ids)
    first = GroupShuffleSplit(n_splits=1, test_size=fractions[1] + fractions[2], random_state=seed)
    train, rest = next(first.split(groups, groups=groups))
    relative_test = fractions[2] / (fractions[1] + fractions[2])
    second = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1)
    val_local, test_local = next(second.split(rest, groups=groups[rest]))
    labels = np.empty(len(groups), dtype=object)
    labels[train] = "train"
    labels[rest[val_local]] = "validation"
    labels[rest[test_local]] = "test"
    return labels.tolist()


def write_manifest(records: list[dict], output: str | Path, metadata: dict) -> str:
    payload = {"metadata": metadata, "records": records}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    payload["manifest_hash"] = digest
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return digest
