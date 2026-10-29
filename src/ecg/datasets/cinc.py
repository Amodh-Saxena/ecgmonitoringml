"""PhysioNet/CinC 2020-21 style WFDB reader (header comments carry Dx SNOMED codes)."""
from pathlib import Path
from ecg.datasets.base import ECGRecord, require_patient_id, source_root


def _comment(comments, key):
    for line in comments:
        line = line.lstrip("#").strip()
        if line.lower().startswith(key.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return None


def iter_wfdb_dx(root, source: str, limit: int | None = None, patient_from_record: bool = False):
    """Yield records from every .hea under root. CinC has no patient linkage -> unique ids."""
    import wfdb
    root = source_root(root)
    for n, hea in enumerate(sorted(root.rglob("*.hea"))):
        if limit is not None and n >= limit:
            break
        base = hea.with_suffix("")
        signal, fields = wfdb.rdsamp(str(base))
        comments = fields.get("comments", [])
        dx = _comment(comments, "Dx") or ""
        age, sex = _comment(comments, "Age"), _comment(comments, "Sex")
        yield ECGRecord(
            signal=signal, sampling_rate=int(fields["fs"]), lead_names=[s.strip() for s in fields["sig_name"]],
            record_id=f"{source}:{base.name}",
            patient_id=require_patient_id(base.name if patient_from_record else None, f"{source}:{base.name}"),
            source=source, raw_label_codes=[c.strip() for c in dx.split(",") if c.strip()],
            meta={"age": float(age) if age and age.replace(".", "").isdigit() else None,
                  "sex": 1.0 if (sex or "").lower().startswith("m") else (0.0 if sex else None),
                  "files": [str(hea), str(base) + ".mat"]})


def iter_cinc(root, limit: int | None = None):
    yield from iter_wfdb_dx(root, "cinc", limit)
