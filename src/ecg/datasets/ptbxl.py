"""PTB-XL reader (records100/records500 + ptbxl_database.csv + scp_statements.csv)."""
import ast
import csv
from pathlib import Path
from ecg.datasets.base import ECGRecord, require_patient_id, source_root


def iter_ptbxl(root, sampling_rate: int = 100, limit: int | None = None, min_likelihood: float = 0.0):
    import wfdb
    root = source_root(root)
    column = "filename_lr" if sampling_rate <= 100 else "filename_hr"
    with open(root / "ptbxl_database.csv", newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f)):
            if limit is not None and n >= limit:
                break
            record_id = row["ecg_id"]
            base = root / row[column]
            signal, fields = wfdb.rdsamp(str(base))
            codes = [c for c, p in ast.literal_eval(row["scp_codes"] or "{}").items() if float(p) >= min_likelihood]
            yield ECGRecord(
                signal=signal, sampling_rate=int(fields["fs"]), lead_names=list(fields["sig_name"]),
                record_id=f"ptbxl:{record_id}",
                patient_id=require_patient_id(row.get("patient_id", "").split(".")[0], record_id),
                source="ptbxl", raw_label_codes=codes,
                meta={"fold": int(float(row["strat_fold"])) if row.get("strat_fold") else None,
                      "age": _num(row.get("age")), "sex": _num(row.get("sex")),
                      "validated_by_human": row.get("validated_by_human", "").lower() == "true",
                      "files": [str(base) + ".hea", str(base) + ".dat"]})


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_scp_statements(root) -> dict[str, str]:
    """The dataset's own statement table: SCP code -> diagnostic superclass."""
    out = {}
    with open(Path(root) / "scp_statements.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("diagnostic_class"):
                out[row[""] or row.get("Unnamed: 0", "")] = row["diagnostic_class"]
    return out
