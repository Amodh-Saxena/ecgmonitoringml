"""Step 1: label matrix + mask from PTB-XL metadata only (no signals are read)."""
import ast
import csv
from collections import Counter
from pathlib import Path
import numpy as np

from ecg.labels.mapping import class_ids, map_codes, unmapped_codes


def read_metadata(root) -> list[dict]:
    root = Path(root)
    rows = []
    with open(root / "ptbxl_database.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"ecg_id": r["ecg_id"], "patient_id": str(r["patient_id"]).split(".")[0],
                         "fold": int(float(r["strat_fold"])), "codes": list(ast.literal_eval(r["scp_codes"] or "{}"))})
    return rows


def scp_diagnostic_classes(root) -> dict[str, str]:
    """The dataset's own statement table: SCP code -> diagnostic superclass (diagnostic rows only)."""
    out = {}
    with open(Path(root) / "scp_statements.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("diagnostic_class"):
                out[r[""]] = r["diagnostic_class"]
    return out


def crosscheck_mapping(root, spec) -> list[str]:
    """Disagreements between our mapping rows and the dataset's statement table."""
    ours = {r["source_code"]: r["target_class"] for r in spec["mappings"] if r["source"] == "ptbxl"}
    theirs = scp_diagnostic_classes(root)
    issues = [f"{c}: ours={ours.get(c)} dataset={t}" for c, t in theirs.items()
              if t in class_ids(spec) and ours.get(c) != t]
    issues += [f"{c}: mapped by us but not a diagnostic statement in dataset" for c in ours if c not in theirs]
    return issues


def build_label_matrix(root, spec, out_path) -> dict:
    rows = read_metadata(root)
    classes = class_ids(spec)
    Y = np.zeros((len(rows), len(classes)), np.float32)
    M = np.zeros_like(Y)
    for i, r in enumerate(rows):
        Y[i], M[i] = map_codes("ptbxl", r["codes"], spec)
    folds = np.array([r["fold"] for r in rows])
    split = np.where(folds == 10, "test", np.where(folds == 9, "validation", "train"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, y=Y, mask=M, ecg_id=np.array([r["ecg_id"] for r in rows]),
                        patient_id=np.array([r["patient_id"] for r in rows]), fold=folds, split=split,
                        classes=np.array(classes), label_version=spec["version"])
    pos = (Y * M).sum(0).astype(int)
    per_split = {s: {c: int((Y[split == s, k] * M[split == s, k]).sum()) for k, c in enumerate(classes)}
                 for s in ("train", "validation", "test")}
    patients = {s: {r["patient_id"] for r, sp in zip(rows, split) if sp == s} for s in ("train", "validation", "test")}
    overlap = {f"{a}&{b}": len(patients[a] & patients[b]) for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test"))}
    return {"n_records": len(rows), "classes": classes, "positives": dict(zip(classes, pos.tolist())),
            "labeled_records": int((M.sum(1) > 0).sum()), "unlabeled_records": int((M.sum(1) == 0).sum()),
            "multi_label_records": int(((Y * M).sum(1) > 1).sum()), "per_split": per_split,
            "split_sizes": {s: int((split == s).sum()) for s in per_split}, "patient_overlap": overlap,
            "unmapped_codes": dict(unmapped_codes("ptbxl", [r["codes"] for r in rows], spec).most_common(15))}
