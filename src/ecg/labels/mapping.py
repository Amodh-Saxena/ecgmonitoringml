"""Validated conversion from source codes to masked multi-label targets."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

DEFAULT_LABELS = Path(__file__).parent / "labels.json"


def load_label_spec(path: str | Path | None = None) -> dict:
    spec = json.loads(Path(path or DEFAULT_LABELS).read_text(encoding="utf-8"))
    ids = [c["id"] for c in spec["classes"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Label class ids must be non-empty and unique")
    seen = set()
    for row in spec.get("mappings", []):
        if row.get("target_class") not in {*ids, "unmapped"}:
            raise ValueError(f"Unknown target class in mapping: {row}")
        if not all(row.get(k) for k in ("source", "source_code", "rule", "reviewer")):
            raise ValueError(f"Incomplete mapping row: {row}")
        if row["rule"] not in ("exact", "parent", "equivalent"):
            raise ValueError(f"Unknown rule in mapping row: {row}")
        key = (row["source"], str(row["source_code"]), row["target_class"])
        if key in seen:
            raise ValueError(f"Duplicate mapping row: {row}")
        seen.add(key)
    return spec


def class_ids(spec: dict) -> list[str]:
    return [c["id"] for c in spec["classes"]]


def covered_classes(source: str, spec: dict) -> set[str]:
    """Classes the source can speak about = classes it has at least one mapping row for."""
    return {r["target_class"] for r in spec.get("mappings", [])
            if r["source"] == source and r["target_class"] != "unmapped"}


def map_codes(source: str, codes: list[str], spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (target y, known-mask m).

    m=1 only for classes the source covers, and only when the record carries at least one
    mapped code; otherwise the record says nothing and every class stays masked. A class the
    source never maps to is always masked, never a silent negative.
    """
    classes = class_ids(spec)
    y, mask = np.zeros(len(classes), dtype=np.float32), np.zeros(len(classes), dtype=np.float32)
    by_code = {}
    for r in spec.get("mappings", []):
        if r["source"] == source:
            by_code.setdefault(str(r["source_code"]), []).append(r["target_class"])
    hit = False
    for code in codes:
        for target in by_code.get(str(code), []):
            if target in classes:
                y[classes.index(target)] = 1.0
                hit = True
    if hit:
        for target in covered_classes(source, spec):
            mask[classes.index(target)] = 1.0
    return y, mask


def unmapped_codes(source: str, code_lists: list[list[str]], spec: dict) -> Counter:
    known = {str(r["source_code"]) for r in spec.get("mappings", [])
             if r["source"] == source and r["target_class"] != "unmapped"}
    return Counter(str(c) for codes in code_lists for c in codes if str(c) not in known)
