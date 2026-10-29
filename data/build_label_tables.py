"""Regenerate the harmonized label tables in src/ecg/labels/.

Rows are drafts built from the PTB-XL statement table and the PhysioNet/CinC 2020-21
SNOMED-CT scored-code list. Every row carries reviewer="unreviewed" until a human signs it off.
Run:  python data/build_label_tables.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "ecg" / "labels"

PTBXL = {
    "NORM": ["NORM"],
    "MI": ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI", "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL"],
    "STTC": ["NDT", "NST_", "DIG", "LNGQT", "ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ANEUR", "EL", "ISCAN"],
    "CD": ["LAFB", "IRBBB", "1AVB", "IVCD", "CRBBB", "CLBBB", "LPFB", "WPW", "ILBBB", "3AVB", "2AVB"],
    "HYP": ["LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP"],
}
# (SNOMED code, description, v1 superclass, rule)
SNOMED = [
    ("426783006", "Sinus rhythm (normal ECG)", "NORM", "exact"),
    ("164865005", "Myocardial infarction", "MI", "exact"),
    ("164861001", "Myocardial ischemia", "STTC", "parent"),
    ("429622005", "ST segment depression", "STTC", "exact"),
    ("164930006", "ST interval abnormal", "STTC", "exact"),
    ("164934002", "T wave abnormal", "STTC", "exact"),
    ("111975006", "Prolonged QT interval", "STTC", "exact"),
    ("270492004", "First degree AV block", "CD", "exact"),
    ("164947007", "Prolonged PR interval", "CD", "equivalent"),
    ("164909002", "Left bundle branch block", "CD", "parent"),
    ("59118001", "Right bundle branch block", "CD", "parent"),
    ("713427006", "Complete right bundle branch block", "CD", "equivalent"),
    ("713426002", "Incomplete right bundle branch block", "CD", "parent"),
    ("445118002", "Left anterior fascicular block", "CD", "exact"),
    ("698252002", "Nonspecific intraventricular conduction disorder", "CD", "exact"),
    ("164873001", "Left ventricular hypertrophy", "HYP", "exact"),
    ("446358003", "Right atrial hypertrophy", "HYP", "exact"),
]
V2_PTBXL = {"AF": ["AFIB"], "LBBB": ["CLBBB", "ILBBB"], "RBBB": ["CRBBB", "IRBBB"]}
V2_SNOMED = [("164889003", "Atrial fibrillation", "AF", "exact"),
             ("164909002", "Left bundle branch block", "LBBB", "exact"),
             ("59118001", "Right bundle branch block", "RBBB", "exact"),
             ("713427006", "Complete right bundle branch block", "RBBB", "equivalent")]
NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
         "CD": "Conduction disturbance", "HYP": "Hypertrophy", "AF": "Atrial fibrillation",
         "LBBB": "Left bundle branch block", "RBBB": "Right bundle branch block"}


def row(source, code, desc, target, rule):
    return {"source": source, "source_code": code, "description": desc, "target_class": target,
            "rule": rule, "reviewer": "unreviewed", "notes": ""}


def build(version, classes, extra_ptbxl, extra_snomed, drop_superclasses=()):
    rows = []
    for target, codes in PTBXL.items():
        if target in drop_superclasses:
            continue
        rows += [row("ptbxl", c, f"SCP statement {c}", target, "exact") for c in codes]
    for target, codes in extra_ptbxl.items():
        rows += [row("ptbxl", c, f"SCP statement {c}", target, "exact") for c in codes]
    snomed = [s for s in SNOMED if s[2] in classes] + list(extra_snomed)
    for src in ("cinc", "chapman", "ningbo"):
        rows += [row(src, *s) for s in snomed]
    return {"version": version, "classes": [{"id": c, "display_name": NAMES[c]} for c in classes],
            "mappings": rows,
            "mapping_note": "A class a source never maps to is masked (m=0), never a silent negative."}


if __name__ == "__main__":
    v1 = ["NORM", "MI", "STTC", "CD", "HYP"]
    (OUT / "labels.json").write_text(json.dumps(build("1.0.0", v1, {}, []), indent=2), encoding="utf-8")
    v2 = v1 + ["AF", "LBBB", "RBBB"]
    (OUT / "labels_v2.json").write_text(json.dumps(build("2.0.0", v2, V2_PTBXL, V2_SNOMED), indent=2), encoding="utf-8")
    print("wrote", OUT)
