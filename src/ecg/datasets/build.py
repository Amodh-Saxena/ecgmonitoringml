"""Dataset build: records -> label mapping (+audit) -> preprocessing -> splits -> memmap cache.

Cache layout: <cache_dir>/<manifest_hash>/{signals,lead_masks,targets,target_masks}.npy + manifest.json
Splits are assigned once per dataset version and are re-used, never silently regenerated.
"""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

from ecg.datasets.manifest import patient_splits, sha256_file, write_manifest
from ecg.labels.mapping import class_ids, map_codes, unmapped_codes
from ecg.preprocess.augment import augment
from ecg.preprocess.signal import PreprocessConfig, preprocess

READERS = {"ptbxl": "ecg.datasets.ptbxl:iter_ptbxl", "cinc": "ecg.datasets.cinc:iter_cinc",
           "chapman": "ecg.datasets.chapman:iter_chapman", "ningbo": "ecg.datasets.chapman:iter_ningbo"}


def get_reader(source: str):
    module, name = READERS[source].split(":")
    return getattr(__import__(module, fromlist=[name]), name)


def preprocess_config_from(data_cfg) -> PreprocessConfig:
    return PreprocessConfig(sampling_rate=data_cfg.sampling_rate, duration_seconds=data_cfg.duration_seconds,
                            low_hz=data_cfg.low_hz, high_hz=data_cfg.high_hz, notch_hz=data_cfg.notch_hz,
                            normalization=data_cfg.normalization)


def audit_labels(records_meta: list[dict], spec: dict, source: str) -> dict:
    """Audit report written before any training: counts, unmapped codes, patient overlap."""
    classes = class_ids(spec)
    per_split: dict[str, dict] = {}
    for r in records_meta:
        s = per_split.setdefault(r["split"], {"records": 0, "positives": [0] * len(classes), "known": [0] * len(classes)})
        s["records"] += 1
        for k in range(len(classes)):
            s["positives"][k] += int(r["y"][k] * r["m"][k])
            s["known"][k] += int(r["m"][k])
    patients_by_split = {}
    for r in records_meta:
        patients_by_split.setdefault(r["split"], set()).add(r["patient_id"])
    names = sorted(patients_by_split)
    overlap = {f"{a}&{b}": len(patients_by_split[a] & patients_by_split[b])
               for i, a in enumerate(names) for b in names[i + 1:]}
    unmapped = unmapped_codes(source, [r["codes"] for r in records_meta], spec)
    return {"source": source, "label_version": spec["version"], "classes": classes,
            "per_split": per_split, "patient_overlap": overlap,
            "unmapped_codes": dict(unmapped.most_common()),
            "missing_patient_id": sum(r["patient_id"].startswith("missing-patient:") for r in records_meta)}


def build_dataset(source: str, root, spec: dict, pre_cfg: PreprocessConfig, cache_dir, split_seed: int = 42,
                  limit: int | None = None, use_official_folds: bool = True, hash_files: bool = True) -> Path:
    """Build (or find) the cached dataset; returns the directory keyed by the manifest hash."""
    reader = get_reader(source)
    kwargs = {"sampling_rate": pre_cfg.sampling_rate} if source == "ptbxl" else {}
    n_leads, length = 12, pre_cfg.sampling_rate * pre_cfg.duration_seconds
    pending = Path(cache_dir) / f"_building_{source}"
    pending.mkdir(parents=True, exist_ok=True)
    signals, lmasks, ys, ms, meta = [], [], [], [], []
    for rec in reader(root, limit=limit, **kwargs):
        x, lm = preprocess(rec.signal, rec.sampling_rate, rec.lead_names, pre_cfg)
        y, m = map_codes(source, rec.raw_label_codes, spec)
        signals.append(x); lmasks.append(lm); ys.append(y); ms.append(m)
        if len(signals) % 2000 == 0:
            print(f"  processed {len(signals)} records", flush=True)
        meta.append({"record_id": rec.record_id, "patient_id": rec.patient_id, "source": source,
                     "codes": rec.raw_label_codes, "y": y.tolist(), "m": m.tolist(),
                     "fold": rec.meta.get("fold"), "age": rec.meta.get("age"), "sex": rec.meta.get("sex"),
                     "file_hashes": [sha256_file(f) for f in rec.meta.get("files", []) if Path(f).exists()] if hash_files else []})
    if not meta:
        raise ValueError(f"No records found for {source} under {root}")
    folds = [r["fold"] for r in meta]
    if source == "ptbxl" and use_official_folds and all(f is not None for f in folds):
        splits = ["test" if f == 10 else "validation" if f == 9 else "train" for f in folds]
    else:
        splits = patient_splits([r["patient_id"] for r in meta], split_seed)
    for r, s in zip(meta, splits):
        r["split"] = s
    audit = audit_labels(meta, spec, source)
    if any(v for v in audit["patient_overlap"].values()):
        raise ValueError(f"Patient leakage across splits: {audit['patient_overlap']}")
    metadata = {"source": source, "label_version": spec["version"], "preprocess": asdict(pre_cfg),
                "split_seed": split_seed, "counts": {k: v["records"] for k, v in audit["per_split"].items()}}
    digest = write_manifest(meta, pending / "manifest.json", metadata)
    out = Path(cache_dir) / digest[:16]
    if (out / "manifest.json").exists():  # same dataset version: reuse, keep stored splits
        return out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "signals.npy", np.stack(signals).astype(np.float32))
    np.save(out / "lead_masks.npy", np.stack(lmasks)); np.save(out / "targets.npy", np.stack(ys))
    np.save(out / "target_masks.npy", np.stack(ms))
    (pending / "manifest.json").replace(out / "manifest.json")
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def load_manifest(cache_path) -> dict:
    return json.loads((Path(cache_path) / "manifest.json").read_text(encoding="utf-8"))


class ECGArrayDataset:
    """Torch-compatible dataset over the memory-mapped cache. Augmentation only on 'train'."""

    def __init__(self, cache_path, split: str | None = None, augmentation: dict | None = None, seed: int = 0,
                 leads: list[str] | None = None, label_fraction: float = 1.0, extra_indices=None):
        from ecg.preprocess.signal import CANONICAL_LEADS
        self.path = Path(cache_path)
        self.manifest = load_manifest(self.path)
        self.signals = np.load(self.path / "signals.npy", mmap_mode="r")
        self.lead_masks = np.load(self.path / "lead_masks.npy", mmap_mode="r")
        self.targets = np.load(self.path / "targets.npy", mmap_mode="r")
        self.target_masks = np.load(self.path / "target_masks.npy", mmap_mode="r")
        records = self.manifest["records"]
        idx = np.arange(len(records)) if split is None else np.array([i for i, r in enumerate(records) if r["split"] == split])
        if label_fraction < 1.0 and split == "train" and len(idx):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, max(1, int(len(idx) * label_fraction)), replace=False))
        self.indices = idx
        self.records = [records[i] for i in idx]
        self.augmentation = augmentation if split == "train" else None  # never touch val/test
        self.rng = np.random.default_rng(seed)
        self.keep = None
        if leads:
            self.keep = np.array([1.0 if l in leads else 0.0 for l in CANONICAL_LEADS], dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x, lm = np.array(self.signals[j]), np.array(self.lead_masks[j])
        if self.keep is not None:
            lm = lm * self.keep; x = x * lm[:, None]
        if self.augmentation:
            x, lm = augment(x, lm, self.rng, **self.augmentation)
        return x, lm, np.array(self.targets[j]), np.array(self.target_masks[j])

    def patient_ids(self):
        return [r["patient_id"] for r in self.records]
