import json
import numpy as np
import pytest
from ecg.config import ExperimentConfig
from ecg.datasets.manifest import patient_splits
from ecg.labels.mapping import class_ids, load_label_spec, map_codes, unmapped_codes
from ecg.preprocess.augment import augment
from ecg.preprocess.signal import CANONICAL_LEADS, preprocess


def test_every_mapping_row_resolves():
    for name in ("labels.json", "labels_v2.json"):
        spec = load_label_spec(__import__("pathlib").Path("src/ecg/labels") / name)
        assert spec["mappings"] and len(class_ids(spec)) >= 5


def test_unmapped_code_is_never_a_silent_negative():
    spec = load_label_spec()
    y, m = map_codes("ptbxl", ["NOT_A_CODE"], spec)
    assert m.sum() == 0 and y.sum() == 0
    assert unmapped_codes("ptbxl", [["NOT_A_CODE", "NORM"]], spec)["NOT_A_CODE"] == 1


def test_mapped_record_marks_covered_classes_known():
    spec = load_label_spec()
    y, m = map_codes("ptbxl", ["IMI"], spec)
    assert y[class_ids(spec).index("MI")] == 1 and m.sum() == len(class_ids(spec))


def test_v2_class_missing_from_source_is_masked():
    spec = load_label_spec(__import__("pathlib").Path("src/ecg/labels/labels_v2.json"))
    y, m = map_codes("cinc", ["426783006"], spec)
    assert m[class_ids(spec).index("NORM")] == 1


def test_split_integrity_no_patient_overlap():
    pids = [f"p{i // 3}" for i in range(300)]
    splits = patient_splits(pids, seed=1)
    seen = {}
    for p, s in zip(pids, splits):
        assert seen.setdefault(p, s) == s
    assert set(splits) == {"train", "validation", "test"}


def test_splits_are_deterministic():
    pids = [f"p{i // 2}" for i in range(200)]
    assert patient_splits(pids, 7) == patient_splits(pids, 7)


def _record(fs=500, seconds=10, leads=CANONICAL_LEADS):
    rng = np.random.default_rng(0)
    return rng.normal(size=(fs * seconds, len(leads))), fs, list(leads)


def test_preprocess_shape_and_determinism():
    x, fs, leads = _record()
    a, ma = preprocess(x, fs, leads)
    b, mb = preprocess(x, fs, leads)
    assert a.shape == (12, 1000) and a.dtype == np.float32 and ma.sum() == 12
    assert np.array_equal(a, b) and np.array_equal(ma, mb)


def test_missing_leads_zero_filled_with_mask():
    x, fs, leads = _record(leads=CANONICAL_LEADS[:6])
    out, mask = preprocess(x, fs, leads)
    assert mask.tolist() == [1] * 6 + [0] * 6 and not out[6:].any()


def test_ptbxl_uppercase_lead_names_are_matched():
    names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    x, fs, _ = _record()
    out, mask = preprocess(x, fs, names)
    assert mask.sum() == 12 and out[3].any() and out[4].any() and out[5].any()


def test_short_record_is_zero_padded():
    x, fs, leads = _record(seconds=5)
    out, _ = preprocess(x, fs, leads)
    assert out.shape[1] == 1000 and not out[:, -10:].any()


def test_augmentation_is_seeded_and_keeps_a_lead():
    x = np.ones((12, 1000), np.float32)
    m = np.ones(12, np.float32)
    kw = dict(noise_std=0.1, wander_amp=0.2, wander_p=1, powerline_amp=0.1, powerline_p=1, lead_dropout_p=1.0,
              time_mask_len=50, time_mask_p=1, jitter=0.02, jitter_p=1, crop_p=1, crop_min_fraction=0.9)
    a, ma = augment(x, m, np.random.default_rng(3), **kw)
    b, mb = augment(x, m, np.random.default_rng(3), **kw)
    assert np.array_equal(a, b) and np.array_equal(ma, mb) and ma.sum() >= 1
    assert not a[ma == 0].any()


def test_config_roundtrip_and_unknown_keys():
    cfg = ExperimentConfig.from_yaml("configs/baseline.yaml")
    assert cfg.run_name(1) == "S1/resnet1d/resnet100hz/1" and len(cfg.config_hash()) == 16
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"train": {"bogus": 1}})
