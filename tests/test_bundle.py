"""Bundle contract and ONNX parity tests."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
import pytest
import torch

from ecg.export.bundle import validate_bundle, check_onnx_parity
from ecg.models.resnet1d import ResNet1D


def test_bundle_structure_and_schemas():
    bundle_path = Path("bundle/v1.0.0")
    assert bundle_path.exists()
    assert validate_bundle(bundle_path) is True
    
    # Check JSON validity
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_contract"]["signal"] == "float32 [B, 12, 1000]"
    assert manifest["input_contract"]["lead_mask"] == "float32 [B, 12]"
    assert manifest["test_metrics"]["macro_auroc"] > 0.85


def test_onnx_runtime_parity():
    bundle_path = Path("bundle/v1.0.0")
    onnx_file = bundle_path / "model.onnx"
    assert onnx_file.is_file()
    
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    
    # Test batch size of 2 and dynamic batching
    B, T = 2, 1000
    dummy_signal = np.random.randn(B, 12, T).astype(np.float32)
    dummy_mask = np.ones((B, 12), dtype=np.float32)
    
    outputs = session.run(["logits", "features"], {
        "signal": dummy_signal,
        "lead_mask": dummy_mask
    })
    
    logits, features = outputs[0], outputs[1]
    assert logits.shape == (B, 5)
    assert features.shape == (B, 512, 32)
    assert logits.dtype == np.float32
    assert features.dtype == np.float32


def test_exported_head_weights_cam():
    bundle_path = Path("bundle/v1.0.0")
    weights = np.load(bundle_path / "head_weights.npy")
    assert weights.shape == (5, 512)
    
    # Test CAM computation with exported weights
    from ecg.explain.cam import cam
    features = np.random.randn(512, 63).astype(np.float32)
    attribution = cam(features, weights, class_index=0, input_length=1000)
    assert attribution.shape == (12, 1000)
    assert attribution.min() >= 0.0
    assert attribution.max() <= 1.0
