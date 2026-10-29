"""Model bundle loader (singleton) that loads all artifacts once at startup."""
import json
from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime as ort

from .config import BUNDLE_DIR


class ModelBundle:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Model bundle directory not found: {self.bundle_dir}")

        # 1. Manifest
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json missing from bundle at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_version = self.manifest.get("bundle_version", "v1.0.0")
        self.manifest_hash = self.manifest.get("manifest_hash", "")
        self.test_metrics = self.manifest.get("test_metrics", {})
        self.license = self.manifest.get("license", "PhysioNet Open Data / MIT Code License")

        # 2. Labels
        labels_path = self.bundle_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json missing from bundle at {labels_path}")
        self.labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
        self.classes = self.labels_data.get("classes", [])
        self.class_codes = [c["id"] for c in self.classes]
        self.class_names = [c["display_name"] for c in self.classes]

        # 3. Preprocess specification
        prep_path = self.bundle_dir / "preprocess.json"
        if not prep_path.exists():
            raise FileNotFoundError(f"preprocess.json missing from bundle at {prep_path}")
        self.prep_config = json.loads(prep_path.read_text(encoding="utf-8"))
        self.target_fs = self.prep_config.get("sampling_rate", 100)
        self.target_duration = self.prep_config.get("duration_seconds", 10)
        self.target_length = self.target_fs * self.target_duration

        # 4. Calibration parameters
        cal_path = self.bundle_dir / "calibration.json"
        if not cal_path.exists():
            raise FileNotFoundError(f"calibration.json missing from bundle at {cal_path}")
        self.cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
        raw_temps = self.cal_data.get("temperature", self.cal_data.get("temperatures", [1.0] * len(self.classes)))
        self.temperatures = np.array(raw_temps, dtype=np.float32)

        # 5. Thresholds
        thr_path = self.bundle_dir / "thresholds.json"
        if not thr_path.exists():
            raise FileNotFoundError(f"thresholds.json missing from bundle at {thr_path}")
        self.thr_data = json.loads(thr_path.read_text(encoding="utf-8"))
        self.thresholds = np.array(self.thr_data.get("thresholds", [0.5] * len(self.classes)), dtype=np.float32)

        # 6. Abstention policy
        abs_path = self.bundle_dir / "abstention.json"
        if not abs_path.exists():
            raise FileNotFoundError(f"abstention.json missing from bundle at {abs_path}")
        self.abs_data = json.loads(abs_path.read_text(encoding="utf-8"))
        self.quality_floor = float(self.abs_data.get("quality_floor", 0.25))
        self.confidence_floor = float(self.abs_data.get("confidence_floor", 0.05))

        # 7. Head weights for CAM
        hw_path = self.bundle_dir / "head_weights.npy"
        if not hw_path.exists():
            raise FileNotFoundError(f"head_weights.npy missing from bundle at {hw_path}")
        self.head_weights = np.load(hw_path).astype(np.float32)  # [5, 512]

        # 8. Model Card
        mc_path = self.bundle_dir / "MODEL_CARD.md"
        self.model_card_text = mc_path.read_text(encoding="utf-8") if mc_path.exists() else ""

        # 9. ONNX Session
        onnx_path = self.bundle_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"model.onnx missing from bundle at {onnx_path}")
        
        # Configure CPU inference options
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]


# Singleton instance
_bundle_instance: Optional[ModelBundle] = None


def get_model_bundle() -> ModelBundle:
    global _bundle_instance
    if _bundle_instance is None:
        _bundle_instance = ModelBundle(BUNDLE_DIR)
    return _bundle_instance
