"""Typed experiment configuration loaded from YAML. One config fully determines a run."""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DataConfig:
    source: str = "ptbxl"
    root: str = "data/raw"
    sampling_rate: int = 100
    duration_seconds: int = 10
    split_seed: int = 42
    labels: str = ""  # path to labels json; empty -> packaged v1 table
    cache_dir: str = "data/cache"
    low_hz: float = 0.5
    high_hz: float = 40.0
    notch_hz: float | None = None
    normalization: str = "record_zscore"
    leads: list[str] | None = None  # evaluate/train on a lead subset (S5)
    label_fraction: float = 1.0  # S6 label-efficiency study
    external_sources: list[str] = field(default_factory=list)  # never used for tuning


@dataclass
class ModelConfig:
    name: str = "resnet1d"
    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    blocks_per_stage: int = 2
    dropout: float = 0.0
    d_model: int = 128  # cnn_transformer
    num_heads: int = 4
    num_layers: int = 2
    pretrained: str = ""  # path to SSL encoder checkpoint (S6)


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 8
    amp: bool = True
    warmup_epochs: int = 2
    schedule: str = "cosine"  # cosine | onecycle
    loss: str = "bce"  # bce | focal
    focal_gamma: float = 2.0
    class_weights: bool = False
    mixup_alpha: float = 0.0
    num_workers: int = 0
    ssl_epochs: int = 20
    ssl_mask_ratio: float = 0.3


@dataclass
class EvalConfig:
    n_boot: int = 1000
    target_sensitivity: float = 0.9
    ece_target: float = 0.05
    ece_bins: int = 10
    faithfulness_windows: int = 3
    window_seconds: float = 1.0


@dataclass
class ExperimentConfig:
    study: str = "S1"
    variant: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    augmentation: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _build(klass, raw: dict):
        known = {f.name for f in fields(klass)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown {klass.__name__} keys: {sorted(unknown)}")
        return klass(**raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        return cls(data=cls._build(DataConfig, raw.get("data", {})),
                   model=cls._build(ModelConfig, raw.get("model", {})),
                   train=cls._build(TrainConfig, raw.get("train", {})),
                   eval=cls._build(EvalConfig, raw.get("eval", {})),
                   study=raw.get("study", "S1"), variant=raw.get("variant", "baseline"),
                   augmentation=raw.get("augmentation", {}))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def run_name(self, seed: int | None = None) -> str:
        """Runs are named study/model/variant/seed."""
        return f"{self.study}/{self.model.name}/{self.variant}/{self.train.seed if seed is None else seed}"
