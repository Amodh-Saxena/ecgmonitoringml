"""Model factory shared by training, evaluation and export."""
from ecg.models.cnn_transformer import CNNTransformer
from ecg.models.inception1d import InceptionTime1D
from ecg.models.resnet1d import ResNet1D


def build_model(cfg, num_classes: int):
    """cfg is a ModelConfig. All models: (signal [B,12,T], lead_mask [B,12]) -> (logits [B,K], features [B,D,T'])."""
    name = cfg.name.lower()
    if name in ("resnet1d", "xresnet1d"):
        return ResNet1D(num_classes, tuple(cfg.channels), cfg.blocks_per_stage, cfg.dropout)
    if name in ("inception1d", "inceptiontime"):
        return InceptionTime1D(num_classes, dropout=cfg.dropout)
    if name == "cnn_transformer":
        return CNNTransformer(num_classes, cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.dropout or 0.1)
    raise ValueError(f"Unknown model: {cfg.name}")
