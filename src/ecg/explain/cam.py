"""Class activation maps for global-pooling CNN heads."""
import numpy as np


def cam(features, head_weights, class_index, input_length):
    """Compute normalized [lead, input_time] CAM from convolutional feature maps."""
    f = np.asarray(features)  # [D, T']
    weights = np.asarray(head_weights)[class_index]
    temporal = np.maximum((weights[:, None] * f).sum(axis=0), 0)
    temporal = np.interp(np.linspace(0, len(temporal) - 1, input_length), np.arange(len(temporal)), temporal)
    peak = temporal.max()
    temporal = temporal / peak if peak > 0 else temporal
    return np.broadcast_to(temporal[None, :], (12, input_length)).copy().astype(np.float32)
