"""Shift recovery (S7): BatchNorm adaptation and small-sample fine-tuning on a target sample."""
import copy
import numpy as np
import torch

from ecg.train.losses import masked_bce_with_logits
from ecg.train.trainer import loader_for


def adapt_batchnorm(model, dataset, device, batch_size: int = 64):
    """Re-estimate BatchNorm statistics on unlabeled target data; weights are untouched."""
    adapted = copy.deepcopy(model).to(device)
    for m in adapted.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None  # cumulative average
    adapted.train()
    with torch.no_grad():
        for x, lm, _, _ in loader_for(dataset, batch_size, False):
            adapted(x.to(device), lm.to(device))
    return adapted.eval()


def finetune_small_sample(model, dataset, device, n_samples: int = 200, epochs: int = 5, lr: float = 1e-4,
                          seed: int = 0, batch_size: int = 32):
    """Fine-tune a copy on a small labeled target sample; returns (model, indices used)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    sub = torch.utils.data.Subset(dataset, idx.tolist())
    tuned = copy.deepcopy(model).to(device)
    opt = torch.optim.AdamW(tuned.parameters(), lr=lr)
    for _ in range(epochs):
        tuned.train()
        for x, lm, y, m in loader_for(sub, batch_size, True, seed=seed):
            x, lm, y, m = (t.to(device) for t in (x, lm, y, m))
            loss = masked_bce_with_logits(tuned(x, lm)[0], y, m)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return tuned.eval(), idx
