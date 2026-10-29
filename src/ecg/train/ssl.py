"""Masked-reconstruction self-supervised pretraining (labels unused), then fine-tune via cfg.model.pretrained."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ecg.datasets.build import ECGArrayDataset
from ecg.models.factory import build_model


class MaskedReconstruction(nn.Module):
    def __init__(self, model, feature_dim: int):
        super().__init__()
        self.model = model
        self.decoder = nn.Sequential(nn.Conv1d(feature_dim, 128, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv1d(128, 12, 1))

    def forward(self, signal, lead_mask, time_mask):
        corrupted = signal * (1 - time_mask).unsqueeze(1)
        z = self.model.encode(corrupted, lead_mask)
        recon = F.interpolate(self.decoder(z), size=signal.shape[-1], mode="linear", align_corners=False)
        return recon


def random_time_mask(batch: int, length: int, ratio: float, segment: int, gen) -> torch.Tensor:
    mask = torch.zeros(batch, length)
    n_seg = max(1, int(ratio * length / segment))
    for b in range(batch):
        starts = torch.randint(0, max(length - segment, 1), (n_seg,), generator=gen)
        for s in starts:
            mask[b, s:s + segment] = 1
    return mask


def pretrain(cfg, cache_path, num_classes: int, out_path, device=None, seed: int | None = None):
    from ecg.train.trainer import loader_for, pick_device, seed_everything
    seed = cfg.train.seed if seed is None else seed
    seed_everything(seed)
    device = device or pick_device()
    ds = ECGArrayDataset(cache_path, "train", None, seed, cfg.data.leads)
    model = build_model(cfg.model, num_classes).to(device)
    with torch.no_grad():
        x0, l0, _, _ = ds[0]
        dim = model.encode(torch.tensor(x0)[None].to(device), torch.tensor(l0)[None].to(device)).shape[1]
    wrapper = MaskedReconstruction(model, dim).to(device)
    opt = torch.optim.AdamW(wrapper.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    gen = torch.Generator(); gen.manual_seed(seed)
    loader = loader_for(ds, cfg.train.batch_size, True, cfg.train.num_workers, seed)
    history = []
    for epoch in range(cfg.train.ssl_epochs):
        wrapper.train(); total = 0.0
        for x, lm, _, _ in loader:
            x, lm = x.to(device), lm.to(device)
            tm = random_time_mask(x.shape[0], x.shape[-1], cfg.train.ssl_mask_ratio, 25, gen).to(device)
            recon = wrapper(x, lm, tm)
            region = tm.unsqueeze(1) * lm.unsqueeze(-1)
            loss = (((recon - x) ** 2) * region).sum() / region.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss) * len(x)
        history.append(total / max(len(ds), 1))
    torch.save({"model": {k: v for k, v in model.state_dict().items() if not k.startswith("head.")},
                "ssl_loss": history}, out_path)
    return history


def load_pretrained_encoder(model, path):
    state = torch.load(path, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in pretrained encoder: {unexpected}")
    return missing
