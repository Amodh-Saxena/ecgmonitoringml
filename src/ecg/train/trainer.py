"""Trainer: masked loss, AdamW, warmup+cosine/one-cycle, AMP, early stopping, resumable checkpoints."""
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg.datasets.build import ECGArrayDataset
from ecg.eval.harness import predict_logits
from ecg.eval.metrics import evaluate_arrays
from ecg.models.factory import build_model
from ecg.train.losses import class_pos_weights, masked_bce_with_logits, masked_focal_with_logits
from ecg.train.tracking import RunTracker


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_scheduler(opt, cfg, steps_per_epoch: int):
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    if cfg.schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, cfg.learning_rate, total_steps=max(total, 1))
    def f(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def loader_for(dataset, batch_size, shuffle, num_workers=0, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g, drop_last=False)


def mixup_batch(x, lm, y, m, alpha, gen):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], generator=gen)
    return lam * x + (1 - lam) * x[perm], torch.minimum(lm, lm[perm]), lam * y + (1 - lam) * y[perm], m * m[perm]


def check_leakage(manifest: dict, cfg) -> str | None:
    """Return a reason string if patient overlap or external data in training is detected."""
    by_split: dict[str, set] = {}
    for r in manifest["records"]:
        by_split.setdefault(r["split"], set()).add(r["patient_id"])
    names = sorted(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if by_split[a] & by_split[b]:
                return f"patient overlap between {a} and {b}"
    if manifest["metadata"]["source"] in cfg.data.external_sources:
        return "external dataset used for training"
    return None


def fit(cfg, cache_path, num_classes: int, runs_root="runs", seed: int | None = None, resume: bool = False,
        use_mlflow: bool = True, device=None, return_model: bool = False):
    seed = cfg.train.seed if seed is None else seed
    seed_everything(seed)
    device = device or pick_device()
    aug = {"fs": cfg.data.sampling_rate, **cfg.augmentation} if cfg.augmentation else None
    train_ds = ECGArrayDataset(cache_path, "train", aug, seed, cfg.data.leads, cfg.data.label_fraction)
    val_ds = ECGArrayDataset(cache_path, "validation", None, seed, cfg.data.leads)
    tracker = RunTracker(cfg, Path(cache_path).name, runs_root, seed, use_mlflow)
    reason = check_leakage(train_ds.manifest, cfg)
    if reason:
        tracker.mark_invalid(reason)
        raise RuntimeError(f"Leakage check failed: {reason}")
    model = build_model(cfg.model, num_classes).to(device)
    if cfg.model.pretrained:
        from ecg.train.ssl import load_pretrained_encoder
        load_pretrained_encoder(model, cfg.model.pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    train_loader = loader_for(train_ds, cfg.train.batch_size, True, cfg.train.num_workers, seed)
    sched = make_scheduler(opt, cfg.train, len(train_loader))
    pos_weight = None
    if cfg.train.class_weights:
        pos_weight = class_pos_weights(np.asarray(train_ds.targets)[train_ds.indices],
                                       np.asarray(train_ds.target_masks)[train_ds.indices]).to(device)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir = tracker.dir
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    start_epoch, best, bad = 0, -1.0, 0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
    gen = torch.Generator(); gen.manual_seed(seed)
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train(); total = 0.0
        for x, lm, y, m in train_loader:
            x, lm, y, m = (t.to(device) for t in (x, lm, y, m))
            if cfg.train.mixup_alpha > 0:
                x, lm, y, m = mixup_batch(x, lm, y, m, cfg.train.mixup_alpha, gen)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits, _ = model(x, lm)
            logits = logits.float()
            if cfg.train.loss == "focal":
                loss = masked_focal_with_logits(logits, y, m, cfg.train.focal_gamma, pos_weight)
            else:
                loss = masked_bce_with_logits(logits, y, m, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += float(loss) * len(x)
        train_loss = total / max(len(train_ds), 1)
        z, Y, M = predict_logits(model, val_ds, device, cfg.train.batch_size)
        auc = evaluate_arrays(Y, 1 / (1 + np.exp(-z)), M)["macro_auroc"]
        auc = -1.0 if auc is None else auc
        tracker.log_metrics({"train_loss": train_loss, "val_macro_auroc": auc}, epoch)
        is_best = auc > best
        print(f"Epoch {epoch+1:2d}/{cfg.train.epochs:2d} | Train Loss: {train_loss:.4f} | Val Macro AUROC: {auc:.4f} {'[BEST]' if is_best else ''}", flush=True)
        if is_best:
            best, bad = auc, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_auroc": auc}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad}, last_path)
        if bad >= cfg.train.patience:
            print(f"Early stopping triggered at epoch {epoch+1} (patience={cfg.train.patience})", flush=True)
            break
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    tracker.log_metrics({"best_val_macro_auroc": best})
    tracker.close()
    result = {"best_checkpoint": str(best_path), "best_val_macro_auroc": best, "run_dir": str(out_dir),
              "manifest_hash": Path(cache_path).name}
    return (result, model) if return_model else result
