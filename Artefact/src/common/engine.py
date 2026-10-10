"""
engine.py
Training and evaluation loop, sized for a 6 GB RTX 3050.

Four memory measures, all standard and all reported in Section 5.2.2:

* automatic mixed precision, which roughly halves activation memory and uses the Ampere tensor cores
* gradient accumulation, so the *effective* batch size is identical across backbones of very
  different sizes and the optimisation budget stays comparable
* channels_last memory format, which suits convolutional backbones on Ampere
* per-fold checkpointing and incremental result logging, so a crash in hour six of an overnight
  run resumes instead of restarting

Everything a run did is appended to results/runs.csv as it happens. Reconstructing experimental
setup from memory afterwards is not possible, so it is recorded at the time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score

from .models import build_model, count_parameters

__all__ = ["TrainConfig", "train_fold", "predict", "amp_autocast", "make_scaler", "append_run"]


# --------------------------------------------------------------------------- AMP compatibility
def amp_autocast(enabled: bool = True):
    """torch.amp is the current API; torch.cuda.amp is the older one. Support both."""
    if not (enabled and torch.cuda.is_available()):
        return torch.autocast(device_type="cpu", enabled=False)
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def make_scaler(enabled: bool = True):
    if not (enabled and torch.cuda.is_available()):
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


# --------------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b0"
    formulation: str = "single4"          # single | single4 | multitask | multitask_seg
    n_out: int = 4
    policy: str = "medium"
    train_side: int = 384
    epochs: int = 30
    physical_batch: int = 16
    effective_batch: int = 32
    lr: float = 3e-4
    head_lr_mult: float = 10.0
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    imbalance: str = "class_weights"      # class_weights | focal | balanced_sampler | none
    focal_gamma: float = 2.0
    seg_loss_weight: float = 0.5
    pattern_loss_weight: float = 1.0
    age_mode: str = "none"
    amp: bool = True
    channels_last: bool = True
    patience: int = 8
    num_workers: int = 8
    seed: int = 42
    occlude_crack: bool = False           # the control experiment
    tag: str = ""

    def asdict(self):
        return asdict(self)


# --------------------------------------------------------------------------- losses
def focal_loss(logits, target, gamma=2.0, weight=None, ignore_index=-100):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none",
                         ignore_index=ignore_index)
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    pt = torch.exp(-ce[valid])
    return ((1 - pt) ** gamma * ce[valid]).mean()


def dice_bce(pred_logits, target):
    """Segmentation loss. Dice carries the heavy foreground/background imbalance; BCE stabilises it.

    Annotated crack regions cover roughly 2-8 % of the frame, so plain BCE alone collapses to
    predicting background everywhere.
    """
    t = F.interpolate(target, size=pred_logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(pred_logits, t)
    p = torch.sigmoid(pred_logits)
    num = 2 * (p * t).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


# --------------------------------------------------------------------------- train / eval
def _build_optimizer(model, cfg: TrainConfig):
    head_names = ("head", "head_det", "head_pat", "seg_head")
    head_p, back_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_p if n.split(".")[0] in head_names else back_p).append(p)
    return torch.optim.AdamW(
        [{"params": back_p, "lr": cfg.lr},
         {"params": head_p, "lr": cfg.lr * cfg.head_lr_mult}],
        weight_decay=cfg.weight_decay)


@torch.no_grad()
def predict(model, loader, device, cfg: TrainConfig, n_age: int = 3):
    """Return logits, probabilities and labels for a loader."""
    model.eval()
    L4, L2, LP, Y4, Y2, YP, IDX = [], [], [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.to(memory_format=torch.channels_last)
        age = F.one_hot(batch["age"].to(device), n_age).float() if cfg.age_mode == "concat" else None
        with amp_autocast(cfg.amp):
            out = model(x, age)
        if "logits" in out:
            L4.append(out["logits"].float().cpu())
        else:
            L2.append(out["logits_det"].float().cpu())
            LP.append(out["logits_pat"].float().cpu())
        Y4.append(batch["y4"]); Y2.append(batch["y2"]); YP.append(batch["ypat"])
        IDX.append(batch["idx"])
    r = {"y4": torch.cat(Y4).numpy(), "y2": torch.cat(Y2).numpy(),
         "ypat": torch.cat(YP).numpy(), "idx": torch.cat(IDX).numpy()}
    if L4:
        r["logits"] = torch.cat(L4).numpy()
    else:
        r["logits_det"] = torch.cat(L2).numpy()
        r["logits_pat"] = torch.cat(LP).numpy()
    return r


def _metrics(pred: dict, cfg: TrainConfig) -> dict:
    """Macro F1 and balanced accuracy, mapped onto the four-class target where possible.

    Multi-task heads are combined into a four-class decision so that every formulation is scored
    on the same quantity and the comparison in notebook 05 is meaningful.
    """
    if "logits" in pred:
        yh = pred["logits"].argmax(1)
        if cfg.n_out == 4:
            y = pred["y4"]
        elif cfg.n_out == 2:
            y = pred["y2"]
        else:
            m = pred["ypat"] >= 0
            y, yh = pred["ypat"][m], yh[m]
    else:
        det = pred["logits_det"].argmax(1)
        pat = pred["logits_pat"].argmax(1)
        # LABEL4 order is Intact, Hairline, Star, Network, so pattern index p maps to p + 1
        yh = np.where(det == 0, 0, pat + 1)
        y = pred["y4"]
    out = {"macro_f1": float(f1_score(y, yh, average="macro", zero_division=0)),
           "balanced_acc": float(balanced_accuracy_score(y, yh)),
           "accuracy": float((y == yh).mean())}
    if "logits" in pred and cfg.n_out == 4:
        out["macro_f1_binary"] = float(
            f1_score(pred["y2"], (yh > 0).astype(int), average="macro", zero_division=0))
    elif "logits_det" in pred:
        out["macro_f1_binary"] = float(
            f1_score(pred["y2"], pred["logits_det"].argmax(1), average="macro", zero_division=0))
    return out


def train_fold(cfg: TrainConfig, train_loader, val_loader, device,
               w4=None, w2=None, wpat=None, ckpt_path: Optional[Path] = None,
               verbose: bool = True, n_age: int = 3):
    """Train one fold. Returns the best validation metrics and the training history."""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()

    model = build_model(cfg.backbone, cfg.formulation, cfg.n_out,
                        pretrained=True, age_mode=cfg.age_mode,
                        img_size=cfg.train_side).to(device)
    if cfg.channels_last and not model.needs_seg:
        model = model.to(memory_format=torch.channels_last)

    opt = _build_optimizer(model, cfg)
    scaler = make_scaler(cfg.amp)
    accum = max(1, cfg.effective_batch // cfg.physical_batch)
    steps = max(1, len(train_loader) // accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=cfg.epochs * steps,
        pct_start=min(0.3, max(0.05, cfg.warmup_epochs / max(cfg.epochs, 1))))

    w4 = None if w4 is None else w4.to(device)
    w2 = None if w2 is None else w2.to(device)
    wpat = None if wpat is None else wpat.to(device)
    use_focal = cfg.imbalance == "focal"

    best = {"macro_f1": -1.0}
    best_state, bad, history = None, 0, []
    t_start = time.time()

    for ep in range(cfg.epochs):
        model.train()
        t_ep = time.time()
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            if cfg.channels_last:
                x = x.to(memory_format=torch.channels_last)
            y4 = batch["y4"].to(device); y2 = batch["y2"].to(device)
            ypat = batch["ypat"].to(device)
            age = F.one_hot(batch["age"].to(device), n_age).float() \
                if cfg.age_mode == "concat" else None

            with amp_autocast(cfg.amp):
                out = model(x, age)
                if "logits" in out:
                    tgt = {4: y4, 2: y2, 3: ypat}[cfg.n_out]
                    w = {4: w4, 2: w2, 3: wpat}[cfg.n_out]
                    loss = (focal_loss(out["logits"], tgt, cfg.focal_gamma, w) if use_focal
                            else F.cross_entropy(out["logits"], tgt, weight=w,
                                                 label_smoothing=cfg.label_smoothing,
                                                 ignore_index=-100))
                else:
                    l_det = (focal_loss(out["logits_det"], y2, cfg.focal_gamma, w2) if use_focal
                             else F.cross_entropy(out["logits_det"], y2, weight=w2,
                                                  label_smoothing=cfg.label_smoothing))
                    l_pat = (focal_loss(out["logits_pat"], ypat, cfg.focal_gamma, wpat) if use_focal
                             else F.cross_entropy(out["logits_pat"], ypat, weight=wpat,
                                                  label_smoothing=cfg.label_smoothing,
                                                  ignore_index=-100))
                    loss = l_det + cfg.pattern_loss_weight * l_pat
                if model.needs_seg and "mask" in batch:
                    loss = loss + cfg.seg_loss_weight * dice_bce(
                        out["seg"], batch["mask"].to(device))
                loss = loss / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            run_loss += float(loss.item()) * accum; nb += 1

        m = _metrics(predict(model, val_loader, device, cfg, n_age), cfg)
        m["epoch"] = ep; m["train_loss"] = run_loss / max(nb, 1)
        m["epoch_seconds"] = time.time() - t_ep
        history.append(m)
        if verbose:
            print(f"    ep {ep:>2}  loss {m['train_loss']:.4f}  "
                  f"val macroF1 {m['macro_f1']:.4f}  bal_acc {m['balanced_acc']:.4f}")

        if m["macro_f1"] > best["macro_f1"]:
            best = dict(m)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"    early stop at epoch {ep} (no gain for {cfg.patience} epochs)")
                break

    best["seconds"] = time.time() - t_start
    best["params_m"] = count_parameters(model) / 1e6
    best["peak_vram_gb"] = (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0.0)
    best["epochs_run"] = len(history)
    if ckpt_path is not None and best_state is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": cfg.asdict(), "metrics": best},
                   ckpt_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best, history, best_state


def append_run(csv_path: Path, cfg: TrainConfig, fold: int, metrics: dict, extra: dict = None):
    """Append one completed run, aligned to the existing header.

    Earlier versions appended with `header=False`, which makes pandas write values positionally.
    Different run types carry slightly different metric keys -- a composed hierarchical result has
    no `epoch`, `train_loss` or `epoch_seconds` -- so rows written later were shifted against a
    header written earlier, and n_train landed in the params_m column. Reading the existing header
    and reindexing to it removes the whole failure mode; genuinely new columns trigger a rewrite
    of the file so nothing is silently dropped.
    """
    import pandas as pd
    row = {**cfg.asdict(), "fold": fold, **metrics, **(extra or {})}
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
        return row

    existing = pd.read_csv(csv_path, nrows=0)
    cols = list(existing.columns)
    new_cols = [c for c in row if c not in cols]
    if new_cols:
        full = pd.read_csv(csv_path)
        for c in new_cols:
            full[c] = np.nan
        full = pd.concat([full, pd.DataFrame([row]).reindex(columns=list(full.columns))],
                         ignore_index=True)
        full.to_csv(csv_path, index=False)
    else:
        pd.DataFrame([row]).reindex(columns=cols).to_csv(
            csv_path, mode="a", header=False, index=False)
    return row
