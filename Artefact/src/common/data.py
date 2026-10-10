"""
data.py
Dataset and DataLoader construction for the eggshell micro-crack artefact.

Images come from the memory-mapped cache written by notebook 02 (652 x 448 x 448 x 3 uint8,
392 MB), not from JPEG files. Decoding a 1700-4400 px JPEG costs 50-150 ms; reading a row of the
cache costs under a millisecond, which on an RTX 3050 is the difference between the GPU waiting on
the CPU and the GPU being the bottleneck.

Order of operations per sample, and the reason for it:

    cache row (448 px, uint8)
      -> resize to the training resolution
      -> augment            (training split only; identical transform applied to the crack mask)
      -> per-image per-channel standardisation
      -> CHW float32 tensor

Standardisation comes *after* augmentation deliberately. Notebook 01b showed that absolute
brightness, contrast and colour balance carry class information that is acquisition rather than
signal; standardising last means the augmented photometric variation is also normalised away and
the network is left with shape and texture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .augmentation import EggAugment, standardise

__all__ = ["EggDataset", "build_crack_masks", "make_loader", "class_weights"]


def build_crack_masks(crack_regions: Sequence, side: int) -> np.ndarray:
    """Rasterise every annotation polygon once, as an (N, side, side) uint8 array.

    Precomputed rather than parsed per sample: 652 x 384 x 384 uint8 is 96 MB, which is cheap
    compared with re-parsing JSON in every DataLoader worker on every epoch.
    """
    out = np.zeros((len(crack_regions), side, side), np.uint8)
    for i, cr in enumerate(crack_regions):
        if isinstance(cr, str) and cr.strip():
            for o in json.loads(cr):
                p = o.get("points", [])
                if len(p) >= 3:
                    pts = np.round(np.asarray(p, float) / 100.0 * side).astype(np.int32)
                    cv2.fillPoly(out[i], [pts], 1)
    return out


class EggDataset(Dataset):
    """One candled egg per item.

    Returns a dict so that the four problem formulations in notebook 05 can each take what they
    need without separate dataset classes:

        image    float32 (3, S, S)
        y4       int64   four-class label index
        y2       int64   binary crack label
        ypat     int64   pattern index, or -100 for intact eggs so cross-entropy ignores them
        age      int64   breeder age-class index
        mask     float32 (1, S, S)   crack mask, only when masks are supplied
        idx      int64   row index, for error analysis
    """

    def __init__(self, cache_path, rows: Sequence[int], labels, train_side: int,
                 policy: Optional[str] = None, masks: Optional[np.ndarray] = None,
                 seed: int = 0, occlude_crack: bool = False,
                 occlude_box_frac: float = 0.45, occlude_invert: bool = False):
        self.cache_path = str(cache_path)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.train_side = int(train_side)
        self.policy = policy
        self.masks = masks
        self.seed = int(seed)
        # Control experiment for notebook 05b. The occlusion is a FIXED-SIZE square, identical
        # for every egg: centred on the crack where there is one, at a random position on the egg
        # where there is not. The first version occluded only the eggs that had a mask, which made
        # "contains a smoothed patch" a perfect label for "cracked" -- binary macro F1 came back
        # at 0.998. Fixing the size and giving intact eggs the same treatment removes both the
        # presence cue and the shape cue, so the only thing the control takes away is the crack.
        self.occlude_crack = bool(occlude_crack)
        self.occlude_box_frac = float(occlude_box_frac)
        # occlude_invert flips the control: keep ONLY the square and blank everything else.
        # The pair answers two different questions. Occluding the crack asks whether it is
        # NECESSARY; keeping only the crack asks whether it is SUFFICIENT. A model can fail the
        # first and pass the second, which is what redundant evidence looks like.
        self.occlude_invert = bool(occlude_invert)

        self.y4 = np.asarray(labels["y4"], dtype=np.int64)
        self.y2 = np.asarray(labels["y2"], dtype=np.int64)
        self.ypat = np.asarray(labels["ypat"], dtype=np.int64)
        self.age = np.asarray(labels["age"], dtype=np.int64)
        self._cache = None          # opened lazily, once per worker process

    def __len__(self):
        return len(self.rows)

    def _get_cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __getitem__(self, i):
        r = int(self.rows[i])
        img = np.asarray(self._get_cache()[r])
        S = self.train_side
        if img.shape[0] != S:
            img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)

        m = None
        if self.masks is not None:
            m = self.masks[r]
            if m.shape[0] != S:
                m = cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
            m = m.copy()

        if self.occlude_crack and m is not None:
            side_px = max(8, int(self.occlude_box_frac * S))
            half = side_px // 2
            rng = np.random.default_rng(10_000 + r)
            if m.any():
                ys, xs = np.nonzero(m)
                cy, cx = int(ys.mean()), int(xs.mean())
            else:
                # intact egg: place the same square at a random position near the centre, where
                # the egg is, so the occlusion looks the same as it does on a cracked egg
                cy = int(rng.normal(S / 2, S * 0.12))
                cx = int(rng.normal(S / 2, S * 0.12))
            y0 = int(np.clip(cy - half, 0, S - side_px))
            x0 = int(np.clip(cx - half, 0, S - side_px))
            y1, x1 = y0 + side_px, x0 + side_px

            box = np.zeros((S, S), np.uint8)
            box[y0:y1, x0:x1] = 1
            k = max(3, int(0.06 * S) | 1)
            ring = (cv2.dilate(box, np.ones((k, k), np.uint8)) - box).astype(bool)
            img = img.copy()
            if ring.sum() > 10:
                fill = img[ring].mean(axis=0)
                sd = max(float(img[ring].std()), 1.0)
                target = (~box.astype(bool)) if self.occlude_invert else box.astype(bool)
                n_px = int(target.sum())
                noise = rng.normal(0, sd, size=(n_px, 3))
                img[target] = np.clip(fill + noise, 0, 255).astype(np.uint8)
            # When the crack has been painted over its mask must go too, or the segmentation
            # head would be supervised towards a region that no longer contains anything. Under
            # inversion the crack is the part that survives, so the mask is kept.
            if not self.occlude_invert:
                m[:] = 0

        if self.policy and self.policy != "none":
            aug = EggAugment(self.policy, seed=None)   # fresh randomness every epoch
            img, m = aug(img, m)

        x = standardise(img).transpose(2, 0, 1)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "y4": torch.tensor(self.y4[r]),
            "y2": torch.tensor(self.y2[r]),
            "ypat": torch.tensor(self.ypat[r]),
            "age": torch.tensor(self.age[r]),
            "idx": torch.tensor(r),
        }
        if m is not None:
            item["mask"] = torch.from_numpy(
                np.ascontiguousarray(m.astype(np.float32))).unsqueeze(0)
        return item


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale stays comparable."""
    counts = np.bincount(y[y >= 0], minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_loader(ds: EggDataset, batch_size: int, train: bool, num_workers: int = 8,
                balanced: bool = False, y_for_balance: Optional[np.ndarray] = None,
                seed: int = 42) -> DataLoader:
    """Build a DataLoader. `balanced=True` oversamples minority classes.

    Star is the minority class at 68 eggs against 317 intact, a ratio of 4.66, so the sampling
    strategy is one of the things notebook 05 compares rather than assumes.
    """
    sampler = None
    shuffle = train
    if train and balanced and y_for_balance is not None:
        yb = y_for_balance[ds.rows]
        counts = np.bincount(yb, minlength=int(yb.max()) + 1).astype(np.float64)
        counts[counts == 0] = 1.0
        w = (1.0 / counts)[yb]
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(ds), replacement=True, generator=g)
        shuffle = False

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
