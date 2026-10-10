"""
preprocess.py
Inference-time preprocessing for a single candled egg image.

This is a faithful extraction of the `neutral_fill` pipeline from notebook 02, which is what
produced the cache every model in this project was trained on. It lives here, rather than being
re-typed inside the Streamlit app, because a mismatch between training and inference preprocessing
is silent: the app would still return a confident prediction, just a wrong one.

The steps and the reason for each, all evidenced in notebooks 01b and 02:

1. **Segment the egg** with Otsu at 512 px and take the largest connected component.
2. **Replace everything outside the egg** with the egg's own median colour plus matched noise.
   Black letterbox padding was the single strongest leakage channel in the raw data -- the fraction
   of near-black pixels was 0.032 for intact eggs and 0.198 for Star -- and neutral fill removes it
   without leaving a suspiciously flat region.
3. **Crop to a square centred on the egg**, side = longer egg dimension x (1 + margin), padding by
   edge replication where it runs off the frame, never with black.
4. **Resize to 448** isotropically, matching the cache.
5. **Resize to the training resolution and standardise per image per channel.** Standardisation is
   what removes the absolute brightness, contrast and colour-balance differences that notebook 01b
   showed carry class information.

If `crop_eggs.py` has already been applied, this still works: the egg is simply found again in the
cropped frame and the black bars are filled in.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .augmentation import standardise

__all__ = ["MASK_SIDE", "CACHE_SIDE", "CROP_MARGIN", "egg_mask_and_bbox", "neutral_fill",
           "preprocess_bgr", "to_model_input"]

MASK_SIDE = 512
CACHE_SIDE = 448
CROP_MARGIN = 0.04


def egg_mask_and_bbox(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Otsu segmentation of the candled egg: full-resolution boolean mask plus bounding box."""
    H, W = gray.shape
    small = cv2.GaussianBlur(
        cv2.resize(gray, (MASK_SIDE, MASK_SIDE), interpolation=cv2.INTER_AREA), (5, 5), 0)
    _, m = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones((H, W), bool), (0, 0, W, H)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    msmall = (lab == k).astype(np.uint8)
    x, y, bw, bh = st[k, :4]
    sx, sy = W / MASK_SIDE, H / MASK_SIDE
    mask = cv2.resize(msmall, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, (int(x * sx), int(y * sy), int(bw * sx), int(bh * sy))


def neutral_fill(img: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Replace everything outside the egg with the egg's own colour statistics."""
    out = img.copy()
    egg = img[mask]
    if egg.size == 0:
        return out
    med = np.median(egg, axis=0)
    sd = np.clip(egg.std(axis=0), 1.0, None)
    bg = ~mask
    n_bg = int(bg.sum())
    if n_bg:
        rng = np.random.default_rng(seed)
        out[bg] = np.clip(med + rng.normal(0.0, sd * 0.5, size=(n_bg, 3)), 0, 255).astype(np.uint8)
    return out


def preprocess_bgr(img_bgr: np.ndarray, side: int = CACHE_SIDE,
                   margin: float = CROP_MARGIN, seed: int = 0) -> Optional[np.ndarray]:
    """BGR image of any size -> RGB uint8 (side, side), identical to a row of the training cache."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    H, W = img_bgr.shape[:2]
    mask, (x, y, bw, bh) = egg_mask_and_bbox(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    img = neutral_fill(img_bgr, mask, seed)

    cx, cy = x + bw / 2.0, y + bh / 2.0
    half = max(bw, bh) * (1.0 + margin) / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    crop = img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]
    if crop.size == 0:
        return None
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    out = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def to_model_input(rgb_448: np.ndarray, train_side: int = 384) -> np.ndarray:
    """Cache-format RGB -> standardised CHW float32, ready for torch.from_numpy."""
    im = rgb_448
    if im.shape[0] != train_side:
        im = cv2.resize(im, (train_side, train_side), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(standardise(im).transpose(2, 0, 1))[None]
