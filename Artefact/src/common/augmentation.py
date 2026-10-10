"""
augmentation.py
Candling-specific augmentation for eggshell micro-crack images.

Written directly on NumPy and OpenCV rather than on Albumentations, for three reasons:

1. Every transform is applied jointly to the image **and** the crack mask with identical
   parameters, which notebook 05 formulation (d) needs for auxiliary segmentation supervision
   and notebook 09 needs for explainability validation.
2. The augmentation API in third-party libraries changes between major versions; this module
   pins the exact behaviour reported in the dissertation so results stay reproducible.
3. Each transform can be switched off individually, which is what the policy comparison in
   notebook 03 requires.

Design constraints specific to this dataset, all evidenced in notebooks 01-02:

* **Photometric augmentation must be strong.** Capture statistics alone still predict class at
  macro F1 0.346 after preprocessing. Brightness, contrast, gamma and channel-balance jitter are
  the mechanism by which the residual acquisition shortcut is made unusable to the network.
* **Geometric augmentation must not distort elongation.** Max elongation is the single most
  discriminative crack feature (epsilon squared = 0.51 in notebook 01). Shear and non-uniform
  scaling are therefore excluded; rotation and isotropic scaling preserve aspect ratios.
* **Crops must not remove crack evidence.** 101 of 652 eggs are partial, concentrated in the
  cracked classes. Scale-down and translation are bounded, and notebook 03 measures the retained
  crack area for every policy rather than assuming it is safe.
* **Blur must stay mild.** A hairline crack is a thin, low-contrast feature; aggressive blur
  erases the signal the model is supposed to learn.
* **Borders are reflected, never filled with black.** Black padding was the strongest single leak
  channel in the raw data; re-introducing it through augmentation would undo the preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["AugmentPolicy", "POLICIES", "EggAugment", "standardise", "to_tensor_hwc"]


# --------------------------------------------------------------------------- policies
@dataclass
class AugmentPolicy:
    """Probabilities and magnitudes for every transform. All ranges are inclusive."""
    name: str = "medium"

    # --- geometric (orientation is arbitrary under candling, so these are free)
    p_hflip: float = 0.5
    p_vflip: float = 0.5
    p_rot90: float = 0.5
    p_affine: float = 0.7
    rotate_deg: float = 15.0          # isotropic, preserves elongation
    scale_range: Tuple[float, float] = (0.92, 1.08)
    translate_frac: float = 0.05

    # --- photometric (the mechanism that neutralises the residual capture shortcut)
    p_brightness_contrast: float = 0.8
    brightness_limit: float = 0.25
    contrast_limit: float = 0.25
    p_gamma: float = 0.5
    gamma_range: Tuple[float, float] = (0.75, 1.35)
    p_channel_jitter: float = 0.6     # breaks the red/blue ratio that leaked class
    channel_gain: float = 0.12
    p_saturation: float = 0.4
    saturation_range: Tuple[float, float] = (0.80, 1.20)
    p_hue: float = 0.25
    hue_shift_deg: float = 6.0        # small: candling colour carries real information

    # --- sensor effects
    p_noise: float = 0.35
    noise_sigma: Tuple[float, float] = (2.0, 9.0)
    p_blur: float = 0.20
    blur_sigma: Tuple[float, float] = (0.3, 1.0)
    p_sharpen: float = 0.15
    sharpen_amount: Tuple[float, float] = (0.2, 0.6)

    # --- occlusion (regularisation; bounded so a crack is never fully hidden)
    p_coarse_dropout: float = 0.15
    dropout_holes: int = 2
    dropout_frac: float = 0.06        # hole side as a fraction of the image side

    def describe(self) -> dict:
        return asdict(self)


POLICIES = {
    "none": AugmentPolicy(
        name="none", p_hflip=0, p_vflip=0, p_rot90=0, p_affine=0,
        p_brightness_contrast=0, p_gamma=0, p_channel_jitter=0, p_saturation=0, p_hue=0,
        p_noise=0, p_blur=0, p_sharpen=0, p_coarse_dropout=0),

    "geometric_only": AugmentPolicy(
        name="geometric_only",
        p_brightness_contrast=0, p_gamma=0, p_channel_jitter=0, p_saturation=0, p_hue=0,
        p_noise=0, p_blur=0, p_sharpen=0, p_coarse_dropout=0),

    "light": AugmentPolicy(
        name="light", p_affine=0.5, rotate_deg=10.0, scale_range=(0.95, 1.05),
        translate_frac=0.03,
        p_brightness_contrast=0.5, brightness_limit=0.12, contrast_limit=0.12,
        p_gamma=0.3, gamma_range=(0.85, 1.18),
        p_channel_jitter=0.3, channel_gain=0.06,
        p_saturation=0.2, p_hue=0.1,
        p_noise=0.2, noise_sigma=(1.0, 5.0),
        p_blur=0.1, p_sharpen=0.1, p_coarse_dropout=0.0),

    "medium": AugmentPolicy(name="medium"),

    "heavy": AugmentPolicy(
        name="heavy", p_affine=0.85, rotate_deg=25.0, scale_range=(0.85, 1.15),
        translate_frac=0.08,
        p_brightness_contrast=0.95, brightness_limit=0.40, contrast_limit=0.40,
        p_gamma=0.75, gamma_range=(0.60, 1.60),
        p_channel_jitter=0.85, channel_gain=0.22,
        p_saturation=0.6, saturation_range=(0.65, 1.35),
        p_hue=0.4, hue_shift_deg=10.0,
        p_noise=0.55, noise_sigma=(3.0, 14.0),
        p_blur=0.3, blur_sigma=(0.3, 1.4),
        p_sharpen=0.25, p_coarse_dropout=0.3, dropout_holes=3, dropout_frac=0.08),
}


# --------------------------------------------------------------------------- transforms
class EggAugment:
    """Apply a policy to an image, and identically to its crack mask if one is supplied.

    Parameters
    ----------
    policy : str or AugmentPolicy
    seed : int, optional
        Fixing the seed makes a call reproducible, which notebook 03 relies on to show the same
        egg under the same transform across policies.

    Returns
    -------
    (image, mask) where image is uint8 HxWx3 RGB and mask is uint8 HxW or None.
    """

    def __init__(self, policy="medium", seed: Optional[int] = None):
        self.p = POLICIES[policy] if isinstance(policy, str) else policy
        self.rng = np.random.default_rng(seed)
        # parameters of the most recent call, so notebook 03 can compute exact crack retention
        self.last_params: dict = {}

    # -- helpers
    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def _hit(self, prob):
        return bool(self.rng.random() < prob)

    # -- geometric
    def _geometric(self, img, mask):
        p = self.p
        if self._hit(p.p_hflip):
            img = img[:, ::-1]
            mask = None if mask is None else mask[:, ::-1]
        if self._hit(p.p_vflip):
            img = img[::-1]
            mask = None if mask is None else mask[::-1]
        if self._hit(p.p_rot90):
            k = int(self.rng.integers(1, 4))
            img = np.rot90(img, k)
            mask = None if mask is None else np.rot90(mask, k)
        img = np.ascontiguousarray(img)
        mask = None if mask is None else np.ascontiguousarray(mask)

        if p.p_affine > 0 and self._hit(p.p_affine):
            h, w = img.shape[:2]
            ang = self._u(-p.rotate_deg, p.rotate_deg)
            sc = self._u(*p.scale_range)                 # isotropic: elongation preserved
            self.last_params.update(angle=ang, scale=sc)
            tx = self._u(-p.translate_frac, p.translate_frac) * w
            ty = self._u(-p.translate_frac, p.translate_frac) * h
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
            M[0, 2] += tx
            M[1, 2] += ty
            # reflected border: never re-introduce the black padding that leaked class
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, mask

    # -- photometric
    def _photometric(self, img):
        p = self.p
        f = img.astype(np.float32)

        if self._hit(p.p_brightness_contrast):
            beta = self._u(-p.brightness_limit, p.brightness_limit) * 255.0
            alpha = 1.0 + self._u(-p.contrast_limit, p.contrast_limit)
            mean = f.mean()
            f = (f - mean) * alpha + mean + beta

        if self._hit(p.p_gamma):
            g = self._u(*p.gamma_range)
            f = np.power(np.clip(f, 0, 255) / 255.0, g) * 255.0

        if self._hit(p.p_channel_jitter):
            # independent per-channel gain destroys the red/blue ratio that leaked class
            gains = 1.0 + self.rng.uniform(-p.channel_gain, p.channel_gain, size=3)
            f = f * gains.reshape(1, 1, 3)

        f = np.clip(f, 0, 255).astype(np.uint8)

        if self._hit(p.p_saturation) or self._hit(p.p_hue):
            hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
            if p.p_saturation > 0:
                hsv[:, :, 1] *= self._u(*p.saturation_range)
            if p.p_hue > 0:
                # OpenCV hue is 0-179 for a 360 degree wheel
                hsv[:, :, 0] = (hsv[:, :, 0]
                                + self._u(-p.hue_shift_deg, p.hue_shift_deg) / 2.0) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return f

    # -- sensor
    def _sensor(self, img):
        p = self.p
        if self._hit(p.p_blur):
            s = self._u(*p.blur_sigma)
            img = cv2.GaussianBlur(img, (0, 0), s)
        elif self._hit(p.p_sharpen):
            s = self._u(*p.sharpen_amount)
            blur = cv2.GaussianBlur(img, (0, 0), 1.0)
            img = np.clip(img.astype(np.float32) * (1 + s) - blur.astype(np.float32) * s,
                          0, 255).astype(np.uint8)
        if self._hit(p.p_noise):
            sg = self._u(*p.noise_sigma)
            img = np.clip(img.astype(np.float32)
                          + self.rng.normal(0, sg, img.shape), 0, 255).astype(np.uint8)
        return img

    def _dropout(self, img, mask):
        p = self.p
        if p.p_coarse_dropout <= 0 or not self._hit(p.p_coarse_dropout):
            return img, mask
        h, w = img.shape[:2]
        side = int(p.dropout_frac * min(h, w))
        img = img.copy()
        for _ in range(p.dropout_holes):
            y = int(self.rng.integers(0, max(h - side, 1)))
            x = int(self.rng.integers(0, max(w - side, 1)))
            # fill with the local mean rather than black, for the same reason as above
            patch = img[y:y + side, x:x + side]
            img[y:y + side, x:x + side] = patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        return img, mask

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        self.last_params = {"angle": 0.0, "scale": 1.0}
        img, mask = self._geometric(img, mask)
        img = self._photometric(img)
        img = self._sensor(img)
        img, mask = self._dropout(img, mask)
        return img, mask


# --------------------------------------------------------------------------- normalisation
def standardise(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-image per-channel standardisation.

    This is the step that removes the absolute brightness, contrast and colour-balance
    differences which notebook 01b showed carry class information (mean luminance 168 for intact
    eggs against 143-154 for cracked ones, with 100 % of that difference surviving exclusion of the
    crack region, i.e. it is acquisition rather than signal). Applied after augmentation so that
    the augmented photometric variation is also normalised away, leaving the network with shape
    and texture only.
    """
    f = img.astype(np.float32)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        f[:, :, c] = (ch - ch.mean()) / (ch.std() + eps)
    return f


def to_tensor_hwc(img: np.ndarray) -> np.ndarray:
    """HWC float32 -> CHW float32, ready for torch.from_numpy."""
    return np.ascontiguousarray(img.transpose(2, 0, 1))
