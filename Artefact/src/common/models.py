"""
models.py
Backbones and heads for the four problem formulations compared in notebook 05.

Formulations
------------
(a) hierarchical   two independent models: a binary crack detector, then a three-class pattern
                   classifier trained on cracked eggs only. Built here as `single` models with
                   `n_out` 2 and 3 and orchestrated as two runs by the notebook.
(b) single4        one four-class model: Intact / Hairline / Star / Network.
(c) multitask      one shared backbone, a detection head and a pattern head. The pattern loss is
                   masked to cracked eggs.
(d) multitask_seg  as (c) plus a lightweight segmentation decoder supervised by the 335 annotation
                   polygons.

Formulation (d) is not an optional extra in this project. Notebook 04 established that the best
deployable classical pipeline reaches macro F1 0.565 on morphology while the same geometry
features computed from human polygons reach 0.829 -- a gap of 0.264 -- and that a classical ridge
detector recovers under 3 % of the annotated crack area. Locating the crack is the hard part, so
supervising the backbone to attend to crack pixels is the most promising intervention available.

Metadata fusion
---------------
`age_mode` controls how breeder age class enters the model, for the notebook 06 ablation:
    "none"   image only
    "concat" one-hot age appended to the pooled features before the heads
Notebook 01 found crack pattern and age class to be statistically independent (chi-square p =
0.225, Cramer's V = 0.092), so "concat" is expected to give no gain. Testing it anyway is the
point; the null result is what justifies fusing metadata at the decision layer instead.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BACKBONES", "build_model", "EggModel", "count_parameters"]

# timm names, chosen to fit a 6 GB RTX 3050 at 384 px with AMP and gradient accumulation
BACKBONES = {
    "mobilenetv3": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
    "resnet18": "resnet18",
    "convnext_tiny": "convnext_tiny",
    "vit_small": "vit_small_patch16_384",
}

# transformer backbones do not expose a usable spatial feature pyramid through timm's
# features_only interface, so the segmentation formulation is restricted to the CNNs
SEG_CAPABLE = {"mobilenetv3", "efficientnet_b0", "resnet18", "convnext_tiny"}


def _make_backbone(name: str, pretrained: bool, want_features: bool, img_size: int = 384):
    """Create a timm backbone and report its true output width.

    The output width is MEASURED with a dummy forward pass rather than read from
    `model.num_features`. In timm 1.x those two disagree for architectures with a pre-classifier
    convolution: `mobilenetv3_large_100` reports num_features = 960 (the trunk width) but
    forward() returns 1280 (after conv_head). Sizing the head from the attribute produced
    `mat1 and mat2 shapes cannot be multiplied (32x1280 and 960x4)`. Measuring is cheap and
    cannot disagree with reality.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "timm is required for the model factory. Install with: pip install timm"
        ) from e
    tname = BACKBONES.get(name, name)

    if want_features:
        m = timm.create_model(tname, pretrained=pretrained, features_only=True)
        return m, list(m.feature_info.channels())

    # Vision transformers need their input size fixed at construction so the position
    # embeddings are interpolated correctly; convolutional backbones ignore the argument.
    try:
        m = timm.create_model(tname, pretrained=pretrained, num_classes=0,
                              global_pool="avg", img_size=img_size)
    except TypeError:
        m = timm.create_model(tname, pretrained=pretrained, num_classes=0, global_pool="avg")

    was_training = m.training
    m.eval()
    with torch.no_grad():
        feat_dim = int(m(torch.zeros(1, 3, img_size, img_size)).shape[1])
    m.train(was_training)
    reported = getattr(m, "num_features", None)
    if reported is not None and int(reported) != feat_dim:
        print(f"[models] {tname}: forward width {feat_dim} differs from "
              f"num_features {reported}; using the measured value")
    return m, feat_dim


class SegHead(nn.Module):
    """Small FPN-style decoder producing a single-channel crack map at 1/4 resolution.

    Deliberately light. Its purpose is to push the backbone towards crack pixels, not to be a
    competitive segmentation model, and on 6 GB of VRAM every parameter spent here is a parameter
    not available to the classifier.
    """

    def __init__(self, channels: list[int], mid: int = 64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, mid, 1) for c in channels])
        self.smooth = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, feats):
        xs = [l(f) for l, f in zip(self.lat, feats)]
        target = xs[0].shape[-2:]
        x = sum(F.interpolate(t, size=target, mode="bilinear", align_corners=False) for t in xs)
        return self.smooth(x)


class EggModel(nn.Module):
    """Backbone plus the heads required by the chosen formulation."""

    def __init__(self, backbone: str = "efficientnet_b0", formulation: str = "single4",
                 n_out: int = 4, pretrained: bool = True, age_mode: str = "none",
                 n_age: int = 3, dropout: float = 0.2, img_size: int = 384):
        super().__init__()
        self.formulation = formulation
        self.age_mode = age_mode
        self.needs_seg = formulation == "multitask_seg"

        if self.needs_seg:
            if backbone not in SEG_CAPABLE:
                raise ValueError(
                    f"formulation 'multitask_seg' needs a CNN backbone; got '{backbone}'. "
                    f"Available: {sorted(SEG_CAPABLE)}")
            self.backbone, chans = _make_backbone(backbone, pretrained, want_features=True,
                                                  img_size=img_size)
            self.seg_head = SegHead(chans)
            feat_dim = chans[-1]
        else:
            self.backbone, feat_dim = _make_backbone(backbone, pretrained, want_features=False,
                                                     img_size=img_size)
            self.seg_head = None

        head_in = feat_dim + (n_age if age_mode == "concat" else 0)
        self.drop = nn.Dropout(dropout)

        if formulation in ("single", "single4"):
            self.head = nn.Linear(head_in, n_out)
            self.head_det = self.head_pat = None
        else:
            self.head = None
            self.head_det = nn.Linear(head_in, 2)
            self.head_pat = nn.Linear(head_in, 3)

    def forward(self, x, age_onehot: Optional[torch.Tensor] = None):
        out = {}
        if self.needs_seg:
            feats = self.backbone(x)
            out["seg"] = self.seg_head(feats)
            pooled = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        else:
            pooled = self.backbone(x)

        if self.age_mode == "concat":
            if age_onehot is None:
                raise ValueError("age_mode='concat' requires age_onehot")
            pooled = torch.cat([pooled, age_onehot.to(pooled.dtype)], dim=1)

        pooled = self.drop(pooled)
        if self.head is not None:
            out["logits"] = self.head(pooled)
        else:
            out["logits_det"] = self.head_det(pooled)
            out["logits_pat"] = self.head_pat(pooled)
        return out


def build_model(backbone: str, formulation: str, n_out: int = 4, pretrained: bool = True,
                age_mode: str = "none", img_size: int = 384) -> EggModel:
    return EggModel(backbone=backbone, formulation=formulation, n_out=n_out,
                    pretrained=pretrained, age_mode=age_mode, img_size=img_size)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
