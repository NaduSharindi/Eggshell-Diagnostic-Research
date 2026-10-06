"""
project_config.py
Single source of truth for paths, class vocabularies and reproducibility settings.

Every notebook imports from here, so a path is changed in one place only.

Resolution order for the artefact root:
    1. environment variable  EGG_ARTEFACT_ROOT
    2. walk up from this file until a directory containing 'data' and 'notebooks' is found
Images directory can be overridden with  EGG_IMAGES_DIR.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- roots
_THIS = Path(__file__).resolve()


def _find_artefact_root() -> Path:
    env = os.environ.get("EGG_ARTEFACT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in _THIS.parents:
        if (parent / "data").is_dir() and (parent / "notebooks").is_dir():
            return parent
    # fallback: <repo>/Artefact assuming src/common/project_config.py
    return _THIS.parents[2]


ARTEFACT_ROOT: Path = _find_artefact_root()
REPO_ROOT: Path = ARTEFACT_ROOT.parent

DATA_DIR: Path = ARTEFACT_ROOT / "data"
NOTEBOOKS_DIR: Path = ARTEFACT_ROOT / "notebooks"
SRC_DIR: Path = ARTEFACT_ROOT / "src"
RESULTS_DIR: Path = ARTEFACT_ROOT / "results"
MODELS_DIR: Path = ARTEFACT_ROOT / "models"
APP_DIR: Path = ARTEFACT_ROOT / "app"

# --------------------------------------------------------------------------- data
# Cropped square candled images, one per egg, named E001.jpg ... E652.jpg
IMAGES_DIR: Path = Path(
    os.environ.get("EGG_IMAGES_DIR", DATA_DIR / "images" / "cropped")
).expanduser()

ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
METADATA_DIR: Path = DATA_DIR / "metadata"

LABELS_CSV: Path = ANNOTATIONS_DIR / "final_labels_v3.csv"
ROOT_CAUSE_RULES_CSV: Path = DATA_DIR / "root_cause_rules.csv"
AGE_CLASS_CSV: Path = ANNOTATIONS_DIR / "age_class.csv"
CROP_LOG_CSV: Path = METADATA_DIR / "crop_log.csv"
MAPPING_CSV: Path = METADATA_DIR / "mapping.csv"
REVIEW_LOG_CSV: Path = METADATA_DIR / "review_log.csv"

IMAGE_GLOB: str = "E*.jpg"

# --------------------------------------------------------------------------- vocab
PATTERNS: list[str] = ["Hairline", "Star", "Network"]
LABEL4: list[str] = ["Intact", "Hairline", "Star", "Network"]
STATUS: list[str] = ["Intact", "Micro_cracked"]
AGE_ORDER: list[str] = ["Early lay", "Peak production", "Late lay"]

LABEL4_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(LABEL4)}
IDX_TO_LABEL4: dict[int, str] = {i: c for c, i in LABEL4_TO_IDX.items()}
AGE_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(AGE_ORDER)}

# Colour-blind safe palette, used consistently in every figure in the dissertation
PALETTE: dict[str, str] = {
    "Intact": "#4C72B0",
    "Hairline": "#DD8452",
    "Star": "#55A868",
    "Network": "#C44E52",
}
AGE_PALETTE: dict[str, str] = {
    "Early lay": "#8C8C8C",
    "Peak production": "#937860",
    "Late lay": "#64B5CD",
}

# --------------------------------------------------------------------------- repro
RANDOM_SEED: int = 42
N_SPLITS: int = 5          # stratified k-fold on train+val
TEST_FRACTION: float = 0.15
VAL_FRACTION: float = 0.15  # of the non-test remainder


def seed_everything(seed: int = RANDOM_SEED) -> int:
    """Seed python, numpy and (if present) torch, including CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def results_dir(stage: str) -> Path:
    """results/<stage>/{figures,tables} created on demand."""
    d = RESULTS_DIR / stage
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d


def describe() -> str:
    lines = [
        f"ARTEFACT_ROOT : {ARTEFACT_ROOT}",
        f"IMAGES_DIR    : {IMAGES_DIR}  (exists={IMAGES_DIR.is_dir()})",
        f"LABELS_CSV    : {LABELS_CSV}  (exists={LABELS_CSV.is_file()})",
        f"RESULTS_DIR   : {RESULTS_DIR}",
        f"RANDOM_SEED   : {RANDOM_SEED}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
