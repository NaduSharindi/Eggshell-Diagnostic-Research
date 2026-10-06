# Research Design Decision Log

**Project:** A Multi-Modal Deep Learning Approach for Eggshell Micro-Crack Morphology Analysis and Root-Cause Diagnosis in Poultry Breeder Farms
**Student:** M.P.N. Sharindi — Coventry Index 16110429
**Status:** v1.0, written before any model is trained
**Purpose:** This document fixes the research design in advance so that Chapter 5 evaluates what Chapter 3 promised. It is the raw material for Chapter 3 and Section 6.5, and it is the thing to put in front of the supervisor at the next meeting.

---

## 1. Confirmed scope change

Objective 3 of the approved proposal (building the IoT edge imaging device) is **withdrawn**. The dataset was captured with a smartphone over a candling light in a light-proof box, which realises the *imaging condition* the IoT device would have provided but not the device itself.

Consequences that must be carried consistently through the whole dissertation:

* The title drops "IOT": *A Multi-Modal Deep Learning Approach for Eggshell Micro-Crack Morphology Analysis and Root-Cause Diagnosis in Poultry Breeder Farms*.
* Chapter 1 Scope lists edge-device engineering as **explicitly out of scope** with the descope reason and date.
* Chapter 2 still reviews Edge-AI/IoT work (Jebari et al., 2023) because it motivates the deployment constraint — the chosen model must be small enough to run on edge hardware — but no device is claimed.
* "Multi-modal" in this project means **image + biological metadata**, not image + sensor stream. This must be defined in Section 1.1 so an examiner is never confused about what the modalities are.
* Research Question 3 is rewritten from "how can an IoT device be built" to a deployment-feasibility question answerable from the evidence actually produced (model size, latency, parameter count on commodity hardware).

---

## 2. What the label file already proves (notebook 01, pre-modelling)

These results come from the annotation file alone and are independent of any model. They are stated here because they *change the architecture*.

### 2.1 Dataset composition

| | |
|---|---|
| Eggs | 652, one candled image each |
| Detection target | 317 Intact / 335 Micro-cracked (near-balanced) |
| Morphology target | Intact 317, Hairline 159, Network 108, Star 68 |
| Imbalance ratio | 4.66 (Intact : Star) |
| Breeder age class | Early lay 218, Peak production 217, Late lay 217 (balanced by design) |
| Annotation polygons | 335, 1–7 per cracked egg |
| Annotators | 1, verified by the hatchery manager |

### 2.2 Crack morphology is statistically independent of breeder age class

χ²(4) = 5.68, p = 0.225, Cramér's V = 0.092. Egg status versus age class: χ²(2) = 3.66, p = 0.160, V = 0.075.

**This is the most consequential finding in the design.** Breeder age carries essentially no information about *which* crack pattern appears. Therefore:

* Feeding age class into the image classifier as a fused feature **should not** improve pattern classification, and if it appears to, that is an overfitting artefact rather than a real gain.
* The correct architecture fuses metadata at the **decision layer** (pattern × age → root cause), not at the feature layer.
* Notebook 06 still runs the image-only versus image+age ablation, because an unevaluated claim is not a finding. A null result there is a *reportable contribution*: it shows that multi-modality in this domain operates diagnostically rather than perceptually, which is a sharper claim than "fusion helps".

This reframes the novelty honestly and defensibly, and it protects against an examiner asking "did you actually test whether the metadata helps?".

### 2.3 The root-cause mapping is a deterministic 3 × 3 expert rule base

| Pattern | Early lay | Peak production | Late lay |
|---|---|---|---|
| Hairline | 30 / 70 | 40 / 60 | 55 / 45 |
| Network | 50 / 50 | 60 / 40 | 75 / 25 |
| Star | 15 / 85 | 20 / 80 | 30 / 70 |

(physiological-nutritional % / mechanical-environmental %)

Ten distinct interpretation texts and ten distinct action texts, one per cell plus the Intact case. Because the mapping is deterministic, **root-cause accuracy is mathematically bounded by pattern-classification accuracy**. The root-cause layer is a knowledge-based expert system, not a learned model, and the dissertation must say so plainly. Notebook 10 therefore measures *error propagation*: how a pattern confusion translates into a wrong cause attribution and a wrong recommended action, including the cost asymmetry (telling a farm to change feed when the real problem is the egg belt is a different kind of error from the reverse).

### 2.4 The morphology taxonomy is structurally real, not nominal

Using only the annotation polygons — no pixels, no deep learning — a logistic-regression classifier separates the three patterns at macro F1 ≈ 0.79 under 5-fold CV. Elongation (ε² = 0.51), annotated area (ε² = 0.49) and band width (ε² = 0.42) all discriminate strongly, matching the guideline's definitions: Hairline is long and thin, Star is fragmented and spatially spread, Network is large and diffuse.

This pre-empts the strongest objection to the research: that the three classes are three names for one phenomenon.

> *Note:* exact values will be recomputed on the real images; those above are from the label file and are final for everything except the image-derived statistics.

---

## 3. Experimental design

### 3.1 Data partitioning (notebook 02)

* **Held-out test set: 15 % (≈ 98 eggs), stratified jointly on `label4` × `age_class`, drawn once and sealed.** It is opened exactly once, in notebook 07. No hyper-parameter, no architecture, no augmentation policy is chosen with reference to it.
* **Remaining 85 % (≈ 554 eggs): stratified 5-fold cross-validation** for every comparison in notebooks 04–06. All reported development numbers are mean ± SD across folds.
* Splitting is by `egg_id` and each egg has exactly one image, so there is no risk of the same egg appearing in both sides of a split.
* Augmentation is applied **inside the training fold only**, never to validation or test data.

Justification for the test fraction: with 68 Star eggs, a 15 % test set contains ≈ 10 Star examples. That is small, and the dissertation says so. The mitigation is bootstrap 95 % confidence intervals on every test metric plus the 5-fold development estimates, rather than pretending a single test number is precise.

### 3.2 Problem formulations to be compared (notebook 05)

| | Formulation | Rationale |
|---|---|---|
| (a) | Two-stage hierarchical: detect crack, then classify pattern | Matches the operational decision sequence; lets each stage be tuned separately |
| (b) | Single 4-class model (Intact/Hairline/Star/Network) | Simplest; one model to deploy |
| (c) | Multi-task: shared backbone, detection head + pattern head | Detection signal may regularise the pattern head |
| (d) | (c) + auxiliary segmentation supervision from the 335 polygons | Forces the backbone to attend to crack pixels; expected to improve Grad-CAM localisation, which is measured in notebook 09 |

Formulation (d) is the one that makes the explainability claim defensible, because it is the only one where the heatmaps are *supervised* toward the annotated crack region.

### 3.3 Backbones

MobileNetV3-Large, EfficientNet-B0, ResNet-18, ConvNeXt-Tiny, and one transformer (ViT-S/16 or Swin-Tiny) under identical training budgets, ImageNet-pretrained via `timm`. Selection criterion is macro F1 on the minority classes under CV, with parameter count and inference latency reported alongside so the edge-deployment argument is evidence-based rather than asserted.

### 3.4 Class imbalance

Compared, not assumed: class-weighted cross-entropy, focal loss, and weighted random sampling. Whichever wins on validation macro F1 is used; the comparison is reported.

### 3.5 Input resolution

Decided by the resolution-sensitivity measurement in notebook 01 §6, not by convention. The expectation is that 224 px discards too much crack detail and that the curve flattens somewhere between 320 and 448 px. The chosen size is defended in Chapter 4 with that figure.

### 3.6 Evaluation protocol (notebooks 08–09), fixed in advance

* Accuracy, macro and per-class precision/recall/F1, confusion matrices.
* ROC-AUC and PR-AUC for crack detection (PR-AUC is the honest one under imbalance).
* Bootstrap 95 % CIs (2000 resamples) on every test-set metric.
* McNemar's test against the runner-up model.
* Calibration curve and expected calibration error — a decision-support tool that reports a confident wrong cause is worse than one that reports uncertainty.
* Grad-CAM and Grad-CAM++ validated **quantitatively** against the polygons: pointing game accuracy, IoU, and energy-inside-polygon, reported per pattern. This is the part most dissertations skip and it is where marks are available.
* Error-analysis gallery of every misclassified test egg.

### 3.7 Pre-registered success thresholds (solution objectives)

Set now, measured in Chapter 5, reported honestly whether met or not.

| ID | Solution objective | Threshold | Measured in |
|---|---|---|---|
| SO1 | Detect micro-cracks in candled images | Test F1 ≥ 0.90, PR-AUC ≥ 0.93 | nb 08 |
| SO2 | Classify crack morphology into three types | Test macro F1 ≥ 0.75, no class below 0.60 | nb 08 |
| SO3 | Produce heatmaps that land on the real crack | Pointing-game ≥ 0.70, median energy-in-polygon ≥ 0.40 | nb 09 |
| SO4 | Determine whether metadata fusion improves perception | Ablation reported with CIs; direction stated either way | nb 06 |
| SO5 | Emit an auditable root-cause attribution per egg | 100 % rule coverage; error propagation quantified | nb 10 |
| SO6 | Run on commodity hardware | < 50 ms/image on GPU, < 25 M parameters | nb 07 |
| SO7 | Be usable by a production manager | SUS ≥ 68, all UAT tasks completed unaided | nb 12 |

Thresholds SO1–SO2 are set below the 98.73 % reported by Cengel et al. (2025) deliberately: that study classified visibly damaged eggs in a controlled setting with a larger dataset, whereas this study detects *micro*-cracks and additionally resolves morphology on 652 eggs. Chapter 5 must make that comparison explicitly rather than appearing to underperform.

---

## 4. Notebook roadmap

| # | Notebook | Status |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | **delivered** |
| 02 | `02_split_and_preprocessing.ipynb` | next |
| 03 | `03_augmentation.ipynb` | |
| 04 | `04_baselines.ipynb` | |
| 05 | `05_model_selection.ipynb` | |
| 06 | `06_metadata_fusion.ipynb` | |
| 07 | `07_final_training_and_test.ipynb` | |
| 08 | `08_evaluation.ipynb` | |
| 09 | `09_explainability.ipynb` | |
| 10 | `10_root_cause_engine.ipynb` | |
| 11 | `app/streamlit_app.py` | |
| 12 | `12_usability_analysis.ipynb` + evaluation pack | |

---

## 5. Risks carried forward

| Risk | Response |
|---|---|
| 68 Star eggs is few for a 4-class problem | Class weighting, augmentation, CV with SD, bootstrap CIs; reported as a limitation, never hidden |
| Single annotator → no inter-rater reliability | Stated as a threat to validity in 5.8. Mitigated if a second rater can label a 60-egg subset (see question 3 below) |
| Capture conditions could act as a shortcut feature | Screened in nb 01 §4; if any variable is significant with ε² > 0.06, it is neutralised in augmentation and the result is reported |
| Root-cause percentages are expert-elicited, not empirically validated | Framed as a knowledge-based layer whose *consistency* is evaluated, not its ground-truth correctness; validating the rule base against hatch outcomes is named as future work |
| Usability sample will be small (likely 5–10 managers) | SUS reported with the sample size stated; no inferential claim made from it |
