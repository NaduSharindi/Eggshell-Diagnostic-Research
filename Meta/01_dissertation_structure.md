# Merged Dissertation Structure

**Project:** A Multi-Modal Deep Learning Approach for Eggshell Micro-Crack Morphology Analysis and Root-Cause Diagnosis in Poultry Breeder Farms
**Student:** M.P.N. Sharindi — Coventry Index 16110429
**Status:** v1.0 — for supervisor approval before Chapter 1 is drafted

## How the merge works

You were given two incompatible schemes: the **chapter outline** (six chapters, fixed top-level numbering) and the **self-check guidelines** (different numbering, extra mandatory items). This structure keeps the **chapter outline's top-level numbering exactly as issued** — so it tallies with what you were told to submit — and places every self-check requirement as a numbered sub-section beneath the heading it logically belongs to. Nothing from either document is dropped.

Column 3 shows which self-check checkbox each sub-section discharges, so you can tick the list directly against this table at submission.

**Target length:** 20,000 words (you confirmed 15,000–25,000). Percentages follow the self-check guideline's own allocation.

---

## Front matter

Title page · Abstract (150–250 words, written last) · Acknowledgements · Table of contents · List of figures · List of tables · List of abbreviations · Declaration

The abstract carries the headings your outline specifies: Background, Problem, Aim, Methodology, Artifact, Evaluation, Findings, Contribution, Keywords (4–7).

---

## Chapter 1 — Introduction  (~2,000 words, 10 %)

| § | Heading | Discharges |
|---|---|---|
| 1.1 | Background | 1.1 Background and Context |
| 1.2 | Research Problem | 1.2 Problem Statement |
| 1.2.1 | — Problem Statement | |
| 1.2.2 | — Provisional Research Gap | 1.3 Provisional Research Gap |
| 1.3 | Objectives of the Project | |
| 1.3.1 | — Aim (one sentence) | 1.4 Aim |
| 1.3.2 | — Project Objectives | 1.5 Project Objectives + "not solution objectives" |
| 1.4 | Research Questions | 1.6 Research Questions |
| 1.5 | Scope of the Research | 1.7 Scope and Delimitations |
| 1.5.1 | — In Scope | |
| 1.5.2 | — Out of Scope and Delimitations | |
| 1.6 | Justification of the Research | 1.8 Significance and Intended Contribution |
| 1.7 | Expected Limitations | |
| 1.8 | Chapter Summary | |
| 1.8.1 | — Structure of the Dissertation | 1.9 Structure of the Dissertation |

**Note under 1.3.2:** an explicit sentence stating that project objectives (what you did) are distinct from solution objectives (what the artefact must achieve), and that the latter appear in 3.5.2. The self-check penalises merging them.

---

## Chapter 2 — Literature Review  (~5,000 words, 25 %)

| § | Heading | Discharges |
|---|---|---|
| 2.1 | Introduction | |
| 2.1.1 | — Review Method (databases, search strings, date range, inclusion/exclusion, screening counts) | 2.1 Review Method |
| 2.2 | Theoretical Background and Conceptual Explanation of Key Concepts | 2.2 Theoretical Background |
| 2.2.1 | — Eggshell formation, breeder age and shell quality | |
| 2.2.2 | — Crack morphology taxonomy and its causal associations | |
| 2.2.3 | — Transillumination (candling) imaging | |
| 2.2.4 | — CNNs and transfer learning for fine-grained defect detection | |
| 2.2.5 | — Multi-modal fusion strategies (early, late, decision-level) | |
| 2.2.6 | — Explainability: the CAM family and localisation metrics | |
| 2.3 | Empirical Review of Previous Studies | 2.3 Thematic Review |
| 2.3.1 | — Theme A: Edge-AI and IoT environmental monitoring in poultry | |
| 2.3.2 | — Theme B: Computer-vision crack and damage detection | |
| 2.3.3 | — Theme C: Non-destructive shell-quality assessment (NIR, acoustic, translucency) | |
| 2.3.4 | — Theme D: Explainability in agricultural vision systems | |
| 2.3.5 | — Comparative Synthesis Table | 2.4 Comparative Synthesis Table |
| 2.3.6 | — Critical Appraisal of Limitations | 2.5 Critical Appraisal |
| 2.4 | Research Gap | 2.6 Gap and Justification |
| 2.5 | Chapter Summary | 2.7 Chapter Summary |

**2.3.5 table columns:** Study · Task · Modality · Dataset (size, source) · Model · Evaluation used · Reported result · Morphology resolved? · Metadata used? · Explainability validated? · Limitation. The last three columns are all "no" down the whole table. That pattern *is* your gap evidence, and you return to this exact table in 6.2.2.

---

## Chapter 3 — Methodology  (~3,000 words, 15 %)

| § | Heading | Discharges |
|---|---|---|
| 3.1 | Introduction | |
| 3.2 | Research Philosophy (pragmatism, justified) | 3.1 Research Approach |
| 3.3 | Research Approach (applied, quantitative, design-oriented) | 3.1 |
| 3.4 | Design Science Research Methodology | |
| 3.4.1 | — Justification of DSR and Rejected Alternatives | 3.2 Justification of DSR |
| 3.4.2 | — Mapping the Six DSRM Activities to This Project (table) | 3.3 Mapping the Six Activities |
| 3.5 | Research Process | |
| 3.5.1 | — Activity 1: Problem Identification | 3.4 Problem Identification |
| 3.5.2 | — Activity 2: Solution Objectives and Traceability Table | 3.5 Solution Objectives |
| 3.5.3 | — Activities 3–6: Design, Demonstration, Evaluation, Communication | |
| 3.5.4 | — Project Plan, Milestones and Risk Register | 3.9 Project Plan and Risks |
| 3.6 | Requirement Gathering | |
| 3.7 | Data Collection Methods | 3.6 Data Requirements |
| 3.7.1 | — Data Requirements, Source and Access Confirmation | |
| 3.7.2 | — Image Acquisition Protocol | |
| 3.7.3 | — Annotation Protocol and Domain-Expert Verification | |
| 3.8 | Software Development Methodology | |
| 3.8.1 | — Tools and Technologies | 3.8 Tools and Technologies |
| 3.9 | Artifact Evaluation Strategy (metrics, baselines, thresholds, test-data origin — all pre-specified) | 3.7 Evaluation Strategy |
| 3.10 | Data Analysis Techniques | |
| 3.11 | Research Validity and Reliability | |
| 3.12 | Ethical, Legal, Social and Professional Considerations | 3.10 ELSP Issues |
| 3.13 | Chapter Summary | 3.11 Chapter Summary |

**3.5.2 is the highest-value section in the chapter.** Five columns, one row per solution objective: *Limitation found in the literature (cite 2.3.6) → Solution objective → Design decision that delivers it (forward-ref 4.3.1) → How it will be measured → Where it is reported (5.x)*. Section 6.3 re-uses this table with the measured values filled in. An unmeasured solution objective is a lost mark.

---

## Chapter 4 — Proposed Solution  (~4,000 words, 20 %)

| § | Heading | Discharges |
|---|---|---|
| 4.1 | Introduction | |
| 4.2 | Requirement Analysis (functional + non-functional, each traced to a 3.5.2 row) | 4.1 Requirements Specification |
| 4.3 | Proposed Solution | |
| 4.3.1 | — Design Decisions and Rejected Alternatives | 4.4 Design Decisions and Rejected Alternatives |
| 4.4 | System Architecture (diagram + explanation) | 4.2 System Architecture |
| 4.5 | System Design (components, data flow, interfaces) | 4.3 Component, Data and Algorithm Design |
| 4.6 | Algorithms/Models | 4.3 |
| 4.6.1 | — Problem formulations compared (hierarchical / 4-class / multi-task / segmentation-supervised) | |
| 4.6.2 | — Backbone selection and the edge-deployment constraint | |
| 4.6.3 | — Metadata fusion design and why it sits at the decision layer | |
| 4.6.4 | — Root-cause inference engine and its evidence base | |
| 4.6.5 | — Explainability design and localisation validation | |
| 4.7 | Data Sets/Data Preparation | |
| 4.7.1 | — Dataset provenance, annotation and expert verification | |
| 4.7.2 | — Pre-processing and the resolution policy (justified empirically) | |
| 4.7.3 | — Augmentation policy and its candling-specific constraints | |
| 4.7.4 | — Partitioning strategy | |
| 4.8 | Technology Selection | |
| 4.9 | Implementation | |
| 4.9.1 | — Implementation of the Core Contribution | 4.5 Core Contribution |
| 4.9.2 | — Supporting Implementation (dashboard, utilities) | 4.6 Supporting Implementation |
| 4.9.3 | — Deviations from the Design and Challenges Faced | 4.7 Deviations and Challenges |
| 4.10 | Chapter Summary | 4.8 Chapter Summary |

**4.9.1 gets the most space in the chapter** and must be explicit about what is yours: the morphology taxonomy and its operationalisation, the decision-layer fusion architecture, the polygon-supervised explainability, and the age-aware root-cause engine. 4.9.3 is where the withdrawn IoT objective and the 6 GB VRAM constraints are reported honestly — that gains marks rather than losing them.

---

## Chapter 5 — Evaluation and Results  (~4,000 words, 20 %)

| § | Heading | Discharges |
|---|---|---|
| 5.1 | Introduction | |
| 5.2 | Evaluation Framework | 5.1 Evaluation Design |
| 5.2.1 | — Evaluation Design (restating 3.9 so the reader sees you followed it) | 5.1 |
| 5.2.2 | — Experimental Setup (hardware, library versions, seeds, repetitions) | 5.2 Experimental Setup |
| 5.2.3 | — Dataset Description and Partitioning (precise origin of the test set) | 5.3 Dataset and Partitioning |
| 5.3 | Evaluation Criteria | |
| 5.4 | Experimental/Test Results — **organised by research question** | 5.5 Results |
| 5.4.1 | — Functional Testing (brief; full tables in appendix) | 5.4 Functional Testing |
| 5.4.2 | — RQ1: micro-crack detection | |
| 5.4.3 | — RQ2: morphology classification and formulation comparison | |
| 5.4.4 | — RQ3: metadata fusion ablation | |
| 5.4.5 | — RQ4: explanation localisation against the crack polygons | |
| 5.4.6 | — RQ5: root-cause error propagation | |
| 5.5 | System Performance Results (latency, VRAM, parameter count, model size) | |
| 5.6 | Accuracy/Precision/Recall or Other Technical Measures | |
| 5.6.1 | — Statistical Treatment (bootstrap CIs, McNemar, calibration) | 5.7 Statistical Treatment |
| 5.7 | User Evaluation (task-based UAT, 10 production managers) | |
| 5.8 | Usability Evaluation (SUS) | |
| 5.9 | Comparison with Existing Solutions | 5.6 Comparative Analysis Against Baselines |
| 5.10 | Results Related to Research Objectives | |
| 5.10.1 | — Threats to Validity | 5.8 Threats to Validity |
| 5.11 | Chapter Summary (headline findings, no interpretation) | 5.9 Chapter Summary |

**5.9 must compare against the strongest alternative, not the easiest.** That means your own best single-formulation model and a re-implementation of a binary classifier in the style of Cengel et al. (2025) trained on your data — not a majority-class baseline alone.

---

## Chapter 6 — Discussion, Conclusions and Recommendations  (~2,000 words, 10 %)

| § | Heading | Discharges |
|---|---|---|
| 6.1 | Introduction | |
| 6.2 | Discussion of Findings | |
| 6.2.1 | — Interpretation of Findings (what the numbers mean) | 6.1 Interpretation |
| 6.2.2 | — Contribution in Relation to the Literature (return to the 2.3.5 table) | 6.3 Contribution vs Literature |
| 6.2.3 | — Limitations | 6.4 Limitations |
| 6.3 | Achievements of Research Objectives (two tables: project objectives with evidence, solution objectives with measured results) | 6.5 Achievement of Aim and Objectives |
| 6.4 | Answers to Research Questions (one at a time, directly) | 6.2 Answers to RQs |
| 6.5 | Recommendations (for breeder-farm practice) | |
| 6.6 | Future Research (follows from 6.2.3, not a wish list) | 6.6 Future Work |
| 6.7 | Conclusion | |
| 6.7.1 | — Critical Reflection (what you learned, what you'd do differently) | 6.7 Critical Reflection |
| 6.7.2 | — Concluding Statement (no new information) | 6.8 Conclusion |
| 6.8 | Chapter Summary | |

---

## Back matter

References (APA 7th) · Appendix A Ethics approval and company data-use consent · Appendix B Progress monitoring sheet and supervision log · Appendix C Full project plan and Gantt chart · Appendix D Annotation guideline and Label Studio configuration · Appendix E Complete test and functional-test tables · Appendix F SUS questionnaire, UAT script and consent text · Appendix G Root-cause rule base with evidence sources · Appendix H Repository link and code listing

> **Evidence lives in the appendices; argument lives in the chapters.** The self-check marks this explicitly.

---

## Revised aim, objectives and research questions

These need your supervisor's sign-off before Chapter 1 is written, because they changed when Objective 3 was withdrawn.

### Aim (1.3.1)

> To design, implement and evaluate a multi-modal deep learning decision-support artefact that detects micro-cracks in candled hatching eggs, classifies their crack morphology, and combines that morphology with breeder age class to attribute a probable root cause for poultry breeder farm managers.

### Project objectives (1.3.2)

| | |
|---|---|
| PO1 | Critically review the literature on automated egg inspection, shell-quality determinants and explainable vision to establish the research gap. |
| PO2 | Prepare, validate and characterise an expert-annotated dataset of candled hatching-egg images with crack morphology and breeder age metadata. |
| PO3 | Design and comparatively evaluate deep learning formulations and backbones for joint micro-crack detection and morphology classification. |
| PO4 | Determine empirically whether breeder age metadata improves perceptual classification, and design the fusion architecture accordingly. |
| PO5 | Implement and quantitatively validate an explainability component against expert crack annotations. |
| PO6 | Implement an age-aware root-cause inference engine and a Streamlit decision-support dashboard. |
| PO7 | Evaluate the artefact technically and with poultry production managers. |
| PO8 | Document the research as a dissertation and research paper. |

### Solution objectives (3.5.2)

Not listed here. They are the artefact-level, measurable targets SO1–SO7 in `00_research_design_decisions.md` §3.7, and they must not be merged with the project objectives above.

### Research questions (1.4)

| | | Objective |
|---|---|---|
| RQ1 | To what extent can a convolutional model detect micro-cracks in candled hatching-egg images, relative to the binary classifiers reported in the literature? | O1 |
| RQ2 | Can crack morphology be classified into hairline, star and network types, and which problem formulation and backbone perform best under cross-validation? | O1 |
| RQ3 | Does fusing breeder age class with image features improve morphology classification, and if not, where in the architecture does the metadata belong? | O2 |
| RQ4 | Do the model's visual explanations localise the expert-annotated crack region, and does this differ by morphology type? | O3 |
| RQ5 | How do morphology classification errors propagate into root-cause attributions and the resulting recommended actions? | O2 |
| RQ6 | How usable and operationally effective is the decision-support dashboard for poultry production managers? | O4 |

### Proposed objective substitution — needs supervisor confirmation

The approved proposal has four objectives. Deleting Objective 3 (IoT) leaves three, which makes the dissertation look reduced against the approved panel document. The cleaner move is to **substitute** rather than delete:

| | Approved proposal | Revised |
|---|---|---|
| O1 | Detect micro-cracks and classify morphology | unchanged |
| O2 | Determine probable root cause from morphology + breeder age | unchanged |
| O3 | ~~Build an IoT edge imaging system~~ | **Provide explainable, spatially validated visual evidence for every diagnostic decision** |
| O4 | Evaluate the Streamlit dashboard with production managers | unchanged |

The replacement O3 is not invented: explainable heatmaps are already named in your proposal's scope and novelty statement, and the quantitative polygon validation is work you are doing regardless. This keeps four objectives, preserves the panel-approved shape, and documents the substitution with a date and reason in 1.5.2 and 4.9.3.

**Ask your supervisor to confirm this substitution in writing and file it in `Administrative/Approvals/`.**
