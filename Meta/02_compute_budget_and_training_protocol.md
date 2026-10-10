# Compute Budget and Staged Training Protocol

**Addendum to** `00_research_design_decisions.md` (v1.0), which it supersedes on §3.3 (backbones) and §5 (risks).
**Reason:** the target hardware is now known and it materially constrains the experimental design.

---

## 1. Target hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, **6 GB VRAM** |
| CPU | Intel Core i7-13650HX (14 cores / 20 threads) |
| System RAM | 16 GB |
| Machine | Lenovo LOQ |

**6 GB is the binding constraint on this entire project.** It is enough — plenty of published work trains on less — but it rules out the naive plan of training five large backbones at high resolution across four formulations and five folds. The protocol below is designed around it, and the constraint itself gets reported in Section 4.9.3 and 5.2.2 as a documented design pressure rather than hidden.

### Before anything else: verify the CUDA stack

You reported CUDA 13.2.2. That number does not match any released CUDA Toolkit version (the toolkit line runs 12.x → 13.0), so it is most likely the **driver** version or a misreading. This matters because PyTorch wheels are built against specific CUDA runtimes.

Run both of these and send me the output:

```bash
nvidia-smi
```
```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

If `torch.cuda.is_available()` is `False`, nothing else in this project will run at a usable speed, so fix it before notebook 02. Install command (CUDA 12.1 wheels work with any driver ≥ 530 and are the safest choice):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Memory strategy

Four measures, all standard, all reportable:

| Measure | Effect | Where implemented |
|---|---|---|
| **Automatic mixed precision** (`torch.amp`, bf16 or fp16 + `GradScaler`) | roughly halves activation memory, ~1.5–2× faster on Ampere tensor cores | nb 05 training loop |
| **Gradient accumulation** | keeps the *effective* batch at 32 while the physical batch fits in 6 GB, so results stay comparable across backbones of different sizes | nb 05 |
| **`channels_last` memory format** | better tensor-core utilisation on Ampere for convolutional backbones | nb 05 |
| **Pre-decoded image cache** | removes the JPEG-decode bottleneck, which on 652 images of 1300–3500 px is the real limiting factor, not the GPU | nb 02 |

### The image cache is the single biggest speed win available

652 images at 448 × 448 × 3 as `uint8` is **392 MB** — trivially cacheable in 16 GB of RAM as a single `numpy` memmap. Decoding a 3000 px JPEG takes 50–150 ms; reading a cached array takes under a millisecond. Across 30 epochs × 5 folds that is the difference between hours and minutes.

So notebook 02 writes `data/processed/cache_448.npy` plus an index, and every subsequent notebook loads from it. Augmentation then operates on the cached 448 px array and random-crops or resizes down to the training resolution, which also means the augmentation pipeline never touches disk.

---

## 3. Revised backbone shortlist

The original five-backbone list in `00_research_design_decisions.md` §3.3 is cut to four plus one conditional entry.

| Backbone | Params | Fits at 384 px? | Keep? |
|---|---|---|---|
| MobileNetV3-Large | 5.4 M | yes, batch 32 | **yes** — the edge-deployment argument depends on it |
| EfficientNet-B0 | 5.3 M | yes, batch 24 | **yes** — strong accuracy per parameter; likely the winner |
| ResNet-18 | 11.7 M | yes, batch 32 | **yes** — the standard reference point; what a reviewer expects to see |
| ConvNeXt-Tiny | 28.6 M | batch 8 + accumulation | **yes** — the modern-architecture representative |
| ViT-S/16 @ 384 | 22 M | marginal; attention is quadratic in tokens | **conditional** — attempt once; if it OOMs at batch 4 with AMP, exclude it and *report the exclusion with the reason* |
| EfficientNetV2-S, Swin, ResNet-50 | — | — | **dropped** — not enough VRAM headroom to train them fairly, and an unfairly-trained comparison is worse than no comparison |

Dropping models for a stated hardware reason is legitimate and marks well. Dropping them silently does not.

> **On the transformer:** with 652 images, a ViT trained from ImageNet weights is unlikely to beat a well-tuned EfficientNet-B0 regardless of VRAM — transformers are data-hungry and this is a small dataset. If it is excluded, Section 5.9 should say that explicitly rather than implying the hardware alone decided it.

---

## 4. Staged experimental protocol

Running every backbone against every formulation across five folds is 100+ training runs. Instead, three stages, each narrowing the field. This is standard practice and is itself a defensible methodological choice to describe in Section 3.9.

### Stage A — backbone screen (25 runs)

Fix the formulation to **(b) single 4-class**. Train all five backbones × 5 folds at the resolution chosen in notebook 01. Select the **top two** by mean macro F1.

*Estimated: 4–7 hours. Run overnight.*

### Stage B — formulation comparison (40 runs)

Take the top two backbones. Run all four formulations — (a) hierarchical, (b) 4-class, (c) multi-task, (d) multi-task + polygon segmentation — × 5 folds.

*Estimated: 6–10 hours. Run overnight.*

### Stage C — refinement and ablation

Best backbone × best formulation only:
* class-imbalance strategy comparison (class weights vs focal loss vs weighted sampling)
* learning rate and augmentation-strength sweep, small grid
* the metadata-fusion ablation of notebook 06

*Estimated: 3–5 hours.*

### Final

Retrain the single selected configuration on all non-test data with 3 seeds, evaluate once on the sealed test set.

*Estimated: under 1 hour.*

**Total GPU time: roughly 15–25 hours, spread over three or four overnight runs.** Every notebook will checkpoint after each fold and write results to CSV incrementally, so an interrupted run resumes instead of restarting.

---

## 5. Non-negotiable instrumentation

Every training run logs, and every notebook writes to a single append-only `results/runs.csv`:

* run id, stage, backbone, formulation, fold, seed
* every hyper-parameter
* per-epoch train/val loss and macro F1
* wall-clock seconds, `torch.cuda.max_memory_allocated()`, physical batch size, effective batch size
* library versions and the git commit hash

This is what fills Section 5.2.2 (Experimental Setup) and 5.5 (System Performance Results). Reconstructing it afterwards from memory is not possible, so it is built in from notebook 05 onward rather than added later.

---

## 6. Updated risk register

| Risk | Response |
|---|---|
| CUDA stack misconfigured or unavailable | Verify before notebook 02 (§1 above). Everything downstream depends on it |
| 6 GB VRAM forces unfair comparisons between large and small backbones | Fix the *effective* batch size at 32 via gradient accumulation so the optimisation budget is identical regardless of physical batch |
| Thermal throttling on a laptop GPU distorts latency measurements | Report latency from a dedicated timing run after a fixed warm-up, not from the training logs |
| 15–25 h of GPU time is lost to a crash or a bug found late | Checkpoint every fold; write results incrementally; run Stage A on a single fold first as a smoke test before committing to the full run |
| Pre-decoded cache at the wrong resolution forces a full re-run | Cache at 448 px — above every candidate training resolution — so the training size can change without regenerating the cache |
