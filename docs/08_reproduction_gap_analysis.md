# Analysis: Reproduction Gap vs Paper Results

> Document analyzing why baseline reproductions in this repo underperform compared to reported paper results.

---

## Executive Summary

| Model | Paper (CAER-S) | Reproduced | Gap |
|-------|----------------|------------|-----|
| CAER-Net-S | 73.51% | ~70% | -3.5% |
| GLAMOR-Net (original) | 77.90% | - | - |
| GLAMOR-Net (ResNet-18) | 89.88% | - | - |
| EmotiCon + CCIM | 91.17% | ~62% | -29% |
| CAHFW-Net | 83.76% | - | - |

The gaps range from minor (~3%) to catastrophic (~29%). The root causes are architectural mismatches, wrong backbones, incorrect training protocols, and missing implementation details.

---

## Root Cause Analysis

### 1. Backbone Mismatch (Critical for ALL models)

**The Problem:**
All configs use `backbone: resnet18`. However, the original papers use **different backbones**:

| Paper | Original Backbone | Our Reproduction |
|-------|-------------------|------------------|
| CAER-Net-S | Custom CNN (5 layers: 32→64→128→256→256) | ResNet-18 |
| GLAMOR-Net (original) | Custom CNN (5 layers: 32→64→128→256→256) | ResNet-18 |
| GLAMOR-Net (reported 89.88%) | ResNet-18 pretrained | ResNet-18 pretrained ✅ |
| Yang CCIM (base model) | ResNet-152 / EmotiCon | ResNet-18 |
| CAHFW-Net | Custom CNN (5 layers) | ResNet-18 |

**Impact:**
- Custom CNNs in papers are **shallow and narrow** (~2-5M params), designed specifically for CAER-S.
- ResNet-18 is deeper (11M params) and may overfit on CAER-S (~49K train images).
- **Pretrained weights matter**: Papers pretrained on ImageNet; we use ImageNet pretrained too, but the architecture itself differs.

**Fix:**
Implement the **custom 5-layer CNN encoder** from the papers as an option in `common.py`:
```python
def _make_custom_encoder(channels_list=[32,64,128,256,256]) -> nn.Module:
    """CAER-Net / GLAMOR-Net / CAHFW-Net shallow encoder."""
```

---

### 2. CCIM: Wrong Confounder Feature Extractor (Critical for Yang CCIM)

**The Problem:**
Yang et al. (CVPR 2023) explicitly state:
> "We use **ResNet-152 pretrained on Places365** to extract context features for building the confounder dictionary."

Our `confounder_builder.py` uses:
```python
context_encoder, _ = _make_encoder(backbone, pretrained=pretrained)
# backbone = resnet18 (from config)
```

**Impact:**
- ResNet-152+Places365 produces **scene-centric features** (critical for identifying context bias).
- ResNet-18+ImageNet produces **object-centric features** (worse for context/scene representation).
- Confounder dictionary quality directly impacts debiasing effectiveness.
- Paper reports 91.17% with **EmotiCon + CCIM**, not standalone CCIM. EmotiCon is a strong baseline (~88%). Our standalone CCIM on weak ResNet-18 cannot reach that.

**Fix:**
1. Download `resnet152_places365` pretrained weights.
2. Use it **only for confounder dictionary extraction**:
   ```python
   # In confounder_builder.py
   if method == "yang_ccim" or method == "cd_ica_net":
       confounder_encoder = load_resnet152_places365()
   ```
3. Consider using a stronger base model (ResNet-50/101) for the main network.

---

### 3. Input Preprocessing Mismatch

**The Problem:**
| Paper | Face Input | Context Input | Our Code |
|-------|-----------|--------------|----------|
| CAER-Net-S | 96×96 (crop face) | 224×224 (mask face→black) | Both 224×224 |
| GLAMOR-Net | 96×96 | 224×224 (mask face→black) | Both 224×224 |
| Yang CCIM | 224×224 | 224×224 (mask face→black) | Both 224×224 |

**Our `caers_dataset.py` face masking:**
```python
draw.rectangle([x1, y1, x2, y2], fill=(127, 127, 127))  # GRAY
```

**Paper masking:**
> "Set pixel = 0 di dalam bounding box" (BLACK, not gray)

Gray (127) leaves some information; black (0) completely removes the face region.

**Impact:**
- Face stream should receive **cropped face only** (96×96 for CAER-Net/GLAMOR).
- Context stream should receive **full image with face completely blacked out**.
- Current code feeds full 224×224 image to face encoder (not cropped) and masks with gray instead of black.

**Fix:**
1. Crop face region from bounding box, resize to 96×96 for face branch (CAER-Net/GLAMOR).
2. Change mask fill from `(127,127,127)` to `(0,0,0)`.
3. For Yang CCIM and CD-ICA-Net, keep 224×224 but ensure proper masking.

---

### 4. Optimizer & Training Protocol Mismatch

**The Problem:**
| Paper | Optimizer | LR | Schedule | Epochs |
|-------|-----------|-----|----------|--------|
| CAER-Net-S | SGD + Nesterov (m=0.9) | 1e-3 | Cosine Annealing | 50+ |
| GLAMOR-Net | SGD + Nesterov (m=0.9) | 1e-3 | Step decay | 50+ |
| Yang CCIM | SGD + Nesterov (m=0.9) | 1e-3 | Cosine Annealing | 50+ |
| CAHFW-Net | SGD + Nesterov (m=0.9) | 5e-4 | Cosine Annealing | 50+ |
| **Our Code** | **AdamW** | **1e-3** | **None** | **30** |

**Impact:**
- AdamW behaves differently from SGD+Nesterov on small datasets.
- 30 epochs may be insufficient (papers train 50+).
- No LR scheduling means model may converge to suboptimal local minima.

**Fix:**
1. Switch to `SGD(lr=1e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)`.
2. Add `CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)`.
3. Increase `num_epochs` to 50-60.

---

### 5. Missing Data Augmentation Strategies

**The Problem:**
Papers use specific augmentation:
- RandomHorizontalFlip
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
- RandomCrop (after resize)

Our `augmented_transform` exists but is **not enabled by default** (`--augment` flag required).

**Impact:**
- Training without augmentation on ~49K images leads to overfitting.
- Paper results likely used augmentation by default.

**Fix:**
Enable augmentation by default or match paper preprocessing exactly.

---

### 6. Yang CCIM: Standalone vs Plug-in Architecture

**The Problem:**
Paper reports **91.17% for EmotiCon + CCIM**, NOT standalone CCIM.

| Setup | Reported Accuracy |
|-------|-------------------|
| EmotiCon alone | ~88% |
| EmotiCon + CCIM | 91.17% |
| Random baseline + CCIM | ~75% |

Our reproduction implements **CCIM as a standalone model** (ResNet-18 + CCIM). Without a strong base model, CCIM cannot perform well.

**Impact:**
- CCIM is a **plug-in debiasing module**, not a standalone architecture.
- The +3% gain from CCIM is on top of an already strong baseline.
- Our standalone CCIM gets ~62%, which is actually reasonable for a weak ResNet-18 + basic fusion + CCIM.

**Fix:**
Implement a stronger base model (e.g., ResNet-50, EfficientNet) before attaching CCIM.

---

### 7. Face Detection & Bounding Box Quality

**The Problem:**
The manifest (`manifest_caers.jsonl`) contains `face_bbox` from some face detector (likely Dlib or MTCNN). If bounding boxes are inaccurate:
- Face crop will include background or miss parts of face.
- Context mask will leave face remnants or mask non-face regions.

**Impact:**
- Noisy face stream → worse face feature extraction.
- Noisy context mask → context stream still sees face → context bias increases.

**Fix:**
Verify face bbox quality. Consider re-running face detection with a modern detector (RetinaFace, YuNet).

---

## Summary of Fixes Needed (Priority Order)

### High Priority (Will Close Most of the Gap)

1. **Implement custom 5-layer CNN encoder** (`common.py`) matching CAER-Net/GLAMOR-Net/CAHFW-Net papers.
2. **Fix face masking**: change fill from gray `(127,127,127)` to black `(0,0,0)`.
3. **Switch optimizer to SGD+Nesterov** with cosine annealing.
4. **Add ResNet-152-Places365** confounder extractor for Yang CCIM.
5. **Separate face/context transforms**: face 96×96 crop, context 224×224 full image with black mask.

### Medium Priority

6. Increase training epochs to 50-60.
7. Enable augmentation by default.
8. Use stronger backbone (ResNet-50) for Yang CCIM base model.

### Low Priority

9. Verify face bounding box quality in manifest.
10. Hyperparameter grid search for optimal LR, WD, dropout.

---

## Expected Results After Fixes

| Model | Current | After Fixes (Estimate) | Paper |
|-------|---------|----------------------|-------|
| CAER-Net-S | ~70% | 72-74% | 73.51% |
| GLAMOR-Net | - | 76-79% | 77.90% |
| Yang CCIM | ~62% | 75-80%* | 91.17%** |
| CAHFW-Net | - | 81-84% | 83.76% |

\* Still below paper because we don't reproduce EmotiCon base model.
\*\* 91.17% is EmotiCon+CCIM; standalone CCIM with ResNet-50 base might reach ~80%.

---

## Action Plan for Next Steps

1. **Phase 1**: Fix preprocessing (black mask, separate face/context sizes).
2. **Phase 2**: Implement custom CNN encoder + switch to SGD+Cosine.
3. **Phase 3**: Add ResNet-152-Places365 confounder extractor.
4. **Phase 4**: Retrain all baselines and compare.
5. **Phase 5**: Proceed with CD-ICA-Net experiments.
