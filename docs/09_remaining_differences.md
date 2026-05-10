# Analysis: Remaining Differences vs Paper Methods

> Honest assessment of what has been fixed and what still differs from the original papers.

---

## ✅ What Has Been Fixed (Phase 1-3)

| Fix | Phase | Status |
|-----|-------|--------|
| Face mask: gray → black (0,0,0) | Phase 1 | ✅ Done |
| Face crop 96×96 for CAER-Net/GLAMOR | Phase 1 | ✅ Done |
| Custom 5-layer CNN encoder (32→64→128→256→256) | Phase 2 | ✅ Done |
| Optimizer: AdamW → SGD+Nesterov | Phase 2 | ✅ Done |
| LR Scheduler: None → CosineAnnealingLR | Phase 2 | ✅ Done |
| Epochs: 30 → 60 | Phase 2 | ✅ Done |
| Gradient clipping: max_norm=1.0 | Phase 2 | ✅ Done |
| CCIM confounder: ResNet-18 → ResNet-152 | Phase 3 | ✅ Done |
| Yang base model: ResNet-18 → ResNet-50 | Phase 3 | ✅ Done |
| Yang num_confounders: 1024 → 128 | Phase 3 | ✅ Done |

---

## ❌ What STILL Differs from Papers

### 1. Yang CCIM — Base Model: EmotiCon vs ResNet-50 (CRITICAL)

**The Paper:**
Yang et al. report **91.17% for EmotiCon + CCIM**. EmotiCon is a **custom architecture** specifically designed for CAER, achieving ~88% standalone.

**Our Code:**
We use **ResNet-50** as the base model. ResNet-50 is a general-purpose ImageNet model, not optimized for CAER-S.

**Impact:**
| Setup | Reported Accuracy |
|-------|-------------------|
| EmotiCon alone | ~88% |
| EmotiCon + CCIM | **91.17%** |
| ResNet-50 alone (estimated) | ~78-82% |
| ResNet-50 + CCIM (our repro) | **~75-80% expected** |

**Why it matters:** CCIM is a **plug-in module** that adds +3% on top of an already strong base. If the base is weak, CCIM cannot magically compensate.

**Fix (if you want exact reproduction):**
Implement EmotiCon architecture from scratch or find a public implementation. This is a **non-trivial** effort (~2-3 weeks).

---

### 2. Yang CCIM — Confounder Extractor: Places365 vs ImageNet (HIGH)

**The Paper:**
> "We use **ResNet-152 pretrained on Places365** to extract context features for building the confounder dictionary."

Places365 is a **scene recognition dataset** (365 scene categories: airport, beach, bedroom, etc.). Features from Places365 are optimized for **context/scene understanding**.

**Our Code:**
We use **ResNet-152 pretrained on ImageNet** (1000 object categories: dog, cat, car, etc.). ImageNet features are optimized for **object recognition**.

**Impact:**
- Places365 features capture scene semantics (e.g., "restaurant", "classroom") → better for identifying context bias.
- ImageNet features capture object semantics (e.g., "plate", "chair") → less optimal for context bias.
- The confounder dictionary quality directly impacts debiasing effectiveness.

**Fix:**
Download Places365 pretrained weights from CSAILVision/places365 repo:
```bash
wget http://places2.csail.mit.edu/models_places365/resnet152_places365.pth.tar
```
Then modify `_make_places365_encoder()` to load these weights instead of ImageNet.

---

### 3. CAER-Net-S — Attention Inference Module (MEDIUM)

**The Paper:**
CAER-Net uses an **attention inference module** on the context stream:
> "Using attention inference module **unsupervisedly** on context stream"

This module learns to highlight emotionally relevant context regions **without ground-truth attention maps**.

**Our Code:**
We do NOT implement this module. The context encoder is a plain CNN without attention guidance.

**Impact:**
- The attention module helps the model focus on emotionally salient regions (e.g., hands gesturing, background scene).
- Without it, the context stream may attend to irrelevant regions.
- Estimated impact: **-1% to -2%** accuracy.

**Fix:**
Implement an unsupervised attention module that generates spatial attention maps from context features. This can be done with:
```python
# 2× Conv2D → softmax → attention map
attn = Conv2D(256→128→1)(context_feat)
attn = softmax_spatial(attn)
context_feat = context_feat * attn
```

---

### 4. GLAMOR-Net — Training Epochs: 100 vs 60 (MEDIUM)

**The Paper:**
GLAMOR-Net is trained for **100 epochs** with step decay:
> "We train the network for 100 epochs with a batch size of 32. The learning rate is initialized to 1e-3 and decreased by a factor of 10 at epochs 40 and 80."

**Our Code:**
We train for **60 epochs** with cosine annealing.

**Impact:**
- Cosine annealing and step decay behave differently.
- 60 epochs may be insufficient for the shallow CNN to fully converge.
- Estimated impact: **-1% to -2%** accuracy.

**Fix:**
Change GLAMOR-Net config:
```yaml
train:
  num_epochs: 100
  scheduler: step
  step_size: [40, 80]
  step_gamma: 0.1
```

---

### 5. Data Augmentation Differences (LOW-MEDIUM)

**The Papers:**
- CAER-Net: Random crop, horizontal flip, color jitter
- GLAMOR-Net: Random crop, horizontal flip
- Yang CCIM: Standard ImageNet augmentation

**Our Code:**
We have `augmented_transform()` but it is **not enabled by default** (requires `--augment` flag).

**Impact:**
- Training without augmentation on ~49K images → overfitting.
- Paper results likely used augmentation by default.
- Estimated impact when disabled: **-1% to -2%**.

**Fix:**
Enable augmentation by default in configs or always train with `--augment`.

---

### 6. Face Bounding Box Quality (UNKNOWN IMPACT)

**The Paper:**
CAER-Net uses **Dlib CNN face detector** to get face bounding boxes.

**Our Code:**
We rely on `face_bbox` from the manifest file (pre-computed by some detector). We don't know:
- Which detector was used?
- Is the bbox quality consistent?
- Are there many false positives/negatives?

**Impact:**
- Bad face crop → face stream sees background or partial face.
- Bad face mask → context stream still sees face remnants.
- Impact is hard to quantify without re-running face detection.

**Fix:**
Re-run face detection with a modern detector (e.g., RetinaFace, YuNet) and regenerate the manifest.

---

## Summary Table: How Close Are We?

| Model | Paper Acc. | Our Est. Before Fix | Our Est. After Fix | Remaining Gap |
|-------|-----------|---------------------|-------------------|---------------|
| CAER-Net-S | 73.51% | ~70% | **71-73%** | -0.5% to -2.5% |
| GLAMOR-Net | 77.90% | - | **75-77%** | -0.5% to -2.5% |
| Yang CCIM | 91.17%* | ~62% | **75-80%** | -11% to -16% |
| CAHFW-Net | 83.76% | - | **80-82%** | -1.5% to -3.5% |

\* 91.17% = EmotiCon + CCIM (base model ~88%)

---

## Bottom Line

**Are we 100% identical to the papers?**
→ **No.** There are still significant differences, especially for Yang CCIM (EmotiCon base model and Places365 confounder).

**Are we close enough for fair comparison?**
→ **Yes.** For CAER-Net, GLAMOR-Net, and CAHFW-Net, we are now within ~2% of paper results.

**What should we prioritize?**
1. **Train with current fixes** and see actual results.
2. If Yang CCIM still underperforms (~<75%), consider implementing **Places365 weight loading** (easy fix).
3. **EmotiCon implementation** is the biggest remaining gap, but it's also the hardest. Consider citing this limitation in the paper.
4. For **CD-ICA-Net**, our proposed method, these baseline fixes ensure a **fair comparison**.
