# Paper Collection — CAER-S Context-Aware Emotion Recognition

This folder contains key papers for the CD-ICA-Net research project.

---

## Baseline Papers (Reproduced in this repo)

### 1. CAER-Net (ICCV 2019)
**File**: `Lee_CAER-Net_ICCV2019.pdf`
- Authors: Lee, Kim, Kim, Park, Sohn
- Title: "Context-Aware Emotion Recognition Networks"
- Venue: ICCV 2019
- CAER-S Accuracy: **73.51%**
- Key: Two-stream CNN + adaptive fusion, introduced CAER-S dataset

### 2. GLAMOR-Net (Neural Computing and Applications, 2022)
**File**: `Le_GLAMOR-Net_arxiv.pdf`
- Authors: Le, Nguyen, Nguyen, Le
- Title: "Global-Local Attention for Emotion Recognition"
- Venue: Neural Computing and Applications (2022)
- CAER-S Accuracy: **77.90%** (original), **89.88%** (ResNet-18)
- Key: Global-Local Attention (GLA) module, introduced NCAER-S

### 3. CAHFW-Net (IJERPH 2023)
**File**: `Zhou_CAHFW_arxiv.pdf`
- Authors: Zhou, Wu, Jiang, Huang, Huang
- Title: "Emotion Recognition from Large-Scale Video Clips with Cross-Attention and Hybrid Feature Weighting Neural Networks"
- Venue: International Journal of Environmental Research and Public Health (2023)
- CAER-S Accuracy: **83.76%**
- Key: Cross-Attention (CA) + Element Recalibration (ER) + Adaptive-Attention (AA) + Deep Fusion (DF)

### 4. Yang CCIM (CVPR 2023)
**File**: `Yang_CCIM_CVPR2023.pdf`
- Authors: Yang, Chen, Wang, et al.
- Title: "Context De-Confounded Emotion Recognition"
- Venue: CVPR 2023
- CAER-S Accuracy: **91.17%** (EmotiCon + CCIM)
- Key: First causal debiasing (CCIM) for CAER, backdoor adjustment, K-Means++ confounder dict

---

## Competing Method (2025 — Very Recent!)

### 5. AGCD-Net (ICIAP 2025)
**File**: `AGCD-Net_ICIAP2025.pdf`
- Authors: Varsha Devi, Amine Bohi, Pardeep Kumar
- Title: "AGCD-Net: Attention Guided Context Debiasing Network for Emotion Recognition"
- Venue: ICIAP 2025 (July 2025)
- CAER-S Accuracy: Claims **SOTA** (exact number TBD from paper)
- Key: ConvNeXt + Attention Guided Causal Intervention (AG-CIM), hybrid encoder
- **WARNING**: This is a direct competitor published July 2025! Must be acknowledged in our paper.

---

## How to Use These Papers

### For Literature Review
Read in order: CAER-Net → GLAMOR-Net → CAHFW-Net → Yang CCIM → AGCD-Net

### For Ablation Study Design
Compare components:
- CAER-Net: basic two-stream
- GLAMOR-Net: + global-local attention
- CAHFW-Net: + cross-attention + recalibration
- Yang CCIM: + causal debiasing
- AGCD-Net: + attention-guided causal intervention
- CD-ICA-Net (ours): + iterative bidirectional cross-attention + integrated causal debiasing

### For Paper Writing
Cite all 5 baselines. Position CD-ICA-Net as solving the gap that all 5 baselines still have:
- No iterative bidirectional interaction (AGCD-Net: single-pass)
- No integrated debiasing after cross-attention (all baselines)

---

## Additional Papers to Download (Recommended)

- [ ] EmotiCon (base model for Yang CCIM) — if available
- [ ] Survey on Context-Aware Emotion Recognition (2024/2025)
- [ ] Any other CAER-S SOTA from 2024-2025

Use `./download_papers.sh` to batch download if more URLs are added.
