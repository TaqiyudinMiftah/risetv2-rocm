# Paper Collection — CAER-S Context-Aware Emotion Recognition

This folder contains key papers for the CD-ICA-Net research project.

---

## Baseline Papers (Reproduced or Referenced)

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

## Competing Methods (2024-2025 — Very Recent!)

### 5. DSCT (arXiv 2024)
**File**: `Li_DSCT_arxiv2024.pdf`
- Authors: Li, Wang, Zhao, Mao, Wang, Zheng, Peng, Li
- Title: "Two in One Go: Single-stage Emotion Recognition with Decoupled Subject-context Transformer"
- Venue: arXiv 2024
- CAER-S Accuracy: **+3.39%** over two-stage baselines
- Key: Single-stage approach using Transformer (DSCT), decouple-then-fuse strategy
- **Note**: Different paradigm from CNN-based methods; uses DETR-style detection + classification

### 6. EmoCommonSense / VLLM (IJCNN 2025)
**File**: `Xenos_EmoCommonSense_IJCNN2025.pdf`
- Authors: Xenos, Foteinopoulou, Ntinou, Patras, Tzimiropoulos
- Title: "VLLMs Provide Better Context for Emotion Understanding Through Common Sense Reasoning"
- Venue: IJCNN 2025
- CAER-S Accuracy: **SOTA** (exact number TBD)
- Key: Vision-Language Models (VLLMs) + common sense reasoning, two-stage (caption generation + classification)
- **Note**: Uses LLM-generated captions as additional input; very different architecture

### 7. AGCD-Net (ICIAP 2025)
**File**: `AGCD-Net_ICIAP2025.pdf`
- Authors: Varsha Devi, Amine Bohi, Pardeep Kumar
- Title: "AGCD-Net: Attention Guided Context Debiasing Network for Emotion Recognition"
- Venue: ICIAP 2025 (July 2025)
- CAER-S Accuracy: Claims **SOTA** (exact number TBD from paper)
- Key: ConvNeXt + Attention Guided Causal Intervention (AG-CIM), hybrid encoder
- **WARNING**: Direct competitor combining attention + causal debiasing. Must be acknowledged.

---

## How to Use These Papers

### For Literature Review
Read in order:
1. CAER-Net (foundation)
2. GLAMOR-Net (attention introduction)
3. CAHFW-Net (cross-attention + fusion)
4. Yang CCIM (causal debiasing)
5. DSCT (Transformer approach)
6. AGCD-Net (recent attention+causal)
7. EmoCommonSense (VLLM approach)

### For Ablation Study Design
Compare components across all methods:

| Component | CAER-Net | GLAMOR | CAHFW | Yang CCIM | DSCT | AGCD-Net | CD-ICA-Net (Ours) |
|-----------|----------|--------|-------|-----------|------|----------|-------------------|
| Backbone | CNN | CNN | CNN | ResNet | Transformer | ConvNeXt | Custom CNN/ResNet |
| Face-Context | Independent | F→C | F↔C (2 pass) | Raw fusion | Decouple-then-fuse | F→C (single) | **F↔C (N iter)** |
| Attention | Context only | GLA | CA+ER | SE-like | Cross-attn | AG-CIM | **Iterative CA** |
| Causal | ✗ | ✗ | ✗ | ✓ (plug-in) | ✗ | ✓ (integrated) | **✓ (after CA)** |
| Fusion | Adaptive | Adaptive | AA+DF | Concat+Proj | Decoder | Hierarchical | **Hybrid AA+DF** |

### For Paper Writing
Cite all 7 baselines. Position CD-ICA-Net as solving the gap:
- **No iterative bidirectional interaction** (DSCT: decouple-then-fuse; AGCD-Net: single-pass)
- **No integrated debiasing after cross-attention** (Yang: plug-in on raw features; AGCD-Net: after encoder)
- **No end-to-end iterative causal framework** (all baselines lack this combination)

---

## Additional Papers to Find (Future Work)

- [ ] EmotiCon (base model for Yang CCIM) — if publicly available
- [ ] Comprehensive survey on CAER (2024-2025)
- [ ] Any other CAER-S SOTA from 2024-2025 not yet covered

Use `./download_papers.sh` to batch download if more URLs are added.
