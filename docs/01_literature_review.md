# 01 — Literature Review

> Ringkasan 4 paper yang menjadi landasan riset CD-ICA-Net

---

## Paper 1: CAER-Net (ICCV 2019)

**Judul**: Context-Aware Emotion Recognition Networks  
**Penulis**: Lee, Kim, Kim, Park, Sohn  
**Institusi**: Yonsei University, EPFL  
**Venue**: ICCV 2019

### Kontribusi Utama
- Memperkenalkan framework CAER-Net dengan arsitektur **two-stream**: face encoding stream dan context encoding stream
- Mengusulkan strategi **face-hiding** pada context stream untuk memaksa jaringan belajar konteks di luar wajah
- Memperkenalkan dataset **CAER** (13.201 klip video dari 79 TV show, 7 kategori emosi) dan **CAER-S** (±70K gambar statis)
- Menggunakan **attention inference module** secara unsupervised pada context stream
- Menggunakan **adaptive fusion network** dengan bobot λ_F dan λ_C untuk menggabungkan fitur wajah dan konteks

### Arsitektur
- **Face encoding stream**: 3D-CNN (5 conv layers, 3×3×3 kernels, BN + ReLU + max-pool)
- **Context encoding stream**: 3D-CNN + attention inference module
- **Adaptive fusion**: softmax-weighted concatenation (λ_F + λ_C = 1)
- Model statis: **CAER-Net-S** (2D-CNN, untuk gambar)
- Model dinamis: **CAER-Net** (3D-CNN, untuk video)

### Hasil Eksperimen
- CAER-Net: **77.04%** akurasi pada dataset CAER
- CAER-Net-S: **73.51%** akurasi pada dataset CAER-S
- Peningkatan +3.53% model dinamis vs statis

### Keterbatasan
- Attention hanya bersifat **satu arah**: konteks yang mendapat attention, wajah tidak memperbarui dirinya dari konteks
- Tidak menangani **context bias** dalam dataset
- Fusi fitur menggunakan concatenation sederhana

---

## Paper 2: CCIM — Context De-confounded Emotion Recognition (CVPR 2023)

**Judul**: Context De-confounded Emotion Recognition  
**Penulis**: Yang, Chen, Wang, et al.  
**Institusi**: Fudan University  
**Venue**: CVPR 2023

### Kontribusi Utama
- **Pertama** mengidentifikasi *context bias* sebagai confounder dalam dataset CAER menggunakan perspektif causal inference
- Membuktikan bias melalui eksperimen conditional entropy pada EMOTIC dan CAER-S
- Mengusulkan **Contextual Causal Intervention Module (CCIM)** berbasis *backdoor adjustment*
- CCIM bersifat **plug-in dan model-agnostic** (dapat dimasukkan ke model CAER manapun)

### Causal Graph
Variabel dalam CAER causal graph:
- **X**: input images
- **S**: subject features
- **C**: context features
- **Z**: confounder (context bias)
- **Y**: predictions

Jalur berbahaya: `X ← Z → C → Y` (backdoor path)  
Tujuan: memblokir jalur ini dengan do-calculus

### Formula Kunci
```
# Konvensional (biased):
P(Y|X) = Σ_z P(Y|X, S=f_s(X), C=f_c(X,z)) P(z|X)

# Causal intervention (debiased):
P(Y|do(X)) = Σ_z P(Y|X, S=f_s(X), C=f_c(X,z)) P(z)
```

### Implementasi CCIM
- **Confounder Dictionary Z** = [z1...zN]: dibangun via K-Means++ dari fitur konteks (masking subjek)
- Backbone untuk ekstraksi: ResNet-152 pretrained pada Places365
- Aproksimasi menggunakan **NWGM** (Normalized Weighted Geometric Mean)
- Ukuran N: 256 (EMOTIC), 128 (CAER-S), 256 (GroupWalk)

### Hasil Eksperimen
- EMOTIC: +2.95% s/d +3.85% mAP pada berbagai baseline
- CAER-S: +1.31% s/d +2.52% akurasi
- GroupWalk: +2.25% s/d +3.73% mAP
- EmotiCon + CCIM: **91.17%** pada CAER-S (SOTA)

### Keterbatasan
- CCIM bekerja pada **fitur mentah** sebelum interaksi wajah-konteks
- Bersifat plug-in, tidak end-to-end terintegrasi dengan backbone
- Tidak memanfaatkan komplementaritas wajah-konteks sebelum debiasing

---

## Paper 3: GLAMOR-Net (Neural Computing and Applications, 2022)

**Judul**: Global-Local Attention for Emotion Recognition  
**Penulis**: Le, Nguyen, Nguyen, Le  
**Institusi**: University of Science Ho Chi Minh City, University of Liverpool  
**Venue**: Neural Computing and Applications (2022)

### Kontribusi Utama
- Mengusulkan **Global-Local Attention (GLA) Module** yang menggunakan fitur wajah (lokal) untuk memandu attention pada konteks (global)
- Memperkenalkan dataset baru **NCAER-S** yang menghilangkan korelasi antara frame training dan testing
- Menunjukkan bahwa wajah dan konteks memiliki hubungan korelatif yang dapat dieksploitasi bersama

### Arsitektur GLAMOR-Net
1. **Facial Encoding Module**: 5 conv layers (32→64→128→256→256 filters)
2. **Context Encoding Module**: sama dengan face, input adalah gambar dengan wajah di-mask (pixel=0)
3. **Global-Local Attention (GLA) Module**:
   - Wajah di-reduce ke vektor v_f via global pooling
   - Untuk setiap cell (i,j) di context feature map: concatenate [v_f, v_{i,j}]
   - Feed-forward network menghasilkan attention score s_{i,j}
   - Softmax → attention map a_{i,j}
   - Output: v_c = Σ_{i,j} (a_{i,j} · v_{i,j})
4. **Fusion Module**: kompetitif weighting antara face branch dan context branch

### Rumus GLA
```
# Attention score per region
s_{i,j} = FC([v_f ; v_{i,j}])

# Normalized attention map
a_{i,j} = exp(s_{i,j}) / Σ_{a,b} exp(s_{a,b})

# Weighted context representation
v_c = Σ_i Σ_j (a_{i,j} ⊙ v_{i,j})
```

### Hasil Eksperimen
- GLAMOR-Net (ResNet-18): **89.88%** pada CAER-S (SOTA saat itu)
- GLAMOR-Net (original): 77.90% pada CAER-S (+4.38% vs CAER-Net-S)
- NCAER-S: 46.91% (baseline CAER-Net-S: 44.14%)

### Keterbatasan
- Interaksi hanya **F → C** (wajah memandu konteks, tidak sebaliknya)
- Tidak ada mekanisme **C → F** (konteks tidak memperbarui representasi wajah)
- Tidak menangani context bias

---

## Paper 4: CAHFW-Net (IJERPH, 2023)

**Judul**: Emotion Recognition from Large-Scale Video Clips with Cross-Attention and Hybrid Feature Weighting Neural Networks  
**Penulis**: Zhou, Wu, Jiang, Huang, Huang  
**Institusi**: Zhejiang Normal University  
**Venue**: International Journal of Environmental Research and Public Health (2023)

### Kontribusi Utama
- Mengidentifikasi kelemahan **simple concatenation** sebagai strategi fusi yang mengabaikan interaksi antar fitur
- Mengusulkan **Cross-Attention (CA) block** untuk menangkap informasi komplementer antar fitur wajah dan konteks secara *cross-channel*
- Mengusulkan **Element Recalibration (ER) block** untuk menyematkan informasi global ke peta fitur
- Mengusulkan **Adaptive-Attention (AA) block** dengan *hybrid feature weighting* (shallow + deep)
- Mengusulkan **Deep Fusion (DF) block** untuk fusi hierarkis dan padat

### Arsitektur CAHFW-Net
```
Input → DBE Network → HAE Network → DF Block → Classifier

DBE Network:
  - TE block (face): 5× Conv2D + BN + ReLU + MaxPool → X̄_F
  - CE block (context): 5× Conv2D + attention highlight module → X̂_C

HAE Network (2 I-R pairs):
  I-R Pair 1:
    CA block: Q_F·K_C^T/√d · V_F → Z^CA_F
    ER block: GIE + element-wise recalibration → R^ER_F
  I-R Pair 2:
    CA block: Q_C·K_{Z_F}^T/√d · V_C → Z^CA_C
    ER block: → R^ER_C
  AA block: hybrid feature weighting (λ_shallow + λ_deep) → 4 adaptive features

DF Block: hierarchical concatenation → emotion classification
```

### Formula CA Block
```
Y^CA_F = softmax(Q_F · K_C^T / √C_D) · V_F  ∈ R^{C×D}
Z^CA_F = δ(B(Conv2D(Reshape(Y^CA_F)))) ∈ R^{C×H×W}
```

### Formula ER Block
```
G_F = softmax(Z_F · W^TM_F · Z_F^T) ⊗ Z_F  ∈ R^{C×(H×W)}
R^ER_F = Reshape(Z_F ⊙ G_F)  ∈ R^{C×H×W}
```

### Hasil Eksperimen
- CAHFW-Net: **83.76%** pada CAER-S
- Peningkatan +10.25% vs CAER-Net-S
- Happy accuracy: +17.35%, Neutral accuracy: +12.35%

### Keterbatasan
- Cross-attention hanya dilakukan **2 kali** (tidak iteratif dengan N yang dapat dikonfigurasi)
- **Tidak menangani context bias** sama sekali
- Tidak ada mekanisme kausal

---

---

## Paper 5: AGCD-Net (ICIAP 2025)

**Judul**: AGCD-Net: Attention Guided Context Debiasing Network for Emotion Recognition  
**Penulis**: Varsha Devi, Amine Bohi, Pardeep Kumar  
**Institusi**: —  
**Venue**: ICIAP 2025 (July 2025)

### Kontribusi Utama
- Mengidentifikasi **context bias** sebagai masalah serius dalam CAER (sama dengan Yang et al.)
- Mengusulkan **Hybrid ConvNeXt** sebagai backbone baru yang mengintegrasikan Spatial Transformer Network dan Squeeze-and-Excitation layers
- Mengusulkan **Attention Guided - Causal Intervention Module (AG-CIM)** yang menggabungkan causal theory dengan attention guidance dari face features
- Mengusulkan **perturbasi konteks** untuk mengisolasi spurious correlations

### Arsitektur AGCD-Net
1. **Hybrid ConvNeXt Encoder**: ConvNeXt + STN + SE layers
2. **Attention Guided - Causal Intervention Module (AG-CIM)**:
   - Perturb context features untuk isolasi spurious correlations
   - Attention-driven correction yang dipandu oleh face features
   - Backdoor adjustment dengan confounder dictionary
3. **Fusion**: Hierarchical feature fusion

### Hasil Eksperimen
- CAER-S: **SOTA** (klaim, exact number TBD)
- Menunjukkan bahwa causal debiasing penting untuk robust emotion recognition

### Keterbatasan
- Interaksi face-context hanya **satu arah** (face memandu konteks, tidak iteratif)
- Cross-attention tidak **bidirectional iteratif**
- Causal intervention dilakukan pada level encoder (bukan setelah cross-attention)

---

## Paper 6: DSCT (arXiv 2024)

**Judul**: Two in One Go: Single-stage Emotion Recognition with Decoupled Subject-context Transformer  
**Penulis**: Li, Wang, Zhao, Mao, Wang, Zheng, Peng, Li  
**Institusi**: —  
**Venue**: arXiv 2024

### Kontribusi Utama
- Mengusulkan **single-stage** emotion recognition (vs two-stage: detect then classify)
- Menggunakan **Decoupled Subject-Context Transformer (DSCT)** untuk simultaneous subject localization dan emotion classification
- Joint supervision: box detection + emotion classification
- **Decouple-then-fuse** strategy untuk fine-grained subject-context interaction

### Arsitektur DSCT
1. **Query tokens**: subject queries + context queries (decoupled)
2. **DSCT layers**: subject dan context queries secara gradual saling "intertwine"
3. **Spatial and semantic relations**: di-exploit dan di-aggregate
4. **Single-stage decoder**: menghasilkan bounding box + emotion label

### Hasil Eksperimen
- CAER-S: **+3.39%** accuracy improvement vs two-stage alternatives
- EMOTIC: **+6.46%** average precision gain
- Fewer parameters than two-stage methods

### Keterbatasan
- Menggunakan **Transformer**, bukan CNN (beda paradigm)
- Interaksi subject-context adalah **decouple-then-fuse**, bukan iterative bidirectional
- Tidak ada mekanisme **causal debiasing**
- Single-stage mungkin tidak cocok untuk semua aplikasi

---

## Paper 7: EmoCommonSense / VLLM (IJCNN 2025)

**Judul**: VLLMs Provide Better Context for Emotion Understanding Through Common Sense Reasoning  
**Penulis**: Xenos, Foteinopoulou, Ntinou, Patras, Tzimiropoulos  
**Institusi**: —  
**Venue**: IJCNN 2025

### Kontribusi Utama
- Menggunakan **Vision-Large-Language Models (VLLMs)** untuk generate natural language descriptions of emotion in context
- **Two-stage approach**: (1) generate captions with VLLM, (2) fuse text+visual features untuk classification
- Leverages **common sense reasoning** dari LLM untuk constrain noisy visual input
- Simplifies training process dibandingkan explicit scene-encoding architectures

### Arsitektur
1. **Stage 1**: Prompt VLLM untuk generate emotion descriptions
   - Input: image
   - Output: natural language description (e.g., "The person looks happy in the garden")
2. **Stage 2**: Transformer-based fusion
   - Visual features + text features → fused representation → classification

### Hasil Eksperimen
- **SOTA** pada CAER-S, EMOTIC, dan BoLD
- Significantly outperforms individual modalities
- No bells and whistles (simple yet effective)

### Keterbatasan
- **Requires LLM inference** (computationally expensive)
- **Two-stage pipeline** (not end-to-end)
- **Caption quality** depends on VLLM capabilities
- Tidak generalizable ke dataset tanpa caption annotations

---

## Tabel Perbandingan Paper (Updated)

| Aspek | CAER-Net | GLAMOR-Net | CCIM | CAHFW-Net | AGCD-Net | DSCT | EmoCommonSense | CD-ICA-Net |
|-------|----------|------------|------|-----------|----------|------|----------------|------------|
| Bidirectional CA | — | Partial (F→C) | — | Ya (2 pass) | Partial (F→C) | Decouple-then-fuse | — | **Full iteratif (N)** |
| Iterative (N>2) | — | — | — | — | — | — | — | **✅ Ya (configurable)** |
| Causal debiasing | — | — | Ya (plug-in) | — | Ya (after encoder) | — | — | **✅ Ya (after CA)** |
| End-to-end integrated | — | — | — | — | — | — | — | **✅ Ya** |
| Backbone | CNN | CNN | ResNet | CNN | ConvNeXt | Transformer | VLLM | Custom CNN/ResNet |
| Dataset CAER-S acc. | 73.51% | 89.88% | 91.17%* | 83.76% | SOTA | +3.39%** | SOTA | **TBD** |

\*EmotiCon + CCIM  \*\*vs two-stage baselines
