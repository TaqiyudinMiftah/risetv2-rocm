# 03 — Arsitektur CD-ICA-Net

> Detail lengkap arsitektur Causal Debiasing Iterative Cross-Attention Network

---

## Overview Arsitektur

```
INPUT
  ├── Face image (96×96, cropped via Dlib face detector)
  └── Context image (224×224, wajah di-mask → pixel = 0)
         ↓
STAGE 1: Dual-Branch CNN Encoder
  ├── CNN_face → F_face ∈ R^{C×H×W}  (shallow face features)
  └── CNN_ctx  → F_ctx  ∈ R^{C×H×W}  (shallow context features)
         ↓
STAGE 2: Iterative Bidirectional Cross-Attention (N=3)
  Inisialisasi: H_face^(0) = F_face, H_ctx^(0) = F_ctx
  For n = 1 to N:
    ├── Cross-Attention F→C: H_face diperbarui oleh H_ctx
    ├── Cross-Attention C→F: H_ctx diperbarui oleh H_face
    ├── ER Block (face): embed global info
    └── ER Block (ctx):  embed global info
  Output: H_face* = H_face^(N), H_ctx* = H_ctx^(N)
         ↓
STAGE 3: Integrated Causal Debiasing (CCIM)
  ├── Confounder Dictionary Z (dibangun offline via K-Means++)
  ├── Backdoor adjustment pada H_face* dan H_ctx*
  └── Output: representasi debiased R_face, R_ctx
         ↓
STAGE 4: Hybrid Adaptive Fusion
  ├── Shallow weights λ_s (dari F_face, F_ctx)
  ├── Deep weights λ_d (dari R_face, R_ctx)
  ├── 4 adaptive features via AA block
  └── Hierarchical concatenation via DF block → X_fusion
         ↓
STAGE 5: Emotion Classifier
  └── FC → Softmax → 7 kelas emosi
```

---

## Stage 1: Dual-Branch CNN Encoder

### Tujuan
Mengekstraksi representasi dangkal (shallow features) dari gambar wajah dan konteks secara paralel dan independen.

### Spesifikasi
Kedua branch (face dan context) menggunakan arsitektur CNN yang sama:

| Layer | Filters | Kernel | Activation | Pooling |
|-------|---------|--------|-----------|---------|
| Conv1 | 32 | 3×3 | BN + ReLU | MaxPool 2×2 |
| Conv2 | 64 | 3×3 | BN + ReLU | MaxPool 2×2 |
| Conv3 | 128 | 3×3 | BN + ReLU | MaxPool 2×2 |
| Conv4 | 256 | 3×3 | BN + ReLU | MaxPool 2×2 |
| Conv5 | 256 | 3×3 | BN + ReLU | — |

Output shape: `C=256, H=7, W=7` (untuk input 224×224)

### Context Encoding Tambahan
Pada akhir context branch, ditambahkan **attention-based highlight module**:
- Input: X̄_C ∈ R^{C×H×W}
- 2× Conv2D → softmax → attention map A ∈ R^{H×W}
- Output: X̂_C = A ⊙ X̄_C (element-wise)

### Preprocessing
- **Face**: deteksi dengan Dlib CNN face detector → crop → resize ke 96×96 → resize ke 224×224
- **Context**: mask region wajah (set pixel = 0 di dalam bounding box) → resize ke 224×224

---

## Stage 2: Iterative Bidirectional Cross-Attention

### Tujuan
Membangun representasi yang saling memperkaya antara wajah dan konteks melalui N iterasi interaksi dua arah.

### Prinsip Desain
Berbeda dari CAHFW-Net yang melakukan 2 pass fixed, CD-ICA-Net menggunakan N iterasi yang dapat dikonfigurasi. Setiap iterasi terdiri dari:
1. Cross-Attention F→C (wajah memperbarui dirinya dengan melihat konteks)
2. Cross-Attention C→F (konteks memperbarui dirinya dengan melihat wajah)
3. Element Recalibration pada kedua branch

### Komponen: Cross-Attention Block (CA Block)

**Tujuan**: Menangkap informasi komplementer antara dua feature map melalui operasi cross-channel.

**Operasi** (untuk CA F→C pada iterasi ke-n):
```
Q_f = W_Q · H_face^(n)   ∈ R^{C×D}  # query dari wajah
K_c = W_K · H_ctx^(n)    ∈ R^{C×D}  # key dari konteks
V_f = W_V · H_face^(n)   ∈ R^{C×D}  # value dari wajah

A_fc = softmax(Q_f · K_c^T / √d)    ∈ R^{C×C}  # attention matrix
Y_fc = A_fc · V_f                    ∈ R^{C×D}

Z^CA_face = δ(BN(Conv2D(Reshape(Y_fc))))  ∈ R^{C×H×W}
```

Di mana `D = H × W` (spasial dimension yang di-flatten).

**Untuk CA C→F** pada iterasi yang sama:
```
Q_c = W_Q · H_ctx^(n)
K_f = W_K · H_face^(n)  # key dari hasil CA F→C (Z^CA_face)
V_c = W_V · H_ctx^(n)

A_cf = softmax(Q_c · K_f^T / √d)
Z^CA_ctx = δ(BN(Conv2D(Reshape(A_cf · V_c))))
```

### Komponen: Element Recalibration Block (ER Block)

**Tujuan**: Menyematkan informasi global ke seluruh peta fitur untuk menonjolkan elemen yang relevan secara emosional dan menekan yang tidak.

**Operasi**:
```
# Flatten Z^CA_face → Z_F ∈ R^{C×(H×W)}
G_F = softmax(Z_F · W^TM_F · Z_F^T) ⊗ Z_F   # Gram-like global info
R^ER_face = Reshape(Z_F ⊙ G_F)   ∈ R^{C×H×W}
```

### Alur Iterasi Lengkap

```python
# Pseudo-code iterative cross-attention
H_face = F_face  # inisialisasi dari Stage 1
H_ctx  = F_ctx   # inisialisasi dari Stage 1

for n in range(N):  # N = 3 (default)
    # Cross-attention F → C
    Z_face = cross_attention(query=H_face, key=H_ctx, value=H_face)
    Z_face = er_block(Z_face)  # element recalibration
    
    # Cross-attention C → F
    Z_ctx  = cross_attention(query=H_ctx, key=Z_face, value=H_ctx)
    Z_ctx  = er_block(Z_ctx)   # element recalibration
    
    # Update untuk iterasi berikutnya
    H_face = Z_face
    H_ctx  = Z_ctx

H_face_star = H_face  # representasi wajah yang sudah diperkaya
H_ctx_star  = H_ctx   # representasi konteks yang sudah diperkaya
```

---

## Stage 3: Integrated Causal Debiasing

### Tujuan
Mengeliminasi efek context bias sebagai confounder, sehingga model belajar dari efek kausal sejati P(Y|do(X)) bukan korelasi semu P(Y|X).

### Perbedaan dengan CCIM (Yang et al. 2023)
| Aspek | CCIM (Yang et al.) | CD-ICA-Net |
|-------|-------------------|------------|
| Posisi dalam pipeline | Setelah encoder (fitur mentah) | Setelah iterative cross-attention |
| Input ke CCIM | Fitur shallow | Fitur yang sudah diperkaya oleh N iterasi cross-attention |
| Integrasi | Plug-in (external) | End-to-end (internal) |

### Membangun Confounder Dictionary Z (Offline)

```
1. Untuk setiap gambar training, mask subjek berdasarkan bounding box
2. Ekstrak fitur konteks menggunakan ResNet-152 pretrained pada Places365
3. Jalankan K-Means++ pada seluruh fitur konteks → K cluster
4. Z = [z_1, z_2, ..., z_K] di mana z_i = rata-rata fitur cluster ke-i
5. P(z_i) = N_i / N_total  (proporsi cluster dalam training data)
```

Default K: 128 untuk CAER-S, 256 untuk EMOTIC dan GroupWalk.

### Backdoor Adjustment

```
h = φ(concat(GAP(H_face_star), GAP(H_ctx_star)))  # joint representation

# Attention weight untuk setiap prototype
λ_i = softmax((W_q · h)^T · (W_k · z_i) / √d)

# Weighted sum of confounders
E_z[g(z)] = Σ_i  λ_i · z_i · P(z_i)

# Final causal prediction
P(Y|do(X)) = W_h · h + W_g · E_z[g(z)]
```

---

## Stage 4: Hybrid Adaptive Fusion

### Tujuan
Menggabungkan fitur dari berbagai level abstraksi (shallow + deep) dengan bobot yang dipelajari secara adaptif.

### AA Block (Adaptive-Attention Block)

```
# Shallow features (dari Stage 1)
xe_F = GAP(F_face)  ∈ R^C
xe_C = GAP(F_ctx)   ∈ R^C

# Deep features (dari Stage 3, setelah debiasing)
re_F = GAP(R_face)  ∈ R^C
re_C = GAP(R_ctx)   ∈ R^C

# Shallow weights
λ_shallow = softmax(concat(Conv1D(Conv1D(xe_F)), Conv1D(Conv1D(xe_C))))  ∈ R^2

# Deep weights
λ_deep = softmax(concat(Conv1D(Conv1D(re_F)), Conv1D(Conv1D(re_C))))  ∈ R^2

# 4 adaptive features
f0_shallow = λ_deep[0] ⊙ xe_C
f1_shallow = λ_deep[1] ⊙ xe_F
f0_deep    = λ_shallow[0] ⊙ re_C
f1_deep    = λ_shallow[1] ⊙ re_F
```

### DF Block (Deep Fusion Block)

```
f1 = Dropout(δ(Conv1D(concat(f0_shallow, f0_deep))))
f2 = Dropout(δ(Conv1D(concat(f1_shallow, f1_deep))))

X_fusion = concat(f1, f2)
x_cls    = softmax(Conv1D(Dropout(δ(Conv1D(X_fusion)))))
p        = argmax(x_cls)  # predicted emotion label
```

---

## Stage 5: Emotion Classifier

**Input**: X_fusion dari DF Block  
**Output**: Distribusi probabilitas 7 kelas emosi  
**Kelas**: angry, disgust, fear, happy, neutral, sad, surprise

---

## Hyperparameter Default

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| N (iterasi CA) | 3 | Perlu ablasi: 1, 2, 3, 5 |
| K (ukuran confounder dict) | 128 (CAER-S), 256 (EMOTIC) | Perlu ablasi: 64, 128, 256, 512 |
| C (channel dim) | 256 | Output CNN encoder |
| d (attention dim) | 256 | Dimension Q/K/V |
| α (weight L_ica) | 0.5 | Perlu grid search |
| β (weight L_reg) | 0.1 | Perlu grid search |
| Dropout rate | 0.5 | Standard |
| Batch size | 32 | |
| Optimizer | SGD + Nesterov momentum | |
| Learning rate | 5e-4 → cosine annealing | |
| Flooding level | 0.05 | Mencegah overfitting |

---

## Perbandingan Parameter dengan Metode Lain

| Model | #Params (est.) | Akurasi CAER-S |
|-------|----------------|----------------|
| CAER-Net-S | ~5M | 73.51% |
| GLAMOR-Net (original) | 2.23M | 77.90% |
| GLAMOR-Net (ResNet-18) | 22.90M | 89.88% |
| CAHFW-Net | ~8M | 83.76% |
| **CD-ICA-Net (est.)** | **~15-20M** | **TBD** |
