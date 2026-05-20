# 04 — Formulas Matematis

> Seluruh formulas matematis CD-ICA-Net secara lengkap dan sistematis

---

## Notasi

| Simbol | Keterangan |
|--------|------------|
| I_F | Input gambar wajah (face-cropped) |
| I_C | Input gambar konteks (face-masked) |
| N | Jumlah gambar dalam batch |
| C | Jumlah channel feature map |
| H, W | Tinggi dan lebar spatial feature map |
| D = H×W | Dimensi spatial yang di-flatten |
| K | Jumlah kelas emosi (= 7) |
| n | Indeks iterasi cross-attention |
| N_iter | Total iterasi cross-attention (default = 3) |
| Z | Confounder dictionary [z_1, ..., z_K_conf] |
| K_conf | Ukuran confounder dictionary |

---

## Stage 1: Dual-Branch CNN Encoder

### Face Encoder

```
F_face = CNN_f(I_F; W_f)   ∈ R^{C×H×W}
```

Di mana `CNN_f` adalah rangkaian 5 convolutional block:

```
F^{k+1} = MaxPool(δ(BN(Conv2D(F^k))))   untuk k = 0, 1, 2, 3
F^5     = δ(BN(Conv2D(F^4)))             (tanpa max-pool di layer terakhir)
F_face  = F^5
```

### Context Encoder

```
X̄_C = CNN_c(I_C; W_c)   ∈ R^{C×H×W}
```

Dengan attention-based highlight module:

```
A = σ(F^2_AH(F^1_AH(X̄_C)))   ∈ R^{H×W}
F_ctx = X̂_C = A ⊙ X̄_C         ∈ R^{C×H×W}
```

Di mana `F^i_AH(·) = δ(BN(Conv2D(·)))` dan σ adalah softmax spasial.

---

## Stage 2: Iterative Bidirectional Cross-Attention

### Inisialisasi

```
H_face^(0) = F_face
H_ctx^(0)  = F_ctx
```

### Satu Iterasi ke-n (Cross-Attention F→C)

**Langkah 1: Proyeksi Q, K, V**
```
Q_f = Flatten(Conv2D(H_face^(n)))   ∈ R^{C×D}
K_c = Flatten(Conv2D(H_ctx^(n)))    ∈ R^{C×D}
V_f = Flatten(Conv2D(H_face^(n)))   ∈ R^{C×D}
```

**Langkah 2: Scaled dot-product attention**
```
A_fc = softmax(Q_f · K_c^T / √C_D)   ∈ R^{C×C}
```

Di mana `C_D` adalah faktor normalisasi (dimensi attention head).

**Langkah 3: Attention output**
```
Y^CA_face = A_fc · V_f   ∈ R^{C×D}
Z^CA_face = δ(BN(Conv2D(Reshape(Y^CA_face))))   ∈ R^{C×H×W}
```

### Satu Iterasi ke-n (Cross-Attention C→F)

```
Q_c  = Flatten(Conv2D(H_ctx^(n)))
K_zf = Flatten(Conv2D(Z^CA_face))   # key dari output CA F→C
V_c  = Flatten(Conv2D(H_ctx^(n)))

A_cf = softmax(Q_c · K_zf^T / √C_D)   ∈ R^{C×C}
Y^CA_ctx = A_cf · V_c
Z^CA_ctx = δ(BN(Conv2D(Reshape(Y^CA_ctx))))   ∈ R^{C×H×W}
```

### Element Recalibration (ER Block)

Untuk feature map Z^CA ∈ R^{C×H×W}:

```
Z_flat = Reshape(Z^CA)   ∈ R^{C×(H×W)}    # flatten spatial

# Gram-like global information extraction
G = softmax(Z_flat · W^TM · Z_flat^T) ⊗ Z_flat   ∈ R^{C×(H×W)}

# Element-wise recalibration
R^ER = Reshape(Z_flat ⊙ G)   ∈ R^{C×H×W}
```

Di mana `W^TM ∈ R^{(H×W)×(H×W)}` adalah transformation matrix (Conv1D layer).

### Update State Iterasi

```
H_face^(n) = R^ER_face   (output ER block pada face branch)
H_ctx^(n)  = R^ER_ctx    (output ER block pada context branch)
```

### Output Stage 2

Setelah N_iter iterasi:
```
H_face* = H_face^(N_iter)
H_ctx*  = H_ctx^(N_iter)
```

---

## Stage 3: Causal Debiasing

### Membangun Confounder Dictionary (Offline)

```
# Untuk setiap gambar training i:
I^masked_i = mask(I_i, bbox_face_i)   # mask wajah
m_i = ResNet152_Places365(I^masked_i)  # ekstrak fitur konteks

# Clustering seluruh fitur konteks
Z, labels = KMeansPlusPlus(M={m_i}, K=K_conf)
z_j = mean({m_i : label_i = j})   untuk j = 1, ..., K_conf

# Prior probability setiap cluster
P(z_j) = N_j / N_total
```

### Backdoor Adjustment

**Joint representation dari Stage 2:**
```
h = φ(concat(GAP(H_face*), GAP(H_ctx*)))   ∈ R^{d_h}
```

Di mana `GAP` adalah Global Average Pooling dan `φ` adalah operasi fusi (concatenation + linear).

**Attention weight per prototype:**

*Dot-product attention:*
```
λ_i = softmax((W_q · h)^T · (W_k · z_i) / √d)
```

*Additive attention (alternatif):*
```
λ_i = softmax(W_t^T · tanh(W_q · h + W_k · z_i))
```

**Weighted confounder expectation:**
```
E_z[g(z)] = Σ_{i=1}^{K_conf} λ_i · z_i · P(z_i)
```

**Causal prediction:**
```
P(Y|do(X)) = W_h · h + W_g · E_z[g(z)]
```

Di mana `W_h ∈ R^{d_m×d_h}` dan `W_g ∈ R^{d_m×d}` adalah parameter yang dipelajari.

---

## Stage 4: Hybrid Adaptive Fusion

### AA Block (Adaptive-Attention)

**Input features:**
```
xe_F = GAP(F_face)   ∈ R^C   # shallow face (dari Stage 1)
xe_C = GAP(F_ctx)    ∈ R^C   # shallow context (dari Stage 1)
re_F = GAP(R_face)   ∈ R^C   # deep debiased face (dari Stage 3)
re_C = GAP(R_ctx)    ∈ R^C   # deep debiased context (dari Stage 3)
```

**Fusion weights:**
```
λ_shallow = σ(concat(C1(C1(xe_F)), C1(C1(xe_C))))   ∈ R^2
λ_deep    = σ(concat(C1(C1(re_F)), C1(C1(re_C))))   ∈ R^2
```

Di mana `C1(·)` adalah Conv1D layer dan σ adalah softmax.

**4 Adaptive features:**
```
f0_shallow = λ_deep[0]    ⊙ xe_C
f1_shallow = λ_deep[1]    ⊙ xe_F
f0_deep    = λ_shallow[0] ⊙ re_C
f1_deep    = λ_shallow[1] ⊙ re_F
```

### DF Block (Deep Fusion)

```
f1 = Dropout(δ(C1(concat(f0_shallow, f0_deep))))
f2 = Dropout(δ(C1(concat(f1_shallow, f1_deep))))

X_fusion = concat(f1, f2)
x_cls    = σ(C1(Dropout(δ(C1(X_fusion)))))
p        = argmax(x_cls)
```

---

## Loss Function

### Loss Total

```
L_total = L_ce + α · L_ica + β · L_reg
```

### Cross-Entropy Loss (L_ce)

```
L_ce = -(1/N) Σ_{i=1}^{N} Σ_{j=1}^{K} y_{ij} · log(p_{ij}) + α_flood
```

Di mana `α_flood = 0.05` adalah flooding level untuk mencegah overfitting (dari Ishida et al. 2020).

### Causal Intervention Loss (L_ica)

```
L_ica = KL(P(Y|X) || P(Y|do(X)))
      = Σ_y P(y|X) · log(P(y|X) / P(y|do(X)))
```

Tujuan: mendorong distribusi prediksi konvensional mendekati distribusi prediksi kausal.

### Regularization Loss (L_reg)

```
L_reg = ||W_h||^2 + ||W_g||^2
```

Regularisasi L2 pada parameter transformasi kausal untuk mencegah overfitting.

### Hyperparameter Loss

| Parameter | Nilai Default | Range Pencarian |
|-----------|--------------|-----------------|
| α (L_ica weight) | 0.5 | {0.1, 0.3, 0.5, 0.7, 1.0} |
| β (L_reg weight) | 0.1 | {0.01, 0.05, 0.1, 0.5} |
| α_flood | 0.05 | {0.01, 0.05, 0.1} |
