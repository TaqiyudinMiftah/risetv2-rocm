# 05 — Training Strategy & Experimental Setup

---

## Strategi Training 3 Fase

Training dilakukan secara bertahap untuk menjamin stabilitas, karena melatih seluruh modul (backbone + iterative CA + causal debiasing + fusion) dari awal secara bersamaan sangat rentan terhadap instabilitas gradien.

---

### Fase 1: Pretraining Backbone CNN

**Tujuan**: Memastikan CNN encoder menghasilkan fitur dangkal yang representatif sebelum modul yang lebih kompleks ditambahkan.

**Yang dilatih**: Dual-branch CNN encoder (Stage 1) saja

**Modul yang di-freeze**: Semua modul selain CNN encoder

**Loss function**:
```
L_fase1 = L_ce (cross-entropy standard)
```

**Durasi**: ~20 epoch atau hingga konvergensi pada validation set

**Learning rate**: 1e-3, cosine annealing

---

### Fase 2: Training ICA Module

**Tujuan**: Melatih modul iterative cross-attention agar dapat menangkap relasi komplementer wajah-konteks secara efektif, tanpa gangguan dari modul causal debiasing.

**Yang dilatih**: Stage 2 (Iterative Bidirectional Cross-Attention + ER blocks)

**Modul yang di-freeze**: Stage 1 (backbone CNN)

**Loss function**:
```
L_fase2 = L_ce + α · L_ica
```

**Catatan**: L_ica pada fase ini dihitung berdasarkan output intermediate (sebelum Stage 3 sepenuhnya aktif), berfungsi sebagai regularisasi awal agar fitur tidak terlalu bias konteks.

**Durasi**: ~30 epoch

**Learning rate**: 5e-4, cosine annealing

---

### Fase 3: End-to-End Finetuning

**Tujuan**: Mengoptimalkan seluruh pipeline secara bersama-sama agar setiap komponen dapat saling mengoptimalkan satu sama lain.

**Yang dilatih**: Semua modul (Stage 1 s/d Stage 5)

**Loss function**:
```
L_fase3 = L_ce + α · L_ica + β · L_reg
```

**Durasi**: ~50 epoch

**Learning rate**: 1e-4, cosine annealing

**Early stopping**: patience = 10 epoch berdasarkan validation accuracy

---

## Detail Implementasi

### Framework & Hardware

```python
# Framework
import torch
import torch.nn as nn
import torchvision

# Hardware (rekomendasi)
# GPU: minimal 1× NVIDIA RTX 3080 (10GB VRAM)
# atau 2× NVIDIA V100 (16GB each)
```

### Preprocessing & Data Augmentation

**Face branch:**
```python
face_transform_train = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

face_transform_test = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Context branch:**
```python
ctx_transform_train = transforms.Compose([
    transforms.Resize((128, 171)),
    transforms.RandomCrop((112, 112)),  # atau RandomResizedCrop(224)
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

ctx_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Optimizer & Scheduler

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=5e-4,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,
    eta_min=1e-6
)
```

### Dropout

- Dropout rate: 0.5 pada DF block dan classifier
- Gradient clipping: max_norm = 1.0 (untuk mencegah exploding gradients pada iterative attention)

---

## Dataset

### 1. CAER-S (Primary benchmark)

| Split | Jumlah Gambar |
|-------|--------------|
| Train | ~49,000 (70%) |
| Val   | ~7,000 (10%) |
| Test  | ~14,000 (20%) |

- Sumber: 13,201 klip video dari 79 TV show
- 7 kelas: angry, disgust, fear, happy, neutral, sad, surprise
- Download: http://caer-dataset.github.io

### 2. EMOTIC (Secondary benchmark)

| Split | Jumlah Gambar |
|-------|--------------|
| Train | ~16,154 (70%) |
| Val   | ~2,310 (10%) |
| Test  | ~4,614 (20%) |

- 26 kategori emosi diskret
- Evaluasi menggunakan mAP (mean Average Precision)
- Catatan: tidak mengandung gambar wajah (face-unvisible), sehingga face branch menggunakan whole-body region

### 3. NCAER-S (Robustness benchmark)

| Split | Jumlah Gambar |
|-------|--------------|
| Train | ~18,555 |
| Val   | - |
| Test  | - |

- Versi CAER-S yang lebih challenging (tidak ada overlap train-test dari video yang sama)
- Download: https://bit.ly/NCAERS_dataset

---

## Evaluation Metrics

| Dataset | Metric |
|---------|--------|
| CAER-S | Classification accuracy (%) |
| NCAER-S | Classification accuracy (%) |
| EMOTIC | mean Average Precision (mAP) |

---

## Baseline Methods untuk Perbandingan

| Model | Venue | CAER-S Acc. |
|-------|-------|------------|
| Fine-tuned AlexNet | - | 61.73% |
| Fine-tuned VGGNet | - | 64.85% |
| Fine-tuned ResNet | - | 68.46% |
| CAER-Net-S | ICCV 2019 | 73.51% |
| GLAMOR-Net (original) | NCA 2022 | 77.90% |
| CAHFW-Net | IJERPH 2023 | 83.76% |
| GLAMOR-Net (ResNet-18) | NCA 2022 | 89.88% |
| EmotiCon + CCIM | CVPR 2023 | 91.17% |
| **CD-ICA-Net (ours)** | - | **TBD** |

---

## Ablation Study yang Direncanakan

### Ablasi Komponen Arsitektur

| ID | Setting | Tujuan |
|----|---------|--------|
| A1 | Backbone CNN saja (tanpa CA, tanpa CCIM) | Baseline murni |
| A2 | + Unidirectional CA (F→C only, 1 pass) | Efek directionality |
| A3 | + Bidirectional CA (F↔C, 1 pass) | Efek bidirectionality |
| A4 | + Iterative Bidirectional CA (N=3) | Efek iterasi |
| A5 | + CCIM (pada fitur mentah, mirip Yang et al.) | CCIM posisi awal |
| A6 | + CCIM terintegrasi setelah ICA (full CD-ICA-Net) | Efek posisi CCIM |

### Ablasi Hyperparameter

| Parameter | Nilai yang Diuji |
|-----------|-----------------|
| N (iterasi CA) | 1, 2, 3, 5 |
| K_conf (ukuran confounder dict) | 64, 128, 256, 512 |
| α (L_ica weight) | 0.1, 0.3, 0.5, 0.7, 1.0 |
