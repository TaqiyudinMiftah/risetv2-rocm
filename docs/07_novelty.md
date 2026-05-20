# 07 — Kebaruan (Novelty)

> Penjelasan mendalam mengapa CD-ICA-Net adalah kontribusi baru yang signifikan

---

## Posisi CD-ICA-Net dalam Lanskap Penelitian

### Landscape 2024-2025 (Updated)

```
                      Bidirectional CA?
                 No        Partial      Full (iterative)
                 |         |            |
No debiasing → CAER-Net  GLAMOR-Net   CAHFW-Net (2 pass, fixed)
                                    DSCT (decouple-then-fuse)
                                    EmoCommonSense (VLLM)
Plug-in debias →         CCIM (pada fitur mentah)
Integrated     →         AGCD-Net     [CD-ICA-Net] ← POSISI KITA
 debias after CA        (after encoder)
```

CD-ICA-Net mengisi kuadran yang **belum ada** dalam literatur yang ada.

### Perbandingan dengan Semua Baselines (2024-2025)

| Paper | CA Direction | Iterasi | Causal Position | Backbone | CAER-S |
|-------|-------------|---------|----------------|----------|--------|
| CAER-Net (2019) | None | 1 | — | CNN | 73.51% |
| GLAMOR-Net (2022) | F→C | 1 | — | CNN | 89.88%* |
| CAHFW-Net (2023) | F↔C | 2 (fixed) | — | CNN | 83.76% |
| Yang CCIM (2023) | — | — | After encoder (plug-in) | ResNet | 91.17%** |
| **AGCD-Net (2025)** | F→C | 1 | After encoder (integrated) | ConvNeXt | SOTA |
| DSCT (2024) | Decouple-then-fuse | Layer-wise | — | Transformer | +3.39% |
| EmoCommonSense (2025) | — | — | — | VLLM | SOTA |
| **CD-ICA-Net** | **F↔C** | **N (configurable)** | **After iterative CA** | Custom/ResNet | **TBD** |

\* ResNet-18 variant  \*\* EmotiCon + CCIM

---

## Novelty 1 — Iterative Bidirectional Cross-Attention (N > 2)

### Apa yang Baru

CD-ICA-Net adalah **yang pertama** mengusulkan mekanisme *iterative bidirectional cross-attention* untuk CAER, di mana fitur wajah dan konteks saling memperbarui satu sama lain secara berulang selama N iterasi yang dapat dikonfigurasi.

### Perbandingan dengan Paper Sebelumnya

| Paper | Directionality | Iterasi |
|-------|---------------|---------|
| CAER-Net (2019) | Unidirectional (context only) | 1 pass |
| GLAMOR-Net (2022) | F → C only (face guides context) | 1 pass |
| CAHFW-Net (2023) | F ↔ C (bidirectional) | 2 pass (fixed) |
| **CD-ICA-Net** | **F ↔ C (bidirectional)** | **N pass (configurable)** |

### Mengapa Ini Penting

Relasi wajah-konteks bersifat rekursif. Analoginya adalah cara kognitif manusia memproses emosi:
1. Pertama kita melihat ekspresi wajah → kesan awal
2. Kita lihat konteks → memperbaiki pemahaman wajah
3. Kembali ke wajah dengan pemahaman konteks yang baru → kesan lebih akurat
4. Dan seterusnya...

Setiap iterasi menghasilkan representasi yang lebih kaya karena:
- Wajah yang sudah "memahami" konteks memberikan key yang lebih informatif untuk memperbarui konteks
- Konteks yang sudah "memahami" wajah memberikan key yang lebih informatif untuk memperbarui wajah

### Cara Mengklaim di Paper

> *"To the best of our knowledge, we are the first to propose an iterative bidirectional cross-attention mechanism for CAER, where face and context representations mutually refine each other across N configurable rounds of interaction, enabling progressively richer complementary feature learning."*

---

## Novelty 2 — Causal Debiasing Terintegrasi Setelah Cross-Attention Stage

### Apa yang Baru

CCIM (Yang et al. 2023) diletakkan pada fitur mentah setelah encoder. CD-ICA-Net **memindahkan posisi debiasing** ke setelah iterative cross-attention, dan mengintegrasikannya secara end-to-end dalam satu arsitektur.

### Perbandingan dengan CCIM (Yang et al. 2023)

| Aspek | CCIM (Yang et al.) | CD-ICA-Net |
|-------|-------------------|------------|
| Posisi CCIM | Setelah encoder (fitur mentah) | Setelah N iterasi cross-attention |
| Input ke CCIM | Shallow features | Features yang sudah diperkaya oleh interaksi wajah-konteks |
| Integrasi | Plug-in (external, model-agnostic) | End-to-end (internal, co-optimized) |
| Training | Terpisah dari backbone | Bersama backbone (3 fase) |

### Mengapa Posisi CCIM Penting

**Argumen utama**: Konfounder (bias konteks) lebih efektif diidentifikasi dan dieliminasi dari representasi yang sudah mengandung informasi relasional yang dalam antara wajah dan konteks.

**Intuisi**: Bayangkan dua skenario:
- Skenario A: Kamu diminta menghilangkan bias "taman = bahagia" dari gambar mentah (kamu belum tahu siapa orang di taman itu dan bagaimana ekspresinya)
- Skenario B: Kamu diminta menghilangkan bias yang sama setelah sudah memahami relasi antara ekspresi wajah dan konteks taman

Skenario B jauh lebih mudah karena kamu sudah memiliki konteks yang lebih kaya untuk membedakan "orang bahagia di taman" vs "orang tidak bahagia di taman yang kebetulan diambil fotonya".

### Cara Mengklaim di Paper

> *"Unlike Yang et al. [ref] where CCIM is applied as a plug-in on raw encoder features, we integrate causal debiasing directly after the cross-attention stage, operating on representations already enriched by iterative face-context interaction. We argue that confounders are more effectively identified and eliminated from semantically richer representations."*

---

## Novelty 3 — Unified End-to-End Training Framework

### Apa yang Baru

CD-ICA-Net adalah **satu kesatuan arsitektur** yang dilatih secara end-to-end, bukan gabungan modul-modul terpisah. Semua komponen (backbone, ICA, CCIM, fusion, classifier) dioptimalkan bersama melalui strategi training 3 fase yang terstruktur.

### Perbandingan dengan Pendekatan Sebelumnya

| Paper | Pendekatan Integrasi |
|-------|---------------------|
| CAER-Net + CCIM | CCIM di-attach ke CAER-Net yang sudah dilatih |
| CAHFW-Net + CCIM | Tidak ada (CAHFW-Net tidak menggunakan CCIM) |
| **CD-ICA-Net** | **Satu arsitektur, dilatih end-to-end dari awal** |

### Keuntungan End-to-End Training

1. **Co-optimization**: Setiap komponen dapat saling mengoptimalkan berdasarkan gradien dari komponen lain
2. **Konsistensi representasi**: Fitur yang dihasilkan backbone sudah "tahu" bahwa ia akan digunakan oleh ICA dan CCIM
3. **Tidak ada mismatch**: Tidak ada distribusi shift antara fitur yang dihasilkan model pre-trained dan yang diharapkan oleh modul eksternal

---

## Ringkasan Kontribusi untuk Paper

Berikut adalah formulasi kontribusi yang siap digunakan di bagian Introduction:

```
The main contributions of this paper are as follows:

1. We identify that existing CAER methods suffer from two simultaneous 
   limitations: insufficient face-context interaction and unaddressed 
   context bias, yet no prior work has resolved both within a single 
   unified framework.

2. We propose an Iterative Bidirectional Cross-Attention (ICA) module 
   that performs N configurable rounds of bidirectional interaction 
   between face and context representations, enabling progressively 
   richer complementary feature learning beyond single-pass approaches.

3. We integrate the Contextual Causal Intervention Module (CCIM) 
   directly after the cross-attention stage — operating on 
   interaction-enriched features rather than raw encoder outputs — 
   and demonstrate that this positioning leads to more effective 
   context debiasing.

4. We unify all components into a single end-to-end trainable 
   framework (CD-ICA-Net) with a structured three-phase training 
   strategy, and conduct extensive experiments on CAER-S, EMOTIC, 
   and NCAER-S to validate each contribution.
```
