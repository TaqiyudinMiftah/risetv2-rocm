# 02 — Research Gap Analysis

> Analisis kesenjangan riset yang menjadi justifikasi CD-ICA-Net

---

## Gap yang Diidentifikasi

### Gap 1 — Causal Debiasing Belum Diintegrasikan dengan Cross-Attention (FOKUS UTAMA)

**Masalah**: Yang et al. (CVPR 2023) membuktikan bahwa context bias adalah confounder serius dalam dataset CAER. Namun CCIM bekerja sebagai plug-in pada fitur mentah, sebelum model memiliki kesempatan memahami relasi wajah-konteks secara mendalam. Di sisi lain, GLAMOR-Net dan CAHFW-Net yang mengeksploitasi interaksi wajah-konteks sama sekali tidak menangani bias ini.

**Gap**: Belum ada metode yang secara terintegrasi menggabungkan causal debiasing dengan cross-attention wajah-konteks dalam satu arsitektur end-to-end. Debiasing seharusnya dilakukan pada representasi yang sudah diperkaya oleh interaksi wajah-konteks, bukan pada fitur mentah.

**Justifikasi**: Konfounder (bias konteks) lebih efektif diidentifikasi dan dieliminasi ketika representasi sudah mengandung informasi relasional yang dalam antara wajah dan konteks.

---

### Gap 2 — Interaksi Wajah-Konteks Masih Satu Arah atau Tidak Iteratif

**Masalah**:
- CAER-Net: attention satu arah pada konteks saja
- GLAMOR-Net: wajah memandu konteks (F→C), namun C→F tidak ada
- CAHFW-Net: dua arah (F↔C) namun hanya 2 kali, tidak iteratif

**Gap**: Belum ada eksplorasi *iterative bidirectional co-attention* di mana kedua representasi saling memperbarui secara berulang (N iterasi, N dapat dikonfigurasi). Relasi wajah-konteks bersifat rekursif: memahami konteks lebih dalam membantu memahami wajah lebih baik, dan sebaliknya.

**Justifikasi**: Analoginya adalah cara manusia memproses emosi secara kognitif — kita bolak-balik antara wajah dan konteks beberapa kali sebelum membuat penilaian emosi yang akurat.

---

### Gap 3 — Semua Metode Masih Berbasis CNN Murni

**Masalah**: Keempat paper seluruhnya menggunakan CNN sebagai backbone. CAHFW-Net menyebut potensi ViT namun tidak mengimplementasikannya. CNN memiliki keterbatasan dalam menangkap long-range dependency antar region wajah dan konteks yang berjauhan secara spasial.

**Gap**: Eksplorasi backbone hybrid CNN-Transformer untuk CAER, terutama dalam menangkap relasi spasial jarak jauh.

**Catatan**: Gap ini bersifat opsional untuk riset ini karena menambah kompleksitas. Prioritas utama adalah Gap 1 dan Gap 2.

---

### Gap 4 — Generalisasi ke Dunia Nyata Masih Lemah

**Masalah**: Semua metode diuji terutama pada CAER-S yang bersumber dari TV show. GLAMOR-Net menunjukkan performa turun drastis dari ~89% (CAER-S) ke ~46% (NCAER-S) ketika korelasi train-test dihilangkan.

**Gap**: Metode yang lebih generalisatif untuk in-the-wild emotion recognition.

**Catatan**: Akan divalidasi dengan menggunakan tiga dataset sekaligus (CAER-S, EMOTIC, NCAER-S).

---

## Fokus Riset yang Dipilih

**Kombinasi Gap 1 + Gap 2** adalah yang paling feasible dan memiliki novelty tinggi:

> "Arsitektur pengenalan emosi berbasis context-aware yang mengintegrasikan causal debiasing dengan mekanisme iterative bidirectional cross-attention dalam satu framework end-to-end."

### Alasan Pemilihan

1. **Novelty jelas**: Belum ada paper yang melakukan dua hal ini sekaligus dalam satu arsitektur
2. **Dataset tersedia**: CAER-S, EMOTIC, NCAER-S sudah ada dan bisa diakses
3. **Baseline ada**: CAER-Net, GLAMOR-Net, CAHFW-Net, CCIM bisa direproduksi untuk perbandingan
4. **Kompleksitas terkontrol**: Tidak memerlukan dataset baru atau hardware ekstrem
5. **Justifikasi teoritis kuat**: Ada dasar dari causal inference (Yang et al.) dan cross-attention (CAHFW-Net)

---

## Posisi Riset dalam Landscape

```
                    Bidirectional CA?
                    No          Yes (partial)    Yes (full, iterative)
                    |           |                |
No debiasing  →  CAER-Net    GLAMOR-Net       CAHFW-Net
                                               (2 pass only)
Plug-in debias →             CCIM
                             (on raw features)
Integrated    →                               [CD-ICA-Net ← posisi kita]
debias after CA
```

CD-ICA-Net mengisi kuadran yang kosong: **full iterative bidirectional CA + integrated causal debiasing**.
