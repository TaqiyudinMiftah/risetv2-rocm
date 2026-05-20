# CD-ICA-Net Research Project

> **Causal Debiasing Iterative Cross-Attention Network for Context-Aware Emotion Recognition**

## Deskripsi Singkat

Riset ini mengusulkan arsitektur baru bernama **CD-ICA-Net** untuk menyelesaikan dua masalah fundamental yang belum diselesaikan secara bersamaan dalam task *Context-Aware Emotion Recognition* (CAER):

1. Interaksi wajah-konteks yang masih dangkal, satu arah, dan tidak iteratif pada metode-metode sebelumnya
2. *Context bias* dalam dataset yang mendorong model belajar korelasi semu antara konteks tertentu dengan kategori emosi tertentu

## Struktur Folder

```
cd-ica-net-research/
├── README.md                  ← dokumen ini (entry point untuk AI assistant)
├── 01_literature_review.md    ← ringkasan 4 paper yang dieksplorasi
├── 02_research_gap.md         ← analisis gap dan justifikasi riset
├── 03_architecture.md         ← detail arsitektur CD-ICA-Net lengkap
├── 04_mathematics.md          ← formulas matematis seluruh komponen
├── 05_training_strategy.md    ← strategi training 3 fase + loss function
├── 06_abstract.md             ← abstrak paper (versi konferensi IEEE)
└── 07_novelty.md              ← penjelasan kebaruan vs paper sebelumnya
```

## Quick Summary untuk AI Assistant

- **Task**: Context-Aware Emotion Recognition (CAER) dari gambar/video
- **Dataset target**: CAER-S, EMOTIC, NCAER-S
- **Framework**: PyTorch
- **Backbone**: CNN encoder (ResNet-18 atau EfficientNet-B0)
- **Jumlah kelas emosi**: 7 (angry, disgust, fear, happy, neutral, sad, surprise)
- **Target publikasi**: Konferensi internasional IEEE
- **Status**: Proposal / rencana riset (belum ada hasil eksperimen)

## Komponen Utama Arsitektur

| Stage | Nama | Fungsi |
|-------|------|--------|
| 1 | Dual-Branch CNN Encoder | Ekstraksi fitur dangkal wajah & konteks |
| 2 | Iterative Bidirectional Cross-Attention (ICA) | Interaksi mutual wajah↔konteks sebanyak N iterasi |
| 3 | Integrated CCIM (Causal Debiasing) | Eliminasi context bias via backdoor adjustment |
| 4 | Hybrid Adaptive Fusion | Fusi fitur dangkal & dalam dengan bobot adaptif |
| 5 | Emotion Classifier | Softmax 7-kelas |

## Cara Membaca Dokumen Ini

Untuk memahami konteks riset secara menyeluruh, baca file dalam urutan:
`01` → `02` → `03` → `04` → `05` → `06` → `07`

Untuk langsung ke implementasi, baca: `03_architecture.md` dan `04_mathematics.md`
