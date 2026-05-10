# Multi-Method Emotion Recognition Pipeline (CAER-S)

Repository ini berisi pipeline modular untuk emosi recognition pada dataset **CAER-S**, dengan dukungan untuk **banyak metode** dari berbagai paper.

## Metode yang Tersedia

| Method | Paper | Status |
|--------|-------|--------|
| `caernet` | Lee et al., "Context-Aware Emotion Recognition Networks", ICCV 2019 | Ready |
| `zhou_cross_attention` | Zhou et al., "Emotion Recognition from Large-Scale Video Clips with Cross-Attention and Hybrid Feature Weighting Neural Networks", IJERPH 2023 | Ready |
| `yang_ccim` | Yang et al., "Context De-Confounded Emotion Recognition", CVPR 2023 | Ready |
| `glamor_net` | Le et al., "Global-Local Attention for Emotion Recognition", Neural Computing and Applications, 2022 | Ready |
| `cd_ica_net` | **CD-ICA-Net (Proposed)** — Causal Debiasing Iterative Cross-Attention Network | Ready |

## Struktur Repository (Multi-Method)

```text
.
├── bin/                          # Helper bash scripts
│   ├── setup_uv.sh              # UV environment setup
│   ├── build_manifest.sh        # Build data manifest
│   ├── smoke_test.sh            # Data pipeline smoke test
│   ├── train.sh                 # Unified training script
│   ├── evaluate.sh              # Unified evaluation script
│   └── run_all_models.sh        # Run train/eval for all models
├── configs/
│   ├── caernet.yaml             # CAER-Net config
│   ├── zhou_cross_attention.yaml # Zhou et al. config
│   ├── yang_ccim.yaml           # Yang et al. (CCIM) config
│   └── glamor_net.yaml          # Le et al. (GLAMOR-Net) config
├── scripts/
│   ├── build_caers_manifest.py  # Manifest builder CLI
│   ├── smoke_data_pipeline.py   # Smoke test CLI
│   ├── train.py                 # Unified training CLI (multi-method)
│   └── evaluate.py              # Unified evaluation CLI (multi-method)
├── src/
│   ├── config/                  # Configuration loader
│   │   └── config.py
│   ├── datasets/                # Shared dataset utilities
│   │   ├── caers_dataset.py
│   │   └── transforms.py
│   ├── engine/                  # Shared training/eval engine
│   │   ├── metrics.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── models/                  # All methods/models
│   │   ├── common.py            # Shared encoder builder
│   │   ├── caernet/             # CAER-Net method
│   │   │   └── model.py
│   │   ├── zhou_cross_attention/ # Zhou et al. method
│   │   │   └── model.py
│   │   ├── yang_ccim/           # Yang et al. (CCIM) method
│   │   │   ├── model.py
│   │   │   └── confounder_builder.py
│   │   └── glamor_net/          # Le et al. (GLAMOR-Net) method
│   │       └── model.py
│   └── cd_ica_net/          # CD-ICA-Net (proposed method)
│       ├── model.py
│       ├── ica_module.py
│       ├── ccim_module.py
│       ├── fusion_module.py
│       └── confounder_builder.py
│   └── utils/
│       ├── io_utils.py
│       └── data_manifest.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Menambahkan Metode Baru

Untuk menambahkan metode dari paper baru:

1. **Buat direktori model**: `src/models/<method_name>/`
2. **Implementasikan model**: Buat `model.py` dengan class yang mengimplementasikan `forward(face_image, context_image) -> dict`
3. **Update config loader**: Tambahkan method-specific config di `src/config/config.py`
4. **Buat config file**: `configs/<method_name>.yaml`
5. **Register di scripts**: Update `build_model()` di `scripts/train.py` dan `scripts/evaluate.py`

Contoh minimal model:
```python
class MyNewModel(nn.Module):
    def forward(self, face_image, context_image):
        # ... your architecture ...
        return {"logits": logits}
```

## Setup

### NVIDIA CUDA (default)

```bash
./bin/setup_uv.sh
# atau manual:
#   uv venv --python 3.12
#   source .venv/bin/activate
#   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   uv pip install -e ".[dev]"
```

### AMD ROCm

```bash
# Otomatis mendeteksi ROCm dan install PyTorch ROCm wheel
USE_ROCM=1 ./bin/setup_uv.sh

# atau manual dengan versi ROCm tertentu (default 6.2):
#   uv venv --python 3.12
#   source .venv/bin/activate
#   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
#   uv pip install -e ".[dev]"
```

> **Catatan ROCm:** PyTorch untuk ROCm menggunakan namespace `torch.cuda.*` karena kompatibilitas HIP. Oleh karena itu, kode seperti `torch.cuda.is_available()` dan `torch.device("cuda")` **tetap bekerja** di GPU AMD dengan ROCm. Tidak perlu mengubah config `device` kecuali ingin memaksa device tertentu.

### Verifikasi GPU

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Pipeline Steps

### 1. Build Manifest + Diagnostics

```bash
./bin/build_manifest.sh
# atau dengan config tertentu:
./bin/build_manifest.sh --config configs/zhou_cross_attention.yaml
```

### 2. Smoke Test Data Pipeline

```bash
./bin/smoke_test.sh
```

### 3. Train

```bash
# CAER-Net
./bin/train.sh --config configs/caernet.yaml

# Zhou et al. Cross-Attention
./bin/train.sh --config configs/zhou_cross_attention.yaml

# Yang et al. CCIM (causal intervention)
./bin/train.sh --config configs/yang_ccim.yaml

# Le et al. GLAMOR-Net (global-local attention)
./bin/train.sh --config configs/glamor_net.yaml

# CD-ICA-Net (proposed)
./bin/train.sh --config configs/cd_ica_net.yaml

# Dengan augmentasi
./bin/train.sh --config configs/zhou_cross_attention.yaml --augment

# Custom run name
./bin/train.sh --config configs/caernet.yaml --run-name "caernet_baseline"

# Resume dari checkpoint
./bin/train.sh --config configs/caernet.yaml --resume checkpoints/caernet/best_model.pt

# Training + evaluasi test dalam satu W&B run
./bin/train.sh --config configs/cd_ica_net.yaml --eval-after-train
```

### 4. Evaluate

```bash
# Auto-detect checkpoint berdasarkan method di config
./bin/evaluate.sh --config configs/caernet.yaml

# Evaluasi split val
./bin/evaluate.sh --config configs/zhou_cross_attention.yaml --split val

# Yang et al. CCIM
./bin/evaluate.sh --config configs/yang_ccim.yaml

# Le et al. GLAMOR-Net
./bin/evaluate.sh --config configs/glamor_net.yaml

# CD-ICA-Net (proposed)
./bin/evaluate.sh --config configs/cd_ica_net.yaml

# Custom checkpoint
./bin/evaluate.sh --config configs/caernet.yaml --checkpoint checkpoints/caernet/best_model.pt

# Evaluasi dengan melanjutkan W&B run yang sama (gunakan run ID dari training)
./bin/evaluate.sh --config configs/cd_ica_net.yaml --resume-run-id <RUN_ID> --checkpoint checkpoints/cd_ica_net/best_model.pt
```

### 5. Run All Models (Batch)

Jalankan training atau evaluasi untuk **semua model secara berurutan** dengan satu perintah:

```bash
# Training semua model
./bin/run_all_models.sh --mode train

# Training semua model dengan augmentasi + W&B offline
./bin/run_all_models.sh --mode train --augment --offline

# Evaluasi semua model pada split test
./bin/run_all_models.sh --mode evaluate --split test

# Evaluasi semua model pada split val + offline
./bin/run_all_models.sh --mode evaluate --split val --offline
```

Script ini menjalankan model secara sequential dengan fail-safe: jika satu model gagal, script tetap melanjutkan ke model berikutnya dan menampilkan summary sukses/gagal di akhir.

## Ablation Studies (CAER-Net)

Ubah `train.stream_mode` di `configs/caernet.yaml`:

```yaml
train:
  stream_mode: face      # Face-only baseline
  # stream_mode: context # Context-only baseline
  # stream_mode: multimodal # Full two-stream (default)
```

## Monitoring dengan W&B

Training & evaluation otomatis log ke Weights & Biases:
- Metrics per epoch
- Model checkpoints sebagai artifact
- Per-class accuracy tables & charts

Set environment variables untuk konfigurasi W&B Anda sendiri:
```bash
export WANDB_API_KEY="your_key"
export WANDB_PROJECT="your_project"
```

## Notes

- Dataset CAER-S harus memiliki struktur `train/` dan `test/` dengan subfolder per kelas.
- Semua metode menggunakan **two-stream input** (face + context) kecuali ablasi single-stream.
- Pretrained backbones sangat direkomendasikan karena ukuran CAER-S yang relatif kecil.
- **Yang CCIM**: Confounder dictionary dibangun otomatis dari training data saat pertama kali training. File akan disimpan di `checkpoints/yang_ccim/confounder_dict.pt` dan bisa digunakan kembali untuk resume/evaluasi.
