# CD-ICA-Net Research — AI Agent Guide

> Research project: **Causal Debiasing Iterative Cross-Attention Network for Context-Aware Emotion Recognition**
> Target venue: IEEE international conference

## Research Direction (Primary Source)

See `docs/` for full research documentation. Read in this order:
1. `docs/README.md` — project overview and quick summary
2. `docs/01_literature_review.md` — 4 baseline papers (CAER-Net, CCIM, GLAMOR-Net, CAHFW-Net)
3. `docs/02_research_gap.md` — gap analysis justifying CD-ICA-Net
4. `docs/03_architecture.md` — full architecture specification (5 stages)
5. `docs/04_mathematics.md` — complete mathematical formulations
6. `docs/05_training_strategy.md` — 3-phase training strategy, datasets, metrics
7. `docs/06_abstract.md` — conference abstract
8. `docs/07_novelty.md` — novelty claims vs prior work

### Core Goal
Implement **CD-ICA-Net**, a new architecture that simultaneously solves:
1. **Shallow/non-iterative face-context interaction** → via Iterative Bidirectional Cross-Attention (ICA)
2. **Context bias** → via Integrated CCIM (causal debiasing) placed *after* cross-attention, operating on enriched representations

## Current Codebase Status

This repo currently contains **baseline reproductions** for comparison:
- `caernet` — CAER-Net (ICCV 2019)
- `zhou_cross_attention` — CAHFW-Net (IJERPH 2023)
- `yang_ccim` — Context De-confounded Emotion Recognition (CVPR 2023)
- `glamor_net` — GLAMOR-Net (Neural Computing and Applications, 2022)

The final goal is to add `cd_ica_net` as the proposed method alongside these baselines.

## Architecture to Implement (5 Stages)

Refer to `docs/03_architecture.md` and `docs/04_mathematics.md` for full detail.

| Stage | Component | Key Notes |
|-------|-----------|-----------|
| 1 | Dual-Branch CNN Encoder | 5 conv blocks each branch; context gets attention-based highlight module |
| 2 | Iterative Bidirectional Cross-Attention (ICA) | N configurable rounds (default N=3); each round has CA F→C, CA C→F, ER blocks on both |
| 3 | Integrated Causal Debiasing (CCIM) | Confounder dict Z built offline (K-Means++, K=128 for CAER-S); backdoor adjustment on enriched features |
| 4 | Hybrid Adaptive Fusion (AA + DF blocks) | Joint weighting of shallow and deep features |
| 5 | Emotion Classifier | FC → Softmax over 7 emotion classes |

### Loss Function
```
L_total = L_ce + α · L_ica + β · L_reg
```
- `L_ce`: cross-entropy with flooding (α_flood = 0.05)
- `L_ica`: KL(P(Y|X) || P(Y|do(X))) — causal intervention loss
- `L_reg`: L2 on W_h and W_g

### Training Strategy (3 Phases)
See `docs/05_training_strategy.md`:
1. **Phase 1**: Pretrain backbone CNN only (~20 epochs, lr=1e-3)
2. **Phase 2**: Train ICA module only (~30 epochs, lr=5e-4)
3. **Phase 3**: End-to-end finetuning all modules (~50 epochs, lr=1e-4, early stopping patience=10)

## Project Structure

```
.
├── bin/                    # Bash helper scripts
│   ├── setup_uv.sh
│   ├── build_manifest.sh
│   ├── smoke_test.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── run_all_models.sh   # Batch run all models (train or eval)
├── configs/
│   ├── caernet.yaml
│   ├── zhou_cross_attention.yaml
│   ├── yang_ccim.yaml
│   ├── glamor_net.yaml
│   └── cd_ica_net.yaml     # TODO: add this
├── scripts/
│   ├── build_caers_manifest.py
│   ├── smoke_data_pipeline.py
│   ├── train.py            # Unified training CLI (multi-method)
│   └── evaluate.py         # Unified evaluation CLI (multi-method)
├── src/
│   ├── config/
│   │   └── config.py       # Config loader — add cd_ica_net here
│   ├── datasets/
│   │   ├── caers_dataset.py
│   │   └── transforms.py
│   ├── engine/
│   │   ├── metrics.py
│   │   ├── trainer.py      # May need 3-phase training logic
│   │   └── evaluator.py
│   ├── models/
│   │   ├── common.py       # Shared encoder builder
│   │   ├── caernet/
│   │   ├── zhou_cross_attention/
│   │   ├── yang_ccim/
│   │   ├── glamor_net/
│   │   └── cd_ica_net/     # TODO: add this directory
│   │       ├── model.py    # Main CD-ICA-Net model
│   │       ├── ica_module.py        # Iterative Bidirectional Cross-Attention
│   │       ├── ccim_module.py       # Integrated causal debiasing
│   │       ├── fusion_module.py     # Hybrid Adaptive Fusion (AA + DF)
│   │       └── confounder_builder.py # Offline K-Means++ confounder dict
│   └── utils/
│       ├── io_utils.py
│       └── data_manifest.py
├── docs/                   # Research documentation (DO NOT MODIFY without permission)
├── checkpoints/
├── logs/
├── pyproject.toml
└── requirements.txt
```

## Development Environment

- **Python**: 3.12 (managed via `uv`)
- **Framework**: PyTorch
- **GPU**: CUDA (default) or ROCm (auto-detected; RX 6600/gfx1032 workaround applied in scripts)
- **Virtual env**: `.venv/` (managed by `uv`)

### Setup
```bash
./bin/setup_uv.sh
# Or for AMD ROCm:
USE_ROCM=1 ./bin/setup_uv.sh
```

### Verify GPU
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Coding Conventions

- Follow existing code style in `src/` and `scripts/`.
- All models must implement `forward(face_image, context_image) -> dict` returning at least `{"logits": logits}`.
- Use type hints where possible.
- Keep modules in `src/models/<method_name>/` with `model.py` as the entry point.
- Register new methods in:
  - `src/config/config.py` (config loader)
  - `scripts/train.py` (`build_model()`)
  - `scripts/evaluate.py` (`build_model()`)
- Config files go in `configs/<method_name>.yaml`.
- Use `src/models/common.py` for shared encoder building if applicable.

## Datasets

| Dataset | Type | Metric | Notes |
|---------|------|--------|-------|
| CAER-S | Primary | Accuracy | 7 classes; ~49K train, ~7K val, ~14K test |
| EMOTIC | Secondary | mAP | 26 discrete categories; face-unvisible cases use whole-body |
| NCAER-S | Robustness | Accuracy | Challenging; no train-test video overlap |

## Important Notes for Implementation

1. **CCIM Position**: Unlike Yang et al. (CVPR 2023) where CCIM is plug-in on raw features, CD-ICA-Net places CCIM **after** iterative cross-attention. This is the key novelty claim.
2. **Confounder Dictionary**: Build offline once from training data via K-Means++. Cache to `checkpoints/cd_ica_net/confounder_dict.pt`.
3. **Iterative CA**: N is configurable (default 3). Need ablation support for N ∈ {1,2,3,5}.
4. **Gradient Clipping**: Use max_norm=1.0 during ICA training to prevent exploding gradients.
5. **Flooding**: Implement `L_ce` with flooding level α_flood = 0.05.
6. **W&B Logging**: All training/eval scripts automatically log to Weights & Biases. Set `WANDB_API_KEY` and `WANDB_PROJECT` env vars if needed.

## How to Add CD-ICA-Net

1. Create `src/models/cd_ica_net/` with model components.
2. Implement `model.py` with `forward(face_image, context_image)`.
3. Create `configs/cd_ica_net.yaml`.
4. Register in `src/config/config.py`, `scripts/train.py`, `scripts/evaluate.py`.
5. Add bash scripts to `bin/` if method-specific helpers are needed.
6. Update `README.md` to include `cd_ica_net` in method tables.

## Running Experiments

```bash
# Build manifest
./bin/build_manifest.sh

# Smoke test
./bin/smoke_test.sh

# Train single model
./bin/train.sh --config configs/cd_ica_net.yaml

# Evaluate single model
./bin/evaluate.sh --config configs/cd_ica_net.yaml

# Train all baselines + proposed (once cd_ica_net is ready)
./bin/run_all_models.sh --mode train
```

## Ablation Studies to Plan

See `docs/05_training_strategy.md` for full list. Key ones:
- Component ablation (backbone only → +unidirectional CA → +bidirectional CA → +iterative CA → +CCIM raw → +CCIM integrated)
- Hyperparameter ablation: N ∈ {1,2,3,5}, K_conf ∈ {64,128,256,512}, α ∈ {0.1,0.3,0.5,0.7,1.0}

## Contact / Maintainer

Researcher: Taqiyudin Miftah
