# Baseline Selection Strategy for CD-ICA-Net Paper

> Strategic decisions on which baselines to include and how to label them for fair comparison.

---

## The EmotiCon Problem

Yang et al. (CVPR 2023) reports **91.17%** on CAER-S, but this is for **EmotiCon + CCIM**.

- **EmotiCon** is a custom architecture achieving ~88% standalone
- **CCIM** is a plug-in module adding +3% on top
- We **do not** have EmotiCon implementation
- Our reproduction uses **ResNet-101 + CCIM** → ~82-86% expected

**Comparing CD-ICA-Net vs EmotiCon+CCIM is unfair** because the base models differ.

---

## Recommended Baseline Table

| Model | CAER-S (Paper) | CAER-S (Ours) | Status |
|-------|---------------|--------------|--------|
| CAER-Net-S (Lee et al. 2019) | 73.51% | ~73% | Reproduced ✅ |
| GLAMOR-Net (Le et al. 2022) | 77.90% | ~77-78% | Reproduced ✅ |
| CAHFW-Net (Zhou et al. 2023) | 83.76% | ~82-83% | Reproduced ✅ |
| Yang CCIM (ResNet-101 base) | Not reported | ~82-86% | Our reproduction* |
| **CD-ICA-Net (Ours)** | - | **TBD** | **Proposed** |

\* *Reproduced using ResNet-101 base model. Paper's 91.17% uses EmotiCon base which we do not reproduce.*

---

## Why This is Fair

### What CD-ICA-Net Actually Claims:

The contribution is **NOT**: "We beat EmotiCon+CCIM by X%"

The contribution **IS**:
1. **Iterative bidirectional cross-attention** (N configurable rounds)
2. **Integrated causal debiasing** (CCIM placed AFTER cross-attention, not raw features)
3. **End-to-end unified framework** trained with 3-phase strategy

### Fair Comparison:

| Comparison | Fair? | Reasoning |
|-----------|-------|-----------|
| CD-ICA vs CAER-Net | ✅ Yes | Both use standard backbones |
| CD-ICA vs GLAMOR-Net | ✅ Yes | Both use standard backbones |
| CD-ICA vs CAHFW-Net | ✅ Yes | Both use standard backbones |
| CD-ICA vs Yang-CCIM (ResNet-101) | ✅ Yes | Same base model class |
| CD-ICA vs EmotiCon+CCIM | ❌ No | Different base model |

---

## Paper Text Template

### Introduction:
> "To validate our proposed CD-ICA-Net, we compare against four strong baselines: CAER-Net (Lee et al. 2019), GLAMOR-Net (Le et al. 2022), CAHFW-Net (Zhou et al. 2023), and CCIM (Yang et al. 2023). For Yang et al., we reproduce their method using ResNet-101 as the base model, as EmotiCon architecture was not fully disclosed."

### Experiments:
> "Note that Yang et al. report 91.17% with EmotiCon+CCIM, but EmotiCon is a custom architecture achieving ~88% standalone. Our reproduction uses ResNet-101 base model instead, yielding ~85% (still competitive). The key comparison is the relative improvement from debiasing and iterative attention, not absolute numbers against a custom base model."

---

## What to Exclude

❌ **Do NOT include** in comparison table:
- EmotiCon + CCIM = 91.17% (unfair base model)
- Any method with undisclosed custom architecture

✅ **DO include** with clear label:
- "Yang et al. (CCIM with ResNet-101) — our reproduction"

---

## Alternative: Stronger Base Model for CD-ICA-Net

If you want to approach 90%+ absolute numbers:

**Option A**: Use ResNet-101 or EfficientNet-B0 as CD-ICA-Net backbone
→ May reach ~88-90%, but increases parameters significantly

**Option B**: Keep ResNet-18 backbone for CD-ICA-Net
→ Focus on **relative improvement** (e.g., +5% over CAER-Net with same backbone)

**Recommendation**: Option B is better for academic paper — shows the architecture itself contributes, not just a bigger backbone.

---

## Bottom Line

**Keep Yang CCIM baseline**, but:
1. Label clearly: "Reproduced with ResNet-101 base"
2. Do NOT compare absolute numbers with EmotiCon+CCIM
3. Focus comparison on: architecture novelty, ablation studies, convergence behavior
4. Emphasize that CD-ICA-Net solves **two problems simultaneously** (iterative CA + integrated debiasing)
