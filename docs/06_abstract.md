# 06 — Abstract (Conference Version)

> Target: Konferensi internasional IEEE

---

## Abstract

Context-Aware Emotion Recognition (CAER) has emerged as a critical research direction in affective computing, driven by the limitation of facial-expression-only approaches in handling ambiguous or weakly expressed emotions in unconstrained environments. While recent methods have made notable progress through dual-stream architectures and attention mechanisms, two fundamental challenges remain unresolved: the ineffective exploitation of complementary interactions between facial and contextual features, and the pervasive context bias embedded in existing datasets that misleads models into learning spurious correlations between specific contextual scenes and emotion categories. To address these limitations simultaneously, this paper proposes CD-ICA-Net, a novel Context Debiasing Iterative Cross-Attention Network for context-aware emotion recognition. The proposed framework consists of three core components. First, a dual-branch CNN encoder extracts shallow representations from facial and contextual image streams independently. Second, an Iterative Bidirectional Cross-Attention (ICA) module performs N iterative rounds of bidirectional cross-channel interaction, enabling facial features and contextual features to mutually refine one another through query-key-value attention operations, capturing deeper complementary information than single-pass approaches. Third, an integrated Contextual Causal Intervention Module (CCIM) is embedded directly after the cross-attention stage, employing backdoor adjustment and a confounder prototype dictionary constructed via K-Means++ clustering to approximate the true causal effect P(Y|do(X)), effectively disentangling the model from harmful context bias. The refined debiased features are subsequently fused through a hybrid adaptive attention block that jointly weighs shallow and deep representations for final emotion classification. The proposed model is trained in three progressive phases to ensure training stability across all modules. Extensive experiments are planned on three benchmark datasets, namely CAER-S, EMOTIC, and NCAER-S, to evaluate the effectiveness of each proposed component and demonstrate the superiority of CD-ICA-Net over state-of-the-art context-aware emotion recognition methods.

---

## Keywords

context-aware emotion recognition, cross-attention, causal inference, context debiasing, iterative feature interaction, backdoor adjustment, affective computing
