# LinkedIn Post Draft

I have been building a two-stage multimodal AI project for breast cancer screening and prognosis.

The goal was not to create another toy demo. I wanted to understand what an enterprise-grade medical AI pipeline looks like when it has to work across imaging, pathology, genomics, clinical variables, evaluation, and workflow.

Project: **Breast Cancer Multimodal AI**

What I built:

- Stage 1 mammography screening over VinDr-Mammo
- Four-view ConvNeXt-Base model for exam-level suspicion scoring
- Stage 2 TCGA-BRCA prognosis pipeline
- Pathology foundation model benchmarking with CONCH, UNI2, and CTransPath
- Vision + clinical + genomics fusion for Progression-Free Interval risk modelling
- Isambard GPU workflows for feature extraction and training
- Streamlit research demo for the two-stage workflow
- Enterprise reference architecture with guardrails, audit, monitoring, and human review

Headline results:

- Stage 1 mammography AUROC: `0.7407`
- Stage 2 best model: CONCH + Vision + Clinical + Genomics
- Stage 2 C-index: `0.6093 +/- 0.0441`
- Median split log-rank: `p = 0.005`
- 5-year time-dependent AUC: `0.6123`

The biggest lesson: multimodal AI is not just joining different features together. The real work is in data alignment, provenance, missing-modality handling, evaluation design, and workflow safety.

For a real deployment, I would frame this as clinical decision support rather than autonomous diagnosis. The architecture needs identity, consent, model registry, guardrails, immutable audit, monitoring, and a human reviewer workflow.

This project helped me connect applied ML research with the kind of enterprise AI architecture companies actually need.

#MultimodalAI #MedicalAI #HealthcareAI #MachineLearning #GenAI #ComputerVision #PathologyAI #MLOps #AWS #ResearchEngineering

## Shorter Version

I built a two-stage breast cancer multimodal AI project connecting mammography screening with pathology, genomics, and clinical risk modelling.

Key results:

- Mammography screening AUROC: `0.7407`
- Best TCGA-BRCA prognosis model: CONCH + Vision + Clinical + Genomics
- C-index: `0.6093 +/- 0.0441`
- Median split log-rank: `p = 0.005`

The main learning was architectural: multimodal AI is not only model fusion. It requires data provenance, patient-level alignment, evaluation, guardrails, audit, monitoring, and human review.

I also mapped the research system to an enterprise healthcare architecture, including a DialogXR-style case-review platform and an AWS managed-service deployment pattern.
