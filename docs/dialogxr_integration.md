# DialogXR Integration

This project can be connected to DialogXR as a multimodal healthcare AI capability.

## Platform Framing

DialogXR should not be described as simply "using the model." The stronger enterprise framing is:

> DialogXR provides the secure workflow, orchestration, audit, and human-review layer for multimodal clinical AI. The breast cancer models are specialist services inside that platform.

## Where It Fits

| DialogXR layer | Breast Cancer Multimodal AI role |
| --- | --- |
| Case workspace | Presents mammography, pathology, genomics, and clinical context in one review flow |
| Orchestrator | Routes cases from screening to deeper multimodal assessment |
| Model services | Hosts mammography screening, pathology encoding, genomic feature, and fusion services |
| Evidence panel | Shows model score, cohort context, modality contributions, and missing inputs |
| Guardrail service | Blocks unsupported clinical claims and routes uncertain cases to review |
| Audit service | Records source data, model versions, risk scores, explanations, and reviewer action |
| Evaluation layer | Tracks AUROC, C-index, calibration, subgroup performance, and drift |

## Product Narrative

The product narrative could be:

1. A screening case enters the platform from imaging workflow.
2. Stage 1 mammography model produces a suspiciousness score.
3. If flagged, the case is routed to a deeper diagnostic workup.
4. Pathology, clinical, and genomics signals are aligned for the patient.
5. Stage 2 model estimates progression-risk context.
6. DialogXR shows the result as evidence-backed decision support.
7. Reviewer feedback and final action are logged for governance.

## What Makes This Enterprise-Grade

- Identity and role-based access
- Consent and policy checks
- Data provenance across modalities
- Model registry and versioned inference
- Human-in-the-loop workflow
- Safety thresholds and missing-modality warnings
- Immutable audit trail
- Continuous evaluation and drift monitoring

## Concise Summary

> The research repo proves the modelling pattern. DialogXR is how I would turn that pattern into a governed workflow: secure access, multimodal data connectors, model orchestration, explanation, guardrails, audit, and reviewer feedback.
