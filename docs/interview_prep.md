# Interview Prep

## Thirty-Second Pitch

I built a two-stage breast cancer multimodal AI research system. Stage 1 performs mammography screening using a four-view ConvNeXt-Base model on VinDr-Mammo. Stage 2 performs prognosis-oriented risk modelling on TCGA-BRCA by fusing pathology foundation-model features, clinical variables, and genomic features. The strongest Stage 2 configuration used CONCH with V+C+G cross-attention and achieved a C-index of `0.6093 +/- 0.0441`, with significant risk stratification by Kaplan-Meier analysis.

## Two-Minute Architecture Walkthrough

The system has two layers. The first layer is a screening model over four mammography views: L-CC, R-CC, L-MLO, and R-MLO. It produces an exam-level suspicion score. The second layer is a deeper multimodal pathway for prognosis, where pathology slide features, clinical covariates, and RNA-seq-derived genomic features are aligned at patient level and fused.

For pathology, I benchmarked foundation encoders including CONCH, UNI2, and CTransPath. I used five-fold cross-validation and evaluated C-index, time-dependent AUC, and risk stratification. The final architecture was not just a model; it included preprocessing, feature extraction, Slurm workflows on Isambard, evaluation scripts, Streamlit demo views, and report artifacts.

## Results to Quote

| Area | Result |
| --- | --- |
| Stage 1 mammography | AUROC `0.7407` on VinDr-Mammo test set |
| Stage 1 confidence interval | AUROC 95 percent CI `0.6492-0.8254` |
| Stage 2 best configuration | CONCH + Vision + Clinical + Genomics |
| Stage 2 C-index | `0.6093 +/- 0.0441` |
| Stage 2 median split log-rank | `p = 0.0050` |
| Stage 2 5-year time-dependent AUC | `0.6123` |

## Why This Is Multimodal

The project combines several medical data types:

- Mammography images for population-level screening
- Whole-slide pathology images for tissue representation
- Genomics features from TCGA-BRCA
- Clinical covariates
- Optional literature and verifier agents in the broader codebase

The important point is that multimodal AI is not just concatenating features. The hard parts are cohort alignment, missing data handling, preprocessing differences, model-version tracking, and evaluation across clinically meaningful endpoints.

## How It Connects to DialogXR

DialogXR can become the enterprise case-review and orchestration layer. The research models become services. DialogXR would handle identity, case routing, patient context, multimodal data connectors, reviewer UI, guardrails, audit logs, and feedback loops.

Strong interview framing:

> My model work sits inside a larger clinical AI architecture. I would not claim autonomous diagnosis. I would expose the model as decision support with provenance, confidence, missing-modality warnings, guardrails, and a human reviewer workflow.

## Questions You May Get

**Why two stages?**

Because screening and diagnostic/prognostic assessment are different clinical problems. Mammography screening works at population scale, while pathology/genomics/clinical fusion is used after deeper workup. The two-stage architecture is closer to how real healthcare workflows are organized.

**Why use foundation models for pathology?**

Whole-slide images are huge and expensive to train from scratch. Foundation encoders provide strong tile or slide representations that can be benchmarked under the same downstream survival pipeline.

**Why did CONCH win?**

In the final benchmark, CONCH with V+C+G cross-attention produced the strongest C-index. I would describe this as an empirical result on the aligned TCGA-BRCA setup, not a universal claim that CONCH is always best.

**Is the performance clinically deployable?**

No, not by itself. It is a research-grade benchmark with promising but modest signal. Deployment would require prospective validation, calibration, subgroup safety checks, data governance, and clinical oversight.

**What was the hardest engineering part?**

Patient-level alignment and reproducible evaluation. It is easy to produce a demo from unlinked modalities; it is much harder to keep slides, clinical rows, genomic features, folds, endpoints, and evaluation artifacts consistent.

**Why not just use a large VLM end to end?**

For medical imaging, domain-specific encoders and validated evaluation are still essential. A general VLM can help with explanation or report summarisation, but the core imaging and survival modelling need careful medical ML pipelines.

**How would you productionize it?**

Use a secure platform: imaging store, clinical data integration, feature extraction workers, model registry, model endpoints, policy guardrails, human review UI, immutable audit, and monitoring for drift and subgroup performance.

## Strong Points to Emphasize

- You used real public biomedical datasets, not synthetic toy data.
- You connected model work to clinical workflow.
- You benchmarked foundation encoders rather than assuming one model.
- You kept negative or non-improving results, such as the CMMD auxiliary ablation.
- You added statistical depth: confidence intervals, fold variance, time-dependent AUC, and KM analysis.
- You can discuss both research and enterprise architecture.

## Weak Points to Acknowledge

- The two stages are connected as a deployment pathway, not validated as a single patient-linked clinical trial.
- Stage 2 C-index is modest.
- Five-fold tests have limited statistical power for pairwise encoder significance.
- A real hospital deployment would need regulatory, data protection, clinical safety, and prospective validation work.

Owning these limitations makes the project more credible.
