# Paper 2 Manuscript Update Note

## Stage 1 statistical-depth results

Final Stage 1 mammography evaluation was completed from the legacy
ConvNeXt-Base checkpoint on the VinDr-Mammo test set.

- Test set size: `750` exams
- Positive exams: `30`
- AUROC: `0.7407` (95% CI `0.6492–0.8254`)
- Sensitivity at 90% specificity: `0.3667` (95% CI `0.2000–0.5946`)
- Specificity at 90% sensitivity: `0.3847` (95% CI `0.3505–0.5843`)
- Brier score: `0.0366`

### Suggested manuscript wording

`On the VinDr-Mammo test set (n=750 exams; 30 positive), the final legacy ConvNeXt-Base screening model achieved an AUROC of 0.7407 (95% CI 0.6492–0.8254). At an operating point targeting 90% specificity, sensitivity was 0.3667 (95% CI 0.2000–0.5946). At an operating point targeting 90% sensitivity, specificity was 0.3847 (95% CI 0.3505–0.5843). The Brier score was 0.0366.`

### Confusion-matrix values

At the 90%-specificity operating point:

- Threshold: `0.024771`
- TN: `650`
- FP: `70`
- FN: `19`
- TP: `11`

At the Youden threshold:

- Threshold: `0.014913`
- TN: `563`
- FP: `157`
- FN: `13`
- TP: `17`

### Density subgroup summary

After refreshing VinDr preprocessing to retain density metadata, Stage 1
predictions were stratified by `exam_density`.

- Density B: `n=73`, positives `5`, prevalence `0.0685`, AUROC `0.8265`
- Density C: `n=564`, positives `24`, prevalence `0.0426`, AUROC `0.7257`
- Density D: `n=113`, positives `1`, prevalence `0.0088`, AUROC `0.7054`

### Suggested density subgroup wording

`In a supplementary density-stratified analysis using refreshed VinDr metadata, the test cohort contained density categories B (n=73, 5 positive), C (n=564, 24 positive), and D (n=113, 1 positive). AUROC estimates were 0.8265 for density B and 0.7257 for density C. The density-D estimate (0.7054) should be interpreted cautiously because only one positive exam was present in that subgroup.`

## Stage 2 statistical-depth highlights

### Pairwise encoder significance

Exact sign-flip p-values over the five cross-validation folds:

- CONCH vs CTransPath: mean C-index difference `0.0059`, `p=0.9375`
- CONCH vs UNI2: mean C-index difference `0.0198`, `p=0.5000`
- CTransPath vs UNI2: mean C-index difference `0.0139`, `p=0.6250`

### Ablation significance

- V+G vs V: mean difference `0.0172`, `p=0.8125`
- V+C+G vs V: mean difference `0.0418`, `p=0.3125`
- V+C+G vs V+G: mean difference `0.0247`, `p=0.5000`
- V+C vs V: mean difference `-0.0060`, `p=0.7500`

### Time-dependent AUC

- 2-year AUROC: `0.5855`
- 3-year AUROC: `0.5686`
- 5-year AUROC: `0.6123`

At 5 years by encoder:

- CONCH: `0.5760`
- CTransPath: `0.5520`
- UNI2: `0.4918`

### Alternative stratification

Median split:

- Low-risk group: `n=521`, event rate `0.1036`
- High-risk group: `n=522`, event rate `0.1590`
- Log-rank `p=0.0050`

## Reporting caveat

For Stage 2 B1/B2 comparisons, inference is based on five paired
cross-validation folds. Exact sign-flip p-values are the appropriate headline
tests here, and non-significant results should be interpreted as limited power
rather than proof of equivalence.
