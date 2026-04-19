# METABRIC External Validation Feasibility

## Conclusion

Public `METABRIC` is **not currently a drop-in external validation cohort**
for the paper's full pathology-plus-genomics-plus-clinical pipeline.

The blocker is not molecular or clinical data. Those are available. The
blocker is that the publicly accessible METABRIC resource used through
`cBioPortal` is a **tabular genomics/clinical dataset**, not a public
whole-slide-image cohort that can be run directly through the same
`CONCH` / `UNI2` / `CTransPath` feature extraction workflow used for
`TCGA-BRCA`.

## What Is Publicly Available

### Available

- Clinical and survival data through cBioPortal
- Gene expression / transcriptomic profiles through cBioPortal
- Copy-number and related molecular features through cBioPortal

### Not verified as publicly available in the same package

- Diagnostic H&E whole-slide images matched to the cBioPortal METABRIC cohort
- A public slide archive with patient-level linkage that can be consumed by the
  current WSI feature extraction pipeline without a separate access process

## Source-Based Assessment

1. The public METABRIC study exposed through cBioPortal is described as a
   cBioPortal dataset with downloadable `TSV` files and a small tabular dataset
   footprint, consistent with clinical/molecular data rather than pathology WSI
   storage.
2. Recent papers using METABRIC repeatedly describe it as a `cBioPortal`
   molecular/clinical cohort, typically using microarray expression, subtype,
   and survival endpoints from that portal.
3. I did not identify a primary-source public METABRIC WSI archive analogous to
   `TCGA` diagnostic slides that could be processed immediately with the
   repo's existing pathology feature extraction flow.

## Practical Implication for This Paper

### Full external validation of the current V+C+G model

**Not feasible immediately from public METABRIC alone**, because the current
best model depends on pathology slide embeddings and METABRIC does not appear to
provide a publicly linked matched WSI cohort in the same public access path.

### Partial external validation options

The following are feasible if a revision specifically asks for some external
validation signal:

1. **Genomics + clinical only external test**
   - Build a METABRIC-compatible external model that excludes pathology.
   - This is not a like-for-like validation of the published full multimodal
     model, but it would test whether the pathway/clinical signal generalizes.

2. **Cohort-shift analysis without model execution**
   - Compare feature availability, endpoint definitions, and subtype
     distributions between `TCGA-BRCA` and `METABRIC`.
   - Use this to justify why full external WSI validation is deferred.

3. **New data acquisition path**
   - Obtain a breast cancer WSI cohort with matched transcriptomics, clinical
     data, and survival.
   - Re-extract pathology features with `CONCH` / `UNI2` / `CTransPath`.
   - Recompute pathway features from the external transcriptomic platform.
   - Harmonize endpoints and evaluate the locked model.

## If Full METABRIC Validation Is Attempted Later

The required workflow would be:

1. Confirm access to matched METABRIC H&E WSIs with patient/sample identifiers.
2. Map those identifiers to the METABRIC molecular and clinical records.
3. Extract slide embeddings with the same pathology encoder used in the paper.
4. Harmonize transcriptomic features to the pathway representation used in the
   paper.
5. Harmonize survival endpoint definitions with the paper's target endpoint.
6. Run the trained model on METABRIC and report external `C-index`.

## Recommendation

For the initial submission:

- Keep `METABRIC external validation` in the limitations / future work section.
- Do **not** promise immediate full external validation using public METABRIC.
- If reviewers ask for external evidence, offer either:
  - a genomics+clinical-only METABRIC analysis, or
  - a scoped plan for acquiring a true matched external WSI cohort.

## Sources

- cBioPortal METABRIC study entry:
  `https://www.cbioportal.org/study/summary?id=brca_metabric`
- MSK data catalog entry for the public cBioPortal METABRIC dataset:
  `https://datacatalog.mskcc.org/dataset/11457`
- Example recent papers using METABRIC via cBioPortal as a molecular/clinical
  cohort rather than a public WSI cohort:
  - `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310557`
  - `https://breast-cancer-research.biomedcentral.com/articles/10.1186/s13058-022-01550-y`
