# Domain Expert Review Pack

## Current Run Summary

- Vision dataset rows: 1980
- EHR dataset rows: 3218
- Genomics dataset rows: 5000
- Literature dataset rows: 4000
- Verifier dataset rows: 3960

## Latest Metrics

### Vision
- val_accuracy: 0.9675
- dataset_rows: 1980
- num_train: 1411
- num_val: 308
- loss_proxy: 0.0825
- perplexity_proxy: 1.033
- label_distribution: {'benign': 414, 'normal': 1242, 'malignant': 324}

### Ehr
- val_accuracy: 0.8423
- dataset_rows: 3218
- num_train: 2252
- num_val: 482
- loss_proxy: 0.2077
- perplexity_proxy: 1.1708
- label_distribution: {'low_risk': 2116, 'high_risk': 1102}

### Genomics
- val_accuracy: 0.7384
- dataset_rows: 5000
- num_train: 4531
- num_val: 1395
- loss_proxy: 0.3116
- perplexity_proxy: 1.2991
- label_distribution: {'pathogenic_variant': 2500, 'benign_variant': 2500}

### Literature
- val_accuracy: 0.7143
- dataset_rows: 4000
- num_train: 2924
- num_val: 679
- loss_proxy: 0.3357
- perplexity_proxy: 1.3307
- label_distribution: {'supportive_evidence': 2000, 'limited_evidence': 2000}

### Verifier
- val_accuracy: 0.9816
- num_samples: 2772

## Example Fused Outputs

- `case_01_benign` -> `monitor`
- `case_02_malignant` -> `high_concern`
- `case_03_brca` -> `high_concern`

## Known Limitations

- The four modality agents now train on much larger public datasets, but they are still stronger baselines rather than full domain-specific transformer fine-tuning.
- The verifier is expanded through weakly aligned synthetic/public bundling, not true patient-linked multimodal cohorts.
- These results support technical review and data-quality feedback, not clinical validation.
