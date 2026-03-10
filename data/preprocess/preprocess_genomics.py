from __future__ import annotations

import csv
import gzip

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "genomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    variant_tsv = settings.raw_data_root / "genomics" / "variant_summary.txt.gz"
    variant_csv = settings.raw_data_root / "genomics" / "genomics_variants.csv"
    if variant_tsv.exists():
        genes_of_interest = {"BRCA1", "BRCA2", "PALB2", "TP53", "CHEK2", "ATM"}
        with gzip.open(variant_tsv, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            label_counts = {"benign_variant": 0, "pathogenic_variant": 0}
            for row in reader:
                gene = row.get("GeneSymbol", "")
                if gene not in genes_of_interest:
                    continue
                significance = (row.get("ClinicalSignificance", "") or "").lower()
                if "pathogenic" in significance:
                    label = "pathogenic_variant"
                elif "benign" in significance:
                    label = "benign_variant"
                else:
                    continue
                if label_counts[label] >= 1500:
                    continue
                label_counts[label] += 1
                flattened = {
                    "gene": gene,
                    "clinical_significance": row.get("ClinicalSignificance", ""),
                    "review_status": row.get("ReviewStatus", ""),
                    "origin_simple": row.get("OriginSimple", ""),
                    "phenotype_list": row.get("PhenotypeList", ""),
                    "type": row.get("Type", ""),
                }
                rows.append(
                    {
                        "sample_id": row.get("VariationID", f"{gene}_{label_counts[label]}"),
                        "label": label,
                        "text": flatten_payload(flattened),
                        "metadata": {"source": str(variant_tsv)},
                    }
                )
                if sum(label_counts.values()) >= 3000:
                    break
    elif variant_csv.exists():
        with variant_csv.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "text": flatten_payload(row),
                        "metadata": {"source": str(variant_csv)},
                    }
                )
    else:
        for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
            case = read_json(case_path)
            label = "pathogenic_variant" if "pathogenic" in flatten_payload(case["genomics"]) else "benign_variant"
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "label": label,
                    "text": flatten_payload(case["genomics"]),
                    "metadata": {"source": str(case_path)},
                }
            )

    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"genomics processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
