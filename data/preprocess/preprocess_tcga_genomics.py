from __future__ import annotations

"""Process TCGA-BRCA STAR counts into patient-level genomics tensors."""

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from config import load_settings


PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6", "PGR", "PHGDH",
    "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T", "BCL2A1", "CENPA",
]


def _patient_from_path(path: Path) -> str:
    """Extract patient barcode from a normalized download file name."""
    return path.name.split("__", 1)[0]


def _load_expression_frame(path: Path) -> pd.Series:
    """Load one STAR counts TSV as a gene-level expression series."""
    frame = pd.read_csv(path, sep="\t", comment="#")
    gene_column = "gene_name" if "gene_name" in frame.columns else "gene_id"
    value_column = next(
        (column for column in ["tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"] if column in frame.columns),
        None,
    )
    if value_column is None:
        numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        if not numeric_columns:
            raise ValueError(f"could not identify expression column in {path}")
        value_column = numeric_columns[-1]
    series = frame[[gene_column, value_column]].dropna()
    series = series.groupby(gene_column)[value_column].mean()
    series.index = series.index.astype(str)
    return series


def main() -> None:
    """Build patient-level TCGA genomics embeddings and matrix exports."""
    parser = argparse.ArgumentParser(description="Process TCGA STAR counts into genomics tensors")
    parser.add_argument("--rnaseq-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--matrix-output", default=None)
    args = parser.parse_args()

    settings = load_settings()
    rnaseq_dir = Path(args.rnaseq_dir) if args.rnaseq_dir else settings.project_root / "tcga-brca" / "rnaseq"
    output_dir = Path(args.output_dir) if args.output_dir else settings.project_root / "tcga-brca" / "genomics"
    matrix_output = Path(args.matrix_output) if args.matrix_output else settings.project_root / "tcga-brca" / "full_expression_matrix.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_rows: dict[str, pd.Series] = {}
    for index, path in enumerate(sorted(rnaseq_dir.glob("*")), start=1):
        if not path.is_file():
            continue
        patient_rows[_patient_from_path(path)] = _load_expression_frame(path)
        if index % 50 == 0:
            print(f"loaded {index} RNA-seq files")
    matrix = pd.DataFrame.from_dict(patient_rows, orient="index").fillna(0.0)
    variances = matrix.var(axis=0).sort_values(ascending=False)
    pam50 = [gene for gene in PAM50_GENES if gene in matrix.columns]
    top_variable = [gene for gene in variances.index if gene not in pam50][:500]
    selected_genes = pam50 + top_variable
    selected_matrix = matrix[selected_genes]
    selected_matrix.to_csv(matrix_output)

    gene_list_path = matrix_output.with_suffix(".genes.csv")
    with gene_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene"])
        for gene in selected_genes:
            writer.writerow([gene])

    for index, (patient_barcode, row) in enumerate(selected_matrix.iterrows(), start=1):
        tensor = torch.tensor(row.to_numpy(dtype="float32"))
        torch.save(tensor, output_dir / f"{patient_barcode}.pt")
        if index % 50 == 0:
            print(f"saved {index} genomics tensors")
    print(f"processed {len(selected_matrix)} patients")


if __name__ == "__main__":
    main()
