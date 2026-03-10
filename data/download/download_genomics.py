from __future__ import annotations

import csv

from config import load_settings
from data.common import write_json


def main() -> None:
    settings = load_settings()
    out_dir = settings.raw_data_root / "genomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_file = out_dir / "genomics_variants.csv"
    with variants_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "gene", "variant", "label"])
        writer.writerow(["gen_001", "BRCA1", "c.68_69delAG", "pathogenic_variant"])
        writer.writerow(["gen_002", "BRCA2", "c.5946delT", "pathogenic_variant"])
        writer.writerow(["gen_003", "PALB2", "synonymous", "benign_variant"])
        writer.writerow(["gen_004", "BRCA1", "wildtype", "benign_variant"])

    manifest = {
        "dataset": "public_brca_seed",
        "target_dir": str(out_dir),
        "status": "seeded",
        "raw_file": str(variants_file),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"genomics download manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
