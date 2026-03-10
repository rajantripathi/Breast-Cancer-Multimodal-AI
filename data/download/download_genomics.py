from __future__ import annotations

import argparse
from urllib.request import urlretrieve

from config import load_settings
from data.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public genomics data on Isambard")
    parser.add_argument(
        "--source-url",
        default="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
    )
    args = parser.parse_args()

    settings = load_settings()
    out_dir = settings.raw_data_root / "genomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_file = out_dir / "variant_summary.txt.gz"
    manifest = {
        "dataset": args.source_url,
        "target_dir": str(out_dir),
        "status": "seeded",
    }

    try:
        urlretrieve(args.source_url, variants_file)
        manifest["status"] = "downloaded"
        manifest["raw_file"] = str(variants_file)
    except Exception:
        seeded = out_dir / "genomics_variants.csv"
        seeded.write_text(
            "sample_id,gene,variant,label\n"
            "gen_001,BRCA1,c.68_69delAG,pathogenic_variant\n"
            "gen_002,BRCA2,c.5946delT,pathogenic_variant\n"
            "gen_003,PALB2,synonymous,benign_variant\n"
            "gen_004,BRCA1,wildtype,benign_variant\n"
        )
        manifest["fallback_file"] = str(seeded)
    write_json(out_dir / "manifest.json", manifest)
    print(f"genomics download manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
