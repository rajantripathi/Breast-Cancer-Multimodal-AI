from __future__ import annotations

"""Download EMBED open-data tables and selected images from AWS.

The EMBED open-data release is exposed through the AWS Open Data program under
the ``embed-dataset-open`` bucket. The official FAQ recommends downloading the
tables first, then selecting a cohort-specific image subset from the metadata
table before transferring image objects.

This script follows that pattern:

1. Download the table objects from ``tables/`` into a local raw directory.
2. Optionally download image objects listed in a manifest produced by
   ``agents.mammography.preprocessing.prepare_embed``.

The script is intentionally conservative: it does not try to infer a labeling
policy or image subset itself. Those decisions live in the prepare step.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from data.common import write_json


DEFAULT_BUCKET = "embed-dataset-open"
DEFAULT_TABLE_KEYS = (
    "tables/EMBED_OpenData_clinical.csv",
    "tables/EMBED_OpenData_metadata.csv",
    "tables/EMBED_OpenData_clinical_reduced.csv",
    "tables/EMBED_OpenData_metadata_reduced.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EMBED open-data tables and selected DICOMs from AWS")
    parser.add_argument("--output-dir", required=True, help="Target raw EMBED directory")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="AWS Open Data bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region for the EMBED bucket")
    parser.add_argument("--profile", default=None, help="Optional AWS profile name")
    parser.add_argument(
        "--unsigned",
        action="store_true",
        help="Use unsigned anonymous S3 access. Useful when the EMBED bucket is readable without configured AWS credentials.",
    )
    parser.add_argument(
        "--table-key",
        action="append",
        default=[],
        help="Explicit table object keys to download. Defaults to the four standard EMBED table files.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip downloading table files. Useful when only syncing image objects from a manifest.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional newline-delimited object key manifest. When provided with --download-images, only these DICOMs are fetched.",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Download image objects listed in --manifest into output-dir/images.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional limit for smoke tests. Applies after manifest loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing files instead of skipping them.",
    )
    return parser.parse_args()


def _build_s3_client(profile: str | None, region: str, *, unsigned: bool):
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for EMBED download. Install it into the active environment.") from exc
    try:
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError("botocore is required for EMBED download. Install boto3/botocore into the active environment.") from exc

    session = boto3.session.Session(profile_name=profile, region_name=region)
    if unsigned:
        return session.client("s3", config=Config(signature_version=UNSIGNED))
    return session.client("s3")


def _normalize_manifest_key(raw: str, bucket: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith(f"s3://{bucket}/"):
        return text[len(f"s3://{bucket}/") :]
    if text.startswith("s3://"):
        raise ValueError(f"Manifest key points at a different bucket: {text}")
    return text.lstrip("/")


def _read_manifest(path: str | Path, bucket: str) -> list[str]:
    keys: list[str] = []
    for line in Path(path).read_text().splitlines():
        key = _normalize_manifest_key(line, bucket)
        if key:
            keys.append(key)
    deduped = list(dict.fromkeys(keys))
    return deduped


def _download_object(client, bucket: str, key: str, target: Path, *, overwrite: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return "skipped_existing"
    client.download_file(bucket, key, str(target))
    return "downloaded"


def _download_many(
    client,
    bucket: str,
    pairs: Iterable[tuple[str, Path]],
    *,
    overwrite: bool,
) -> dict[str, int]:
    counts = {"downloaded": 0, "skipped_existing": 0, "failed": 0}
    for key, target in pairs:
        try:
            status = _download_object(client, bucket, key, target, overwrite=overwrite)
        except Exception:
            counts["failed"] += 1
            continue
        counts[status] += 1
    return counts


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_tables_dir = output_dir / "tables"
    raw_images_dir = output_dir / "images"
    raw_tables_dir.mkdir(parents=True, exist_ok=True)
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    client = _build_s3_client(args.profile, args.region, unsigned=bool(args.unsigned))

    table_keys = list(dict.fromkeys(args.table_key or list(DEFAULT_TABLE_KEYS)))
    table_counts = {"downloaded": 0, "skipped_existing": 0, "failed": 0}
    if not args.skip_tables:
        table_pairs = [(key, raw_tables_dir / Path(key).name) for key in table_keys]
        table_counts = _download_many(client, args.bucket, table_pairs, overwrite=args.overwrite)

    image_counts = {"downloaded": 0, "skipped_existing": 0, "failed": 0}
    selected_manifest: list[str] = []
    if args.download_images:
        if not args.manifest:
            raise ValueError("--manifest is required when --download-images is set")
        selected_manifest = _read_manifest(args.manifest, args.bucket)
        if args.limit_images is not None:
            selected_manifest = selected_manifest[: int(args.limit_images)]
        image_pairs = [(key, raw_images_dir / key) for key in selected_manifest]
        image_counts = _download_many(client, args.bucket, image_pairs, overwrite=args.overwrite)

    summary = {
        "bucket": args.bucket,
        "region": args.region,
        "profile": args.profile or os.environ.get("AWS_PROFILE"),
        "unsigned": bool(args.unsigned),
        "tables": {
            "requested_keys": table_keys,
            **table_counts,
        },
        "images": {
            "requested_count": len(selected_manifest),
            "manifest": args.manifest,
            **image_counts,
        },
    }
    write_json(output_dir / "download_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
