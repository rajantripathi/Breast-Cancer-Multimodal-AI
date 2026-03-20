from __future__ import annotations

"""Build a newline-delimited list of TCGA tile files missing extraction outputs."""

import argparse
from pathlib import Path

from config import load_settings


def build_missing_tile_list(model: str, output_path: Path) -> Path:
    """Write missing tile paths for slides lacking either embedding output."""
    settings = load_settings()
    tiles_dir = settings.project_root / "tcga-brca" / "tiles"
    slide_dir = settings.project_root / "tcga-brca" / "embeddings" / model
    patch_dir = settings.project_root / "tcga-brca" / "patch_embeddings" / model

    tile_paths = sorted(tiles_dir.glob("*.h5"))
    slide_stems = {path.stem for path in slide_dir.glob("*.pt")}
    patch_stems = {path.stem for path in patch_dir.glob("*.pt")}
    missing = [str(path) for path in tile_paths if path.stem not in slide_stems or path.stem not in patch_stems]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(missing) + ("\n" if missing else ""))
    print(
        f"missing_tile_count={len(missing)} "
        f"tiles_dir={tiles_dir} slide_dir={slide_dir} patch_dir={patch_dir} output={output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build list of TCGA slides missing embeddings")
    parser.add_argument("--model", default="uni2", choices=["uni2", "ctranspath", "virchow"])
    parser.add_argument("--output", default=None, help="Output path for newline-delimited tile list")
    args = parser.parse_args()

    settings = load_settings()
    default_output = settings.repo_root / "reports" / f"tcga_missing_tiles_{args.model}.txt"
    output_path = Path(args.output) if args.output else default_output
    build_missing_tile_list(args.model, output_path)


if __name__ == "__main__":
    main()
