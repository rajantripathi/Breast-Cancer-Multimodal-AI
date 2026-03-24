from __future__ import annotations

"""Extract TCGA slide and patch embeddings from tiled HDF5 inputs."""

import argparse
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _resolve_tile_paths(tiles_dir: Path, tile_list_path: Path | None) -> list[Path]:
    """Return the extraction target tile paths."""
    if tile_list_path is None:
        return sorted(tiles_dir.glob("*.h5"))
    paths: list[Path] = []
    for raw_line in tile_list_path.read_text().splitlines():
        candidate_text = raw_line.strip()
        if not candidate_text:
            continue
        candidate = Path(candidate_text)
        if not candidate.is_absolute():
            candidate = tiles_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Tile path from list does not exist: {candidate}")
        paths.append(candidate)
    return paths


def _device_from_arg(device_arg: str | None) -> torch.device:
    """Resolve the extraction device."""
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_one(
    tile_path: Path,
    model_key: str,
    model: Any,
    transform,
    device: Any,
    batch_size: int,
    print_h5_diagnostic: bool,
) -> tuple[Any, Any]:
    """Extract patch and slide embeddings from one tiled slide."""
    import h5py
    import torch
    from PIL import Image
    from agents.vision.foundation_models import get_embed_dim

    with h5py.File(tile_path, "r") as handle:
        if print_h5_diagnostic:
            tiles_shape = handle["tiles"].shape if "tiles" in handle else "<missing>"
            print(f"H5 keys: {list(handle.keys())}, shape: {tiles_shape}", flush=True)
        tiles = handle["tiles"]
        patch_embeddings: list[torch.Tensor] = []
        for start in range(0, len(tiles), batch_size):
            batch = [transform(Image.fromarray(tile)) for tile in tiles[start : start + batch_size]]
            inputs = torch.stack(batch, dim=0).to(device)
            with torch.no_grad():
                if hasattr(model, "encode_image"):
                    outputs = model.encode_image(inputs)
                else:
                    outputs = model(inputs)
            if not isinstance(outputs, torch.Tensor):
                if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
                    outputs = outputs[0]
                else:
                    raise TypeError(f"{model_key} returned non-tensor outputs")
            patch_embeddings.append(outputs.detach().cpu())
    if not patch_embeddings:
        raise ValueError(f"{tile_path.name} contains no tiles")
    patch_tensor = torch.cat(patch_embeddings, dim=0)
    slide_embedding = patch_tensor.mean(dim=0)
    expected_dim = get_embed_dim(model_key)
    if slide_embedding.shape[-1] != expected_dim:
        raise AssertionError(f"{model_key} embedding dim {slide_embedding.shape[-1]} != {expected_dim}")
    return patch_tensor, slide_embedding


def main() -> None:
    """CLI entrypoint for TCGA slide feature extraction."""
    parser = argparse.ArgumentParser(description="Extract TCGA slide embeddings from tiled HDF5 files")
    parser.add_argument("--model", default="uni2", choices=["uni2", "conch", "ctranspath", "virchow"])
    parser.add_argument("--tiles-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patch-output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--tile-list",
        default=None,
        help="Optional newline-delimited file of .h5 tile paths or basenames to process",
    )
    args = parser.parse_args()

    import torch

    from agents.vision.foundation_models import get_embed_dim, load_model
    from config import load_settings

    settings = load_settings()
    tiles_dir = Path(args.tiles_dir) if args.tiles_dir else settings.project_root / "tcga-brca" / "tiles"
    output_dir = Path(args.output_dir) if args.output_dir else settings.project_root / "tcga-brca" / "embeddings" / args.model
    patch_output_dir = (
        Path(args.patch_output_dir) if args.patch_output_dir else settings.project_root / "tcga-brca" / "patch_embeddings" / args.model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_output_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(args.device)
    print(f"Loading model {args.model}...", flush=True)
    model_load_start = time.monotonic()
    model, transform = load_model(args.model)
    model_load_elapsed = time.monotonic() - model_load_start
    print(f"Model loaded. Embed dim: {get_embed_dim(args.model)}. Load time: {model_load_elapsed:.1f}s", flush=True)
    if model_load_elapsed > 600:
        print(f"WARNING: model loading took {model_load_elapsed:.1f}s for {args.model}", flush=True)
    model = model.to(device)
    tile_list_path = Path(args.tile_list).resolve() if args.tile_list else None
    tile_paths = _resolve_tile_paths(tiles_dir, tile_list_path)
    total_tile_paths = len(tile_paths)
    if tile_list_path is not None:
        if args.shard_index is not None or args.num_shards is not None:
            raise ValueError("--tile-list cannot be combined with --shard-index/--num-shards")
        shard_index = 0
        num_shards = 1
    elif args.shard_index is None and args.num_shards is None:
        shard_index = 0
        num_shards = 1
    elif args.shard_index is not None and args.num_shards is not None:
        shard_index = args.shard_index
        num_shards = args.num_shards
    else:
        raise ValueError("--shard-index and --num-shards must be provided together")
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if tile_list_path is None:
        tile_paths = [path for index, path in enumerate(tile_paths) if index % num_shards == shard_index]
    print(f"Tiles directory: {tiles_dir}", flush=True)
    if tile_list_path is not None:
        print(f"Tile list: {tile_list_path}", flush=True)
    print(f"Slide embedding output directory: {output_dir}", flush=True)
    print(f"Patch embedding output directory: {patch_output_dir}", flush=True)
    print(f"Found {total_tile_paths} tile files to process", flush=True)
    if tile_list_path is None:
        print(f"Shard {shard_index}/{num_shards}: processing {len(tile_paths)} of {total_tile_paths} files", flush=True)
    else:
        print(f"Processing explicit tile list with {len(tile_paths)} files", flush=True)
    processed = 0
    skipped = 0
    printed_h5_diagnostic = False
    for index, tile_path in enumerate(tile_paths, start=1):
        patient_barcode = tile_path.stem
        slide_output = output_dir / f"{patient_barcode}.pt"
        patch_output = patch_output_dir / f"{patient_barcode}.pt"
        print(f"Processing {index}/{len(tile_paths)}: {tile_path.name}", flush=True)
        if slide_output.exists() and patch_output.exists():
            print(f"Skipping existing outputs for {tile_path.name}", flush=True)
            continue
        try:
            patch_tensor, slide_tensor = _extract_one(
                tile_path,
                args.model,
                model,
                transform,
                device,
                args.batch_size,
                print_h5_diagnostic=not printed_h5_diagnostic,
            )
            printed_h5_diagnostic = True
        except (OSError, KeyError, ValueError, TypeError, AssertionError) as exc:
            skipped += 1
            print(f"skip failed tile file {tile_path.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        torch.save(slide_tensor, slide_output)
        torch.save(patch_tensor, patch_output)
        processed += 1
        print(f"Saved embeddings for {tile_path.name}: {slide_output.name}, {patch_output.name}", flush=True)
        if processed % 50 == 0:
            print(f"processed {processed} slides with {args.model}", flush=True)
    print(f"finished extraction with {args.model}, processed={processed}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
