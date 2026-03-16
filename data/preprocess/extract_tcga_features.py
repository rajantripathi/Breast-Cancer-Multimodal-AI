from __future__ import annotations

"""Extract TCGA slide and patch embeddings from tiled HDF5 inputs."""

import argparse
from pathlib import Path
import time

import h5py
import torch
from PIL import Image

from agents.vision.foundation_models import get_embed_dim, load_model
from config import load_settings


def _device_from_arg(device_arg: str | None) -> torch.device:
    """Resolve the extraction device."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_one(
    tile_path: Path,
    model_key: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    print_h5_diagnostic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract patch and slide embeddings from one tiled slide."""
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
                outputs = model(inputs)
            if not isinstance(outputs, torch.Tensor):
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
    parser.add_argument("--model", default="uni2", choices=["uni2", "ctranspath", "virchow"])
    parser.add_argument("--tiles-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patch-output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

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
    tile_paths = sorted(tiles_dir.glob("*.h5"))
    print(f"Tiles directory: {tiles_dir}", flush=True)
    print(f"Slide embedding output directory: {output_dir}", flush=True)
    print(f"Patch embedding output directory: {patch_output_dir}", flush=True)
    print(f"Found {len(tile_paths)} tile files to process", flush=True)
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
