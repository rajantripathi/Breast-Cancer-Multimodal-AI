from __future__ import annotations

"""Tile TCGA WSI slides into tissue-rich patches."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from config import load_settings


def _openslide_module():
    import openslide

    return openslide


def _patient_barcode_from_path(path: Path) -> str:
    """Extract the patient barcode from a normalized download path."""
    return path.name.split("__", 1)[0]


def _slide_level(slide: Any, target_magnification: float = 20.0) -> int:
    """Choose the slide level closest to the target magnification."""
    objective_text = slide.properties.get("openslide.objective-power") or slide.properties.get("aperio.AppMag") or "20"
    try:
        objective = float(objective_text)
    except ValueError:
        objective = target_magnification
    desired_downsample = max(objective / target_magnification, 1.0)
    best_level = 0
    best_delta = float("inf")
    for index, downsample in enumerate(slide.level_downsamples):
        delta = abs(float(downsample) - desired_downsample)
        if delta < best_delta:
            best_delta = delta
            best_level = index
    return best_level


def _tissue_fraction(tile: np.ndarray) -> float:
    """Estimate tissue coverage using Otsu thresholding."""
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    threshold, _mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = gray < threshold
    return float(tissue_mask.mean())


def _create_h5(path: Path) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    """Create the output HDF5 tile file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    tiles_ds = handle.create_dataset(
        "tiles",
        shape=(0, 256, 256, 3),
        maxshape=(None, 256, 256, 3),
        dtype=np.uint8,
        compression="gzip",
        compression_opts=4,
        chunks=(16, 256, 256, 3),
    )
    coords_ds = handle.create_dataset(
        "coords",
        shape=(0, 2),
        maxshape=(None, 2),
        dtype=np.int32,
        compression="gzip",
        compression_opts=4,
        chunks=(256, 2),
    )
    return handle, tiles_ds, coords_ds


def _append_batch(tiles_ds: h5py.Dataset, coords_ds: h5py.Dataset, tiles: list[np.ndarray], coords: list[tuple[int, int]]) -> None:
    """Append a batch of tiles and coordinates to HDF5 datasets."""
    if not tiles:
        return
    start = tiles_ds.shape[0]
    count = len(tiles)
    tiles_ds.resize(start + count, axis=0)
    coords_ds.resize(start + count, axis=0)
    tiles_ds[start : start + count] = np.stack(tiles, axis=0)
    coords_ds[start : start + count] = np.asarray(coords, dtype=np.int32)


def tile_slide(slide_path: Path, output_path: Path) -> int:
    """Tile a slide and save tissue-rich patches to HDF5.

    Args:
        slide_path: Source SVS path.
        output_path: Output HDF5 path.

    Returns:
        Number of kept tiles.
    """
    openslide = _openslide_module()
    slide = openslide.OpenSlide(str(slide_path))
    level = _slide_level(slide)
    width, height = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])
    handle, tiles_ds, coords_ds = _create_h5(output_path)
    kept = 0
    batch_tiles: list[np.ndarray] = []
    batch_coords: list[tuple[int, int]] = []
    try:
        for y in range(0, height - 255, 256):
            for x in range(0, width - 255, 256):
                region = slide.read_region((int(x * downsample), int(y * downsample)), level, (256, 256)).convert("RGB")
                tile = np.asarray(region)
                if _tissue_fraction(tile) <= 0.5:
                    continue
                batch_tiles.append(tile)
                batch_coords.append((x, y))
                kept += 1
                if len(batch_tiles) >= 128:
                    _append_batch(tiles_ds, coords_ds, batch_tiles, batch_coords)
                    batch_tiles.clear()
                    batch_coords.clear()
        _append_batch(tiles_ds, coords_ds, batch_tiles, batch_coords)
    finally:
        handle.close()
        slide.close()
    return kept


def main() -> None:
    """Tile TCGA-BRCA slides into HDF5 patch sets."""
    parser = argparse.ArgumentParser(description="Tile TCGA-BRCA slides at 20x into 256x256 patches")
    parser.add_argument("--slides-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    slides_dir = Path(args.slides_dir) if args.slides_dir else settings.project_root / "tcga-brca" / "slides"
    output_dir = Path(args.output_dir) if args.output_dir else settings.project_root / "tcga-brca" / "tiles"
    slide_paths = sorted(slides_dir.glob("*.svs"))
    if args.max_slides:
        slide_paths = slide_paths[: args.max_slides]

    for index, slide_path in enumerate(slide_paths, start=1):
        patient_barcode = _patient_barcode_from_path(slide_path)
        output_path = output_dir / f"{patient_barcode}.h5"
        tile_count = tile_slide(slide_path, output_path)
        print(f"[{index}/{len(slide_paths)}] {patient_barcode}: kept {tile_count} tiles")


if __name__ == "__main__":
    main()
