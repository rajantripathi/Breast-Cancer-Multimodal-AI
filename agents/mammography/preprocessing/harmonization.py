from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    from PIL import Image
except ImportError:
    Image = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _normalize_unit_range(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.min() < 0.0 or array.max() > 1.0:
        array = array - float(array.min())
        max_value = float(array.max())
        if max_value > 0.0:
            array = array / max_value
    return np.clip(array, 0.0, 1.0)


def load_mammography_array(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        if Image is None:
            raise ImportError("Pillow is required for mammography harmonization")
        image = Image.open(image_path)
        array = np.asarray(image, dtype=np.float32)
        max_value = float(np.max(array)) if array.size else 0.0
        scale = 65535.0 if max_value > 255.0 else 255.0
        if scale > 0.0:
            array = array / scale
        return _normalize_unit_range(array)

    if pydicom is None:
        raise ImportError("pydicom is required for DICOM mammography harmonization")
    dataset = pydicom.dcmread(image_path)
    array = dataset.pixel_array.astype(np.float32)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = float(array.max()) - array
    return _normalize_unit_range(array)


def _iter_exam_view_paths(exams: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for exam in exams:
        source = str(exam.get("dataset_source") or "unknown").strip().lower() or "unknown"
        png_views = exam.get("png_views", {}) or {}
        for view_key in sorted(set(png_views) | set(exam.get("views", {}))):
            candidate = png_views.get(view_key) or exam.get("views", {}).get(view_key)
            if candidate is None:
                continue
            path = Path(candidate)
            if path.exists():
                pairs.append((source, path))
    return pairs


def fit_source_harmonization(
    exams: list[dict[str, Any]],
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    max_images_per_source: int = 256,
) -> dict[str, Any]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")
    if max_images_per_source <= 0:
        raise ValueError("max_images_per_source must be positive")

    sampled_paths: dict[str, list[Path]] = {}
    source_exam_ids: dict[str, set[str]] = {}
    for exam in exams:
        source = str(exam.get("dataset_source") or "unknown").strip().lower() or "unknown"
        source_exam_ids.setdefault(source, set()).add(str(exam.get("sample_id") or exam.get("study_id") or ""))
    for source, path in _iter_exam_view_paths(exams):
        bucket = sampled_paths.setdefault(source, [])
        if len(bucket) < max_images_per_source:
            bucket.append(path)

    payload: dict[str, Any] = {
        "method": "source_percentile",
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "max_images_per_source": int(max_images_per_source),
        "sources": {},
    }
    for source, paths in sorted(sampled_paths.items()):
        lowers: list[float] = []
        uppers: list[float] = []
        for path in paths:
            array = load_mammography_array(path)
            lowers.append(float(np.quantile(array, lower_quantile)))
            uppers.append(float(np.quantile(array, upper_quantile)))
        lower = float(np.median(lowers)) if lowers else 0.0
        upper = float(np.median(uppers)) if uppers else 1.0
        if upper <= lower:
            lower, upper = 0.0, 1.0
        payload["sources"][source] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "sampled_images": len(paths),
            "sampled_exams": len(source_exam_ids.get(source, set())),
        }
    return payload


def apply_source_harmonization(array: np.ndarray, dataset_source: str | None, stats: dict[str, Any] | None) -> np.ndarray:
    normalized = _normalize_unit_range(array)
    if not stats or not isinstance(stats, dict):
        return normalized
    source = str(dataset_source or "unknown").strip().lower() or "unknown"
    source_stats = (stats.get("sources") or {}).get(source)
    if not source_stats:
        return normalized
    lower = float(source_stats.get("lower_bound", 0.0))
    upper = float(source_stats.get("upper_bound", 1.0))
    if upper <= lower:
        return normalized
    clipped = np.clip(normalized, lower, upper)
    return ((clipped - lower) / max(upper - lower, 1e-6)).astype(np.float32)


def save_harmonization_stats(path: str | Path, stats: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(stats, indent=2))


def load_harmonization_stats(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
