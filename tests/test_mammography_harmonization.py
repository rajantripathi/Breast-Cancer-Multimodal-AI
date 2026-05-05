from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from agents.mammography.preprocessing.harmonization import (
    apply_source_harmonization,
    fit_source_harmonization,
    load_harmonization_stats,
    save_harmonization_stats,
)


def _write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((values * 65535.0).astype(np.uint16)).save(path)


class MammographyHarmonizationTest(unittest.TestCase):
    def test_fit_source_harmonization_returns_per_source_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            low_a = np.linspace(0.05, 0.45, 16, dtype=np.float32).reshape(4, 4)
            high_a = np.linspace(0.10, 0.50, 16, dtype=np.float32).reshape(4, 4)
            low_b = np.linspace(0.55, 0.85, 16, dtype=np.float32).reshape(4, 4)
            high_b = np.linspace(0.60, 0.90, 16, dtype=np.float32).reshape(4, 4)

            a_lcc = root / "vindr" / "a_lcc.png"
            a_rcc = root / "vindr" / "a_rcc.png"
            b_lcc = root / "cbis" / "b_lcc.png"
            b_rcc = root / "cbis" / "b_rcc.png"
            for path, values in [
                (a_lcc, low_a),
                (a_rcc, high_a),
                (b_lcc, low_b),
                (b_rcc, high_b),
            ]:
                _write_png(path, values)

            exams = [
                {
                    "study_id": "vindr_a",
                    "sample_id": "vindr:vindr_a",
                    "dataset_source": "vindr",
                    "views": {"lcc": a_lcc, "rcc": a_rcc},
                    "png_views": {"lcc": a_lcc, "rcc": a_rcc},
                },
                {
                    "study_id": "cbis_b",
                    "sample_id": "cbis_ddsm:cbis_b",
                    "dataset_source": "cbis_ddsm",
                    "views": {"lcc": b_lcc, "rcc": b_rcc},
                    "png_views": {"lcc": b_lcc, "rcc": b_rcc},
                },
            ]
            stats = fit_source_harmonization(exams, lower_quantile=0.1, upper_quantile=0.9, max_images_per_source=4)

            self.assertEqual(stats["method"], "source_percentile")
            self.assertIn("vindr", stats["sources"])
            self.assertIn("cbis_ddsm", stats["sources"])
            self.assertLess(stats["sources"]["vindr"]["lower_bound"], stats["sources"]["vindr"]["upper_bound"])
            self.assertLess(stats["sources"]["cbis_ddsm"]["lower_bound"], stats["sources"]["cbis_ddsm"]["upper_bound"])

    def test_apply_source_harmonization_rescales_to_unit_interval(self) -> None:
        image = np.asarray([[0.2, 0.5], [0.7, 0.9]], dtype=np.float32)
        stats = {
            "method": "source_percentile",
            "sources": {"vindr": {"lower_bound": 0.2, "upper_bound": 0.8}},
        }
        harmonized = apply_source_harmonization(image, "vindr", stats)
        self.assertTrue(np.all(harmonized >= 0.0))
        self.assertTrue(np.all(harmonized <= 1.0))
        self.assertAlmostEqual(float(harmonized[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(harmonized[1, 1]), 1.0, places=6)

    def test_harmonization_stats_roundtrip_json(self) -> None:
        payload = {
            "method": "source_percentile",
            "sources": {"vindr": {"lower_bound": 0.1, "upper_bound": 0.9, "sampled_images": 8, "sampled_exams": 2}},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "harmonization_stats.json"
            save_harmonization_stats(path, payload)
            loaded = load_harmonization_stats(path)
        self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()
