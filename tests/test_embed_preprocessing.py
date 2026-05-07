from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from agents.mammography.preprocessing.prepare_embed import (
    _exam_label,
    _normalize_assessment,
    _normalize_density,
)


class EmbedPreparationHelpersTest(unittest.TestCase):
    def test_normalize_assessment_and_density(self) -> None:
        self.assertEqual(_normalize_assessment("A - recall"), "A")
        self.assertEqual(_normalize_assessment("s"), "S")
        self.assertEqual(_normalize_density("1"), "A")
        self.assertEqual(_normalize_density("D"), "D")

    def test_exam_label_modes(self) -> None:
        self.assertEqual(_exam_label("screening", ["A"], False, "recall"), 1)
        self.assertEqual(_exam_label("screening", ["N"], False, "recall_or_pathology"), 0)
        self.assertEqual(_exam_label("screening", [], True, "recall_or_pathology"), 1)
        self.assertEqual(_exam_label("diagnostic", ["S"], False, "suspicious_assessment"), 1)


class EmbedPreparationIntegrationTest(unittest.TestCase):
    def test_prepare_embed_builds_external_metadata_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            tables_root = raw_root / "tables"
            tables_root.mkdir(parents=True)
            processed_root = root / "processed"

            clinical = pd.DataFrame(
                [
                    {
                        "empi_anon": "p1",
                        "acc_anon": "exam1",
                        "desc": "screening mammogram",
                        "asses": "A",
                        "side": "L",
                        "path_severity": "",
                        "tissueden": "3",
                    },
                    {
                        "empi_anon": "p1",
                        "acc_anon": "exam1",
                        "desc": "screening mammogram",
                        "asses": "N",
                        "side": "R",
                        "path_severity": "",
                        "tissueden": "3",
                    },
                ]
            )
            metadata = pd.DataFrame(
                [
                    {
                        "empi_anon": "p1",
                        "acc_anon": "exam1",
                        "anon_dicom_path": "cohort1/p1/exam1/lcc_2d.dcm",
                        "FinalImageType": "2D",
                        "ImageLateralityFinal": "L",
                        "ViewPosition": "CC",
                        "spot_mag": "0",
                    },
                    {
                        "empi_anon": "p1",
                        "acc_anon": "exam1",
                        "anon_dicom_path": "cohort1/p1/exam1/lmlo_2d.dcm",
                        "FinalImageType": "2D",
                        "ImageLateralityFinal": "L",
                        "ViewPosition": "MLO",
                        "spot_mag": "0",
                    },
                    {
                        "empi_anon": "p1",
                        "acc_anon": "exam1",
                        "anon_dicom_path": "cohort1/p1/exam1/rcc_2d.dcm",
                        "FinalImageType": "2D",
                        "ImageLateralityFinal": "R",
                        "ViewPosition": "CC",
                        "spot_mag": "0",
                    },
                    {
                        "empi_anon": "p1",
                        "acc_anon": "exam1",
                        "anon_dicom_path": "cohort1/p1/exam1/rmlo_2d.dcm",
                        "FinalImageType": "2D",
                        "ImageLateralityFinal": "R",
                        "ViewPosition": "MLO",
                        "spot_mag": "0",
                    },
                ]
            )
            clinical.to_csv(tables_root / "EMBED_OpenData_clinical_reduced.csv", index=False)
            metadata.to_csv(tables_root / "EMBED_OpenData_metadata_reduced.csv", index=False)

            from agents.mammography.preprocessing.prepare_embed import main as prepare_embed_main
            import sys

            argv = sys.argv[:]
            try:
                sys.argv = [
                    "prepare_embed",
                    "--input-dir",
                    str(raw_root),
                    "--output-dir",
                    str(processed_root),
                    "--exam-type",
                    "screening",
                    "--label-mode",
                    "recall_or_pathology",
                    "--image-types",
                    "2D,C-view",
                    "--preferred-image-type",
                    "2D",
                    "--allow-cview-fallback",
                    "--full-field-only",
                ]
                prepare_embed_main()
            finally:
                sys.argv = argv

            output = pd.read_csv(processed_root / "metadata.csv")
            self.assertEqual(output["split"].unique().tolist(), ["external"])
            self.assertEqual(output["dataset_source"].unique().tolist(), ["embed"])
            self.assertEqual(output["study_id"].nunique(), 1)
            self.assertEqual(output.groupby("study_id")["label"].first().iloc[0], 1)
            manifest_lines = (processed_root / "download_manifest.txt").read_text().splitlines()
            self.assertEqual(len(manifest_lines), 4)


if __name__ == "__main__":
    unittest.main()
