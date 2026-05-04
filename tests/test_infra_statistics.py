from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.subgroups import (
    attach_tcga_subgroups,
    load_tcga_clinical_subgroups,
    summarize_survival_subgroups,
)
from evaluation.statistics import (
    binary_auroc,
    binary_brier_score,
    calibration_slope_intercept,
    bootstrap_confidence_interval,
    decision_curve,
    delong_roc_test,
    exact_sign_flip_pvalue,
    paired_bootstrap_test,
    survival_binary_labels_at_horizon,
)
from training.reproducibility import build_run_manifest, set_global_seed


class StatisticsHelpersTest(unittest.TestCase):
    def test_binary_auroc_is_perfect_for_ranked_scores(self) -> None:
        self.assertEqual(binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_delong_returns_valid_payload(self) -> None:
        result = delong_roc_test([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [0.2, 0.3, 0.7, 0.8])
        self.assertIn("p_value", result)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_paired_bootstrap_identical_models_have_zero_delta(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        result = paired_bootstrap_test(
            lambda labels, sample_scores: binary_auroc(labels, sample_scores),
            [0, 0, 1, 1],
            model_a=scores,
            model_b=scores,
            iterations=50,
            stratify=True,
        )
        self.assertAlmostEqual(result.mean_delta, 0.0, places=6)

    def test_bootstrap_confidence_interval_returns_bounds(self) -> None:
        lower, upper = bootstrap_confidence_interval(
            lambda labels, scores: binary_auroc(labels, scores),
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            iterations=50,
            stratify=True,
        )
        self.assertLessEqual(lower, upper)

    def test_exact_sign_flip_pvalue_is_in_unit_interval(self) -> None:
        p_value = exact_sign_flip_pvalue([0.1, 0.2, -0.1, 0.3])
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_survival_horizon_labels_exclude_early_censored_cases(self) -> None:
        payload = survival_binary_labels_at_horizon(
            [100.0, 500.0, 2000.0],
            [0, 1, 0],
            [0.1, 0.8, 0.2],
            365.0,
        )
        self.assertEqual(payload["n_eligible"], 2)
        self.assertEqual(payload["labels"], [0, 0])

    def test_decision_curve_returns_rows(self) -> None:
        rows = decision_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], thresholds=[0.25, 0.5, 0.75])
        self.assertEqual(len(rows), 3)
        self.assertIn("net_benefit_model", rows[0])

    def test_binary_brier_and_calibration_fit(self) -> None:
        self.assertAlmostEqual(binary_brier_score([0, 1], [0.1, 0.9]), 0.01, places=6)
        calibration = calibration_slope_intercept([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertIn("slope", calibration)


class ReproducibilityHelpersTest(unittest.TestCase):
    def test_build_run_manifest_captures_seed_and_inputs(self) -> None:
        seed_state = set_global_seed(7)
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_path = Path(tmp_dir) / "payload.txt"
            payload_path.write_text("hello")
            args = type("Args", (), {"seed": 7, "demo": True})()
            manifest = build_run_manifest(
                task="demo",
                args=args,
                input_paths=[payload_path],
                split_counts={"train": 1},
                seed_state=seed_state,
                extra={"note": "ok"},
                repo_root=Path(__file__).resolve().parents[1],
            )
        self.assertEqual(manifest["task"], "demo")
        self.assertEqual(manifest["seed_state"]["seed"], 7)
        self.assertEqual(manifest["input_paths"][0]["exists"], True)
        self.assertIn("manifest_sha256", manifest)


class SubgroupHelpersTest(unittest.TestCase):
    def test_tcga_subgroup_join_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            clinical_path = Path(tmp_dir) / "clinical.csv"
            clinical_path.write_text(
                "\n".join(
                    [
                        "bcr_patient_barcode,tumor_stage,pathologic_stage,er_status_by_ihc,pr_status_by_ihc,her2_status_by_ihc",
                        "TCGA-01-0001,stage ii,stage ii,positive,negative,positive",
                        "TCGA-02-0002,stage iii,stage iii,negative,negative,negative",
                    ]
                )
                + "\n"
            )
            lookup = load_tcga_clinical_subgroups(clinical_path)
            attached = attach_tcga_subgroups(
                [
                    {"sample_id": "TCGA-01-0001", "survival_time": 1000, "event_observed": 1, "risk_score": 0.8},
                    {"sample_id": "TCGA-02-0002", "survival_time": 1500, "event_observed": 0, "risk_score": 0.2},
                ],
                lookup,
            )
            self.assertEqual(attached[0]["er_status"], "Positive")
            summary = summarize_survival_subgroups(attached, "pathologic_stage", horizon_days=1825.0, min_group_size=1, min_events=0)
            self.assertIn("Stage II", summary)
            self.assertIn("Stage III", summary)


if __name__ == "__main__":
    unittest.main()
