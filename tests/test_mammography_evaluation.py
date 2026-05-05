from __future__ import annotations

import unittest

from agents.mammography.evaluation.evaluate_screener import (
    _infer_model_type_from_checkpoint,
    _normalize_predictions,
    _select_eval_exams,
)


class MammographyEvaluationHelpersTest(unittest.TestCase):
    def test_normalize_predictions_preserves_dataset_source(self) -> None:
        predictions = _normalize_predictions(
            [
                {
                    "study_id": "vindr:123",
                    "dataset_source": "vindr",
                    "true_label": "1",
                    "predicted_probability": 0.7,
                }
            ]
        )
        self.assertEqual(predictions[0]["dataset_source"], "vindr")
        self.assertEqual(predictions[0]["true_label"], 1)

    def test_select_eval_exams_auto_prefers_test_then_external(self) -> None:
        exams = [
            {"study_id": "train_1", "split": "train"},
            {"study_id": "external_1", "split": "external"},
            {"study_id": "test_1", "split": "test"},
        ]
        selected = _select_eval_exams(exams, "auto")
        self.assertEqual([exam["study_id"] for exam in selected], ["test_1"])

        selected_external = _select_eval_exams(
            [{"study_id": "external_1", "split": "external"}, {"study_id": "val_1", "split": "val"}],
            "auto",
        )
        self.assertEqual([exam["study_id"] for exam in selected_external], ["external_1"])

    def test_select_eval_exams_explicit_split_and_all(self) -> None:
        exams = [
            {"study_id": "train_1", "split": "train"},
            {"study_id": "val_1", "split": "val"},
        ]
        self.assertEqual([exam["study_id"] for exam in _select_eval_exams(exams, "val")], ["val_1"])
        self.assertEqual(len(_select_eval_exams(exams, "all")), 2)

    def test_infer_model_type_detects_standard_checkpoint_args(self) -> None:
        standard = {"args": {"effective_batch_size": 8, "loss": "smoothed_bce"}}
        legacy = {"args": {"epochs": 50, "lr": 1e-4}}
        self.assertEqual(_infer_model_type_from_checkpoint(standard), "standard")
        self.assertEqual(_infer_model_type_from_checkpoint(legacy), "legacy")


if __name__ == "__main__":
    unittest.main()
