from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.statistics import harrell_c_index


CONFIGS = {
    "verifier_mean": Path("outputs/stage2/conch/verifier_mean"),
    "verifier_abmil": Path("outputs/stage2/conch/verifier_abmil"),
    "verifier_transmil": Path("outputs/stage2/conch/verifier_transmil"),
    "simple_fusion_abmil": Path("outputs/stage2/conch/simple_fusion_abmil"),
    "simple_fusion_transmil": Path("outputs/stage2/conch/simple_fusion_transmil"),
    "simple_fusion_mean": Path("outputs/stage2/conch/simple_fusion_mean"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 2 aggregation follow-up experiments")
    parser.add_argument("--root", type=Path, default=Path("."), help="Experiment root containing outputs/ and reports/")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/paper2/aggregation_summary.json"),
        help="Relative path for the aggregation summary JSON",
    )
    parser.add_argument(
        "--variance-output",
        type=Path,
        default=Path("reports/paper2/aggregation_fold_variance.json"),
        help="Relative path for the aggregation fold variance JSON",
    )
    return parser.parse_args()


def seed_sort_key(path: Path) -> int | str:
    match = re.search(r"(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return path.name


def load_predictions(seed_dir: Path) -> list[dict]:
    artifact_path = seed_dir / "artifact.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        predictions = artifact.get("predictions", [])
        if predictions:
            return predictions
    predictions_path = seed_dir / "predictions.json"
    if predictions_path.exists():
        return json.loads(predictions_path.read_text())
    raise FileNotFoundError(f"No predictions found for {seed_dir}")


def fold_c_indices(predictions: list[dict]) -> list[float]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[int(row["fold"])].append(row)
    values: list[float] = []
    for fold in sorted(grouped):
        rows = grouped[fold]
        values.append(
            harrell_c_index(
                [float(row["survival_time"]) for row in rows],
                [float(row["risk_score"]) for row in rows],
                [int(row["event_observed"]) for row in rows],
            )
        )
    return values


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summary_path = (root / args.summary_output).resolve()
    variance_path = (root / args.variance_output).resolve()

    summary_payload: dict[str, dict] = {}
    variance_payload: dict[str, dict] = {}

    for name, relative_dir in CONFIGS.items():
        config_dir = root / relative_dir
        seed_dirs = sorted(
            [path for path in config_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()],
            key=seed_sort_key,
        )
        if not seed_dirs:
            continue
        seed_labels = [str(seed_sort_key(path)) for path in seed_dirs]
        seed_summaries = [json.loads((path / "summary.json").read_text()) for path in seed_dirs]
        seed_c_indices = [float(item["c_index_mean"]) for item in seed_summaries]
        seed_aurocs = [float(item["auroc_mean"]) for item in seed_summaries]
        per_seed_per_fold = [fold_c_indices(load_predictions(path)) for path in seed_dirs]
        all_folds = [value for folds in per_seed_per_fold for value in folds]
        within_seed_stds = [float(np.std(folds, ddof=0)) for folds in per_seed_per_fold]

        summary_payload[name] = {
            "n_seeds": len(seed_dirs),
            "seed_dirs": [path.name for path in seed_dirs],
            "seed_mean_c_index": {seed: round(value, 4) for seed, value in zip(seed_labels, seed_c_indices)},
            "seed_mean_auroc": {seed: round(value, 4) for seed, value in zip(seed_labels, seed_aurocs)},
            "mean_c_index": round(float(np.mean(seed_c_indices)), 4),
            "std_across_seeds": round(float(np.std(seed_c_indices, ddof=0)), 4),
            "mean_auroc": round(float(np.mean(seed_aurocs)), 4),
        }
        variance_payload[name] = {
            "mean_across_seeds_and_folds": round(float(np.mean(all_folds)), 4),
            "per_seed_per_fold": [[round(float(value), 4) for value in folds] for folds in per_seed_per_fold],
            "std_within_seed": round(float(np.mean(within_seed_stds)), 4),
            "std_across_seeds": round(float(np.std(seed_c_indices, ddof=0)), 4),
            "std_combined": round(float(np.std(all_folds, ddof=0)), 4),
            "seed_means": {seed: round(value, 4) for seed, value in zip(seed_labels, seed_c_indices)},
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    variance_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    variance_path.write_text(json.dumps(variance_payload, indent=2) + "\n")
    print(f"Saved aggregation summary to {summary_path}")
    print(f"Saved aggregation fold variance to {variance_path}")


if __name__ == "__main__":
    main()
