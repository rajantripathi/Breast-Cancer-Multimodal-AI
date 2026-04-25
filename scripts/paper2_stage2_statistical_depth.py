from __future__ import annotations

"""Stage 2 statistical-depth analyses from saved paper artifacts."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "outputs" / "paper"
OUTPUT_DIR = ROOT / "reports" / "paper2" / "stage2_statistics"
OUTPUT_PATH = OUTPUT_DIR / "stage2_statistical_depth.json"


ENCODER_EXPERIMENTS = {
    "CONCH": "conch_ca_vg",
    "CTransPath": "ctranspath_ca_vg",
    "UNI2": "uni2_ca_vg",
}

ABLATION_EXPERIMENTS = {
    "V": "conch_ca_v",
    "V+C": "conch_ca_vc",
    "V+G": "conch_ca_vg",
    "V+C+G": "conch_ca_vcg",
}

TIME_HORIZONS = {
    "2_year": 730.5,
    "3_year": 1095.75,
    "5_year": 1826.25,
}


def load_artifact(name: str) -> dict:
    path = PAPER_DIR / name / "artifact.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text())


def load_predictions(name: str) -> list[dict]:
    artifact = load_artifact(name)
    predictions = artifact.get("predictions", [])
    if not predictions:
        raise RuntimeError(f"No predictions found in {name}/artifact.json")
    return predictions


def harrell_c_index(times: list[float], scores: list[float], events: list[int]) -> float:
    concordant = 0.0
    admissible = 0
    total = len(times)
    for i in range(total):
        for j in range(i + 1, total):
            t_i, t_j = times[i], times[j]
            e_i, e_j = events[i], events[j]
            r_i, r_j = scores[i], scores[j]
            if t_i == t_j and not (e_i or e_j):
                continue
            if t_i < t_j and e_i:
                admissible += 1
                if r_i > r_j:
                    concordant += 1.0
                elif r_i == r_j:
                    concordant += 0.5
            elif t_j < t_i and e_j:
                admissible += 1
                if r_j > r_i:
                    concordant += 1.0
                elif r_i == r_j:
                    concordant += 0.5
    return concordant / admissible if admissible else 0.0


def per_fold_c_indices(predictions: list[dict]) -> tuple[list[float], list[int]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[int(row["fold"])].append(row)
    folds = sorted(grouped)
    values: list[float] = []
    for fold in folds:
        fold_rows = grouped[fold]
        values.append(
            harrell_c_index(
                [float(row["survival_time"]) for row in fold_rows],
                [float(row["risk_score"]) for row in fold_rows],
                [int(row["event_observed"]) for row in fold_rows],
            )
        )
    return values, folds


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    n = len(differences)
    observed = abs(float(np.mean(differences)))
    extreme = 0
    for mask in range(2**n):
        signed = []
        for idx, value in enumerate(differences):
            signed.append(-value if (mask >> idx) & 1 else value)
        if abs(float(np.mean(signed))) >= observed - 1e-12:
            extreme += 1
    return extreme / (2**n)


def paired_significance(name_a: str, name_b: str, folds_a: list[float], folds_b: list[float]) -> dict:
    differences = [a - b for a, b in zip(folds_a, folds_b)]
    try:
        wilcoxon = stats.wilcoxon(folds_a, folds_b, alternative="two-sided", zero_method="wilcox")
        statistic = float(wilcoxon.statistic)
        p_value = float(wilcoxon.pvalue)
    except ValueError:
        statistic = 0.0
        p_value = 1.0
    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "mean_diff": round(float(np.mean(differences)), 4),
        "wilcoxon_statistic": round(statistic, 4),
        "wilcoxon_p": round(p_value, 4),
        "exact_sign_flip_p": round(exact_sign_flip_pvalue(differences), 4),
        "fold_metrics_a": [round(value, 4) for value in folds_a],
        "fold_metrics_b": [round(value, 4) for value in folds_b],
    }


def time_dependent_auc(predictions: list[dict], horizon_days: float) -> dict:
    eligible: list[tuple[int, float]] = []
    for row in predictions:
        survival_time = float(row["survival_time"])
        event_observed = int(row["event_observed"])
        risk_score = float(row["risk_score"])
        if survival_time < horizon_days and event_observed == 0:
            continue
        label = 1 if (event_observed == 1 and survival_time <= horizon_days) else 0
        eligible.append((label, risk_score))

    if not eligible:
        return {"auroc": None, "n_eligible": 0, "n_events": 0}

    y_true = np.asarray([row[0] for row in eligible], dtype=np.int64)
    y_score = np.asarray([row[1] for row in eligible], dtype=np.float64)
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        auroc = None
    else:
        concordant = 0.0
        total = 0
        for pos_score in positives:
            for neg_score in negatives:
                total += 1
                if pos_score > neg_score:
                    concordant += 1.0
                elif pos_score == neg_score:
                    concordant += 0.5
        auroc = concordant / total if total else None

    return {
        "auroc": round(float(auroc), 4) if auroc is not None else None,
        "n_eligible": int(len(eligible)),
        "n_events": int(y_true.sum()),
    }


def normal_survival_prob(z: float) -> float:
    return float(stats.norm.sf(z))


def log_rank_test(times_a: list[float], events_a: list[int], times_b: list[float], events_b: list[int]) -> float:
    all_times = sorted(
        set(
            time
            for time, event in list(zip(times_a, events_a)) + list(zip(times_b, events_b))
            if event == 1
        )
    )
    observed_a = 0.0
    expected_a = 0.0
    variance = 0.0
    for current_time in all_times:
        n_a = sum(1 for time in times_a if time >= current_time)
        n_b = sum(1 for time in times_b if time >= current_time)
        total_at_risk = n_a + n_b
        if total_at_risk <= 1:
            continue
        d_a = sum(1 for time, event in zip(times_a, events_a) if time == current_time and event == 1)
        d_b = sum(1 for time, event in zip(times_b, events_b) if time == current_time and event == 1)
        total_events = d_a + d_b
        if total_events == 0:
            continue
        expected = total_events * n_a / total_at_risk
        observed_a += d_a
        expected_a += expected
        variance += (
            total_events
            * n_a
            * n_b
            * (total_at_risk - total_events)
            / (total_at_risk * total_at_risk * (total_at_risk - 1))
        )
    if variance <= 0:
        return 1.0
    z_score = (observed_a - expected_a) / np.sqrt(variance)
    return 2.0 * normal_survival_prob(abs(z_score))


def calibration_bins(predictions: list[dict], horizon_days: float, n_bins: int = 10) -> list[dict]:
    scores = np.asarray([float(row["probabilities"]["high_concern"]) for row in predictions], dtype=np.float64)
    events = np.asarray([int(row["event_observed"]) for row in predictions], dtype=np.int64)
    times = np.asarray([float(row["survival_time"]) for row in predictions], dtype=np.float64)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
    bins: list[dict] = []
    for idx in range(n_bins):
        low = quantiles[idx]
        high = quantiles[idx + 1]
        if idx == n_bins - 1:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        if not np.any(mask):
            continue
        bin_scores = scores[mask]
        bin_events = events[mask]
        bin_times = times[mask]
        events_in_horizon = int(((bin_events == 1) & (bin_times <= horizon_days)).sum())
        at_risk = int(((bin_events == 1) | (bin_times >= horizon_days)).sum())
        observed_rate = (events_in_horizon / at_risk) if at_risk else 0.0
        bins.append(
            {
                "bin": idx + 1,
                "mean_predicted_probability": round(float(bin_scores.mean()), 4),
                "observed_event_rate": round(float(observed_rate), 4),
                "n_patients": int(mask.sum()),
                "n_events_within_horizon": events_in_horizon,
            }
        )
    return bins


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    encoder_folds: dict[str, list[float]] = {}
    for label, experiment in ENCODER_EXPERIMENTS.items():
        folds, _ = per_fold_c_indices(load_predictions(experiment))
        encoder_folds[label] = folds

    b1_results = {
        f"{left}_vs_{right}": paired_significance(left, right, encoder_folds[left], encoder_folds[right])
        for left, right in [("CONCH", "CTransPath"), ("CONCH", "UNI2"), ("CTransPath", "UNI2")]
    }

    ablation_folds: dict[str, list[float]] = {}
    for label, experiment in ABLATION_EXPERIMENTS.items():
        folds, _ = per_fold_c_indices(load_predictions(experiment))
        ablation_folds[label] = folds

    b2_results = {
        f"{left}_vs_{right}": paired_significance(left, right, ablation_folds[left], ablation_folds[right])
        for left, right in [("V+G", "V"), ("V+C+G", "V"), ("V+C+G", "V+G"), ("V+C", "V")]
    }

    best_predictions = load_predictions("conch_ca_vcg")
    b3_results = {label: time_dependent_auc(best_predictions, horizon) for label, horizon in TIME_HORIZONS.items()}
    for label, experiment in ENCODER_EXPERIMENTS.items():
        b3_results[f"5_year_{label}"] = time_dependent_auc(load_predictions(experiment), TIME_HORIZONS["5_year"])

    b4_results = {
        "horizon_days": TIME_HORIZONS["5_year"],
        "n_bins": 10,
        "calibration_bins": calibration_bins(best_predictions, TIME_HORIZONS["5_year"]),
    }

    scores = np.asarray([float(row["risk_score"]) for row in best_predictions], dtype=np.float64)
    times = np.asarray([float(row["survival_time"]) for row in best_predictions], dtype=np.float64)
    events = np.asarray([int(row["event_observed"]) for row in best_predictions], dtype=np.int64)

    median_score = float(np.median(scores))
    low_mask = scores < median_score
    high_mask = scores >= median_score
    quartiles = np.quantile(scores, [0.25, 0.5, 0.75])
    quartile_groups = np.digitize(scores, quartiles)

    b5_results = {
        "median_split": {
            "median_score": round(median_score, 4),
            "low_risk_n": int(low_mask.sum()),
            "low_risk_event_rate": round(float(events[low_mask].mean()), 4),
            "high_risk_n": int(high_mask.sum()),
            "high_risk_event_rate": round(float(events[high_mask].mean()), 4),
            "log_rank_p": round(
                float(
                    log_rank_test(
                        times[low_mask].tolist(),
                        events[low_mask].tolist(),
                        times[high_mask].tolist(),
                        events[high_mask].tolist(),
                    )
                ),
                4,
            ),
        },
        "quartile_split": {
            "cutpoints": [round(float(value), 4) for value in quartiles],
            "event_rates": [
                round(float(events[quartile_groups == idx].mean()), 4) if np.any(quartile_groups == idx) else 0.0
                for idx in range(4)
            ],
            "group_sizes": [int((quartile_groups == idx).sum()) for idx in range(4)],
        },
    }

    payload = {
        "B1_encoder_pairwise_tests": b1_results,
        "B2_ablation_significance": b2_results,
        "B3_time_dependent_auc": b3_results,
        "B4_survival_calibration": b4_results,
        "B5_alternative_stratification": b5_results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved Stage 2 statistical depth results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
