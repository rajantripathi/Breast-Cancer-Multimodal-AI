from __future__ import annotations

"""Shared statistical helpers for classification and survival evaluation."""

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np


def binary_auroc(y_true: Sequence[int], positive_scores: Sequence[float]) -> float:
    positives = [score for score, label in zip(positive_scores, y_true) if int(label) == 1]
    negatives = [score for score, label in zip(positive_scores, y_true) if int(label) == 0]
    if not positives or not negatives:
        return 0.0
    concordant = 0.0
    total = 0
    for pos_score in positives:
        for neg_score in negatives:
            total += 1
            if pos_score > neg_score:
                concordant += 1.0
            elif pos_score == neg_score:
                concordant += 0.5
    return concordant / total if total else 0.0


def harrell_c_index(
    survival_times: Sequence[float],
    risk_scores: Sequence[float],
    event_observed: Sequence[int],
) -> float:
    concordant = 0.0
    admissible = 0
    total = len(survival_times)
    for i in range(total):
        for j in range(i + 1, total):
            t_i, t_j = float(survival_times[i]), float(survival_times[j])
            e_i, e_j = int(event_observed[i]), int(event_observed[j])
            r_i, r_j = float(risk_scores[i]), float(risk_scores[j])
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


def expected_calibration_error(
    y_true: Sequence[int],
    confidences: Sequence[float],
    predictions: Sequence[int],
    num_bins: int = 10,
) -> float:
    if not y_true:
        return 0.0
    ece = 0.0
    for bin_index in range(num_bins):
        start = bin_index / num_bins
        end = (bin_index + 1) / num_bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if confidence >= start and (confidence < end or (bin_index == num_bins - 1 and confidence <= end))
        ]
        if not members:
            continue
        accuracy = sum(int(predictions[index] == y_true[index]) for index in members) / len(members)
        avg_confidence = sum(confidences[index] for index in members) / len(members)
        ece += (len(members) / len(y_true)) * abs(accuracy - avg_confidence)
    return float(ece)


def calibration_bins(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    num_bins: int = 10,
) -> list[dict[str, float | int]]:
    if not y_true:
        return []
    true_array = np.asarray(y_true, dtype=np.int64)
    score_array = np.asarray(y_score, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[dict[str, float | int]] = []
    for idx in range(num_bins):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if idx == num_bins - 1:
            mask = (score_array >= low) & (score_array <= high)
        else:
            mask = (score_array >= low) & (score_array < high)
        if not np.any(mask):
            continue
        bins.append(
            {
                "bin": idx + 1,
                "lower": round(low, 4),
                "upper": round(high, 4),
                "mean_predicted_probability": round(float(score_array[mask].mean()), 4),
                "observed_positive_rate": round(float(true_array[mask].mean()), 4),
                "n_samples": int(mask.sum()),
            }
        )
    return bins


def binary_brier_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    labels = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.size == 0:
        return 0.0
    return float(np.mean((labels - scores) ** 2))


def calibration_slope_intercept(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    eps: float = 1e-6,
) -> dict[str, float | None]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return {"intercept": None, "slope": None}
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"intercept": None, "slope": None}
    logits = np.log(np.clip(scores, eps, 1.0 - eps) / np.clip(1.0 - scores, eps, 1.0))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits.reshape(-1, 1), labels)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def decision_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.size == 0:
        return []
    prevalence = float(labels.mean())
    threshold_grid = thresholds or [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]
    rows: list[dict[str, float]] = []
    for threshold in threshold_grid:
        threshold = float(threshold)
        if threshold <= 0.0 or threshold >= 1.0:
            continue
        preds = (scores >= threshold).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        n = len(labels)
        odds = threshold / (1.0 - threshold)
        net_benefit_model = (tp / n) - (fp / n) * odds
        net_benefit_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": float(net_benefit_model),
                "net_benefit_treat_all": float(net_benefit_all),
                "net_benefit_treat_none": 0.0,
            }
        )
    return rows


def binary_confusion_at_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
) -> dict[str, int | float]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    preds = (scores >= float(threshold)).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {"threshold": float(threshold), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def sensitivity_at_specificity(
    y_true: Sequence[int],
    y_score: Sequence[float],
    target_specificity: float = 0.90,
) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    thresholds = np.unique(scores)
    best_sensitivity = 0.0
    best_threshold = 0.5
    best_gap = float("inf")
    for threshold in thresholds[::-1]:
        confusion = binary_confusion_at_threshold(labels, scores, float(threshold))
        tn = int(confusion["tn"])
        fp = int(confusion["fp"])
        fn = int(confusion["fn"])
        tp = int(confusion["tp"])
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        gap = abs(specificity - target_specificity)
        if gap < best_gap or (gap == best_gap and sensitivity > best_sensitivity):
            best_gap = gap
            best_sensitivity = sensitivity
            best_threshold = float(threshold)
    return float(best_sensitivity), float(best_threshold)


def specificity_at_sensitivity(
    y_true: Sequence[int],
    y_score: Sequence[float],
    target_sensitivity: float = 0.90,
) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    thresholds = np.unique(scores)
    best_specificity = 0.0
    best_threshold = 0.5
    best_gap = float("inf")
    for threshold in thresholds[::-1]:
        confusion = binary_confusion_at_threshold(labels, scores, float(threshold))
        tn = int(confusion["tn"])
        fp = int(confusion["fp"])
        fn = int(confusion["fn"])
        tp = int(confusion["tp"])
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        gap = abs(sensitivity - target_sensitivity)
        if gap < best_gap or (gap == best_gap and specificity > best_specificity):
            best_gap = gap
            best_specificity = specificity
            best_threshold = float(threshold)
    return float(best_specificity), float(best_threshold)


def survival_binary_labels_at_horizon(
    survival_times: Sequence[float],
    event_observed: Sequence[int],
    risk_scores: Sequence[float],
    horizon_days: float,
) -> dict[str, list[float] | list[int] | int]:
    labels: list[int] = []
    scores: list[float] = []
    indices: list[int] = []
    event_count = 0
    for index, (time_value, event_value, score_value) in enumerate(zip(survival_times, event_observed, risk_scores)):
        survival_time = float(time_value)
        event = int(event_value)
        if survival_time < horizon_days and event == 0:
            continue
        label = 1 if (event == 1 and survival_time <= horizon_days) else 0
        labels.append(label)
        scores.append(float(score_value))
        indices.append(index)
        event_count += label
    return {
        "labels": labels,
        "scores": scores,
        "indices": indices,
        "n_eligible": len(labels),
        "n_events": event_count,
    }


def _bootstrap_indices(
    y_true: Sequence[int],
    iterations: int,
    seed: int,
    stratify: bool,
) -> list[np.ndarray]:
    n = len(y_true)
    rng = np.random.RandomState(seed)
    if n == 0:
        return []
    if not stratify:
        return [rng.randint(0, n, size=n) for _ in range(iterations)]

    labels = np.asarray(y_true, dtype=np.int64)
    strata = {int(label): np.where(labels == label)[0] for label in np.unique(labels)}
    if any(len(indices) == 0 for indices in strata.values()):
        return [rng.randint(0, n, size=n) for _ in range(iterations)]
    samples: list[np.ndarray] = []
    for _ in range(iterations):
        parts = [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        samples.append(np.concatenate(parts, axis=0))
    return samples


def bootstrap_confidence_interval(
    metric_fn: Callable[..., float],
    *arrays: Sequence[float | int],
    iterations: int = 1000,
    seed: int = 7,
    stratify: bool = False,
) -> tuple[float, float]:
    if not arrays:
        return 0.0, 0.0
    indices_list = _bootstrap_indices(arrays[0], iterations, seed, stratify)
    values: list[float] = []
    for sample_indices in indices_list:
        sampled = [
            [array[index] for index in sample_indices]
            for array in arrays
        ]
        try:
            value = float(metric_fn(*sampled))
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return 0.0, 0.0
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return lower, upper


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    iterations: int


def paired_bootstrap_test(
    metric_fn: Callable[..., float],
    *arrays: Sequence[float | int],
    model_a: Sequence[float | int],
    model_b: Sequence[float | int],
    iterations: int = 2000,
    seed: int = 7,
    stratify: bool = False,
) -> PairedBootstrapResult:
    if len(model_a) != len(model_b):
        raise ValueError("Paired bootstrap requires equal-length model outputs")
    indices_list = _bootstrap_indices(arrays[0] if arrays else model_a, iterations, seed, stratify)
    deltas: list[float] = []
    for sample_indices in indices_list:
        sampled_arrays = [[array[index] for index in sample_indices] for array in arrays]
        sample_a = [model_a[index] for index in sample_indices]
        sample_b = [model_b[index] for index in sample_indices]
        try:
            metric_a = float(metric_fn(*sampled_arrays, sample_a))
            metric_b = float(metric_fn(*sampled_arrays, sample_b))
        except Exception:
            continue
        delta = metric_a - metric_b
        if math.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        return PairedBootstrapResult(0.0, 0.0, 0.0, 1.0, iterations)
    ordered = sorted(deltas)
    mean_delta = float(sum(deltas) / len(deltas))
    ci_lower = ordered[int(0.025 * (len(ordered) - 1))]
    ci_upper = ordered[int(0.975 * (len(ordered) - 1))]
    p_value = 2.0 * min(
        sum(delta <= 0.0 for delta in deltas) / len(deltas),
        sum(delta >= 0.0 for delta in deltas) / len(deltas),
    )
    return PairedBootstrapResult(mean_delta, float(ci_lower), float(ci_upper), float(min(p_value, 1.0)), iterations)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = [float(value) for value in differences]
    if not values:
        return 1.0
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 2 ** len(values)
    for mask in range(total):
        signed = []
        for index, value in enumerate(values):
            signed.append(-value if (mask >> index) & 1 else value)
        if abs(float(np.mean(signed))) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _fast_delong(predictions_sorted_transposed: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = positive_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    score_matrix = np.vstack([scores_a, scores_b]).astype(np.float64)
    order = np.argsort(-labels)
    label_ordered = labels[order]
    positive_count = int(label_ordered.sum())
    if positive_count == 0 or positive_count == len(label_ordered):
        return {"auc_a": 0.0, "auc_b": 0.0, "z_score": 0.0, "p_value": 1.0}
    aucs, covariance = _fast_delong(score_matrix[:, order], positive_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T)[0, 0])
    if variance <= 0:
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "z_score": 0.0,
            "p_value": 1.0,
        }
    z_score = abs(float(np.diff(aucs)[0])) / math.sqrt(variance)
    p_value = math.erfc(z_score / math.sqrt(2.0))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "z_score": float(z_score),
        "p_value": float(min(max(p_value, 0.0), 1.0)),
    }
