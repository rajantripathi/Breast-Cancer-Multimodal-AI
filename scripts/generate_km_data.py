from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def risk_group_tertiles(predictions: list[dict[str, Any]]) -> tuple[dict[str, list[tuple[str, float, float, int]]], list[tuple[str, float, int, int]]]:
    ranked = sorted(
        [
            (
                str(item.get("sample_id", "")),
                float(item.get("risk_score", item.get("probabilities", {}).get("high_concern", 0.0))),
                float(item.get("survival_time", 0.0)),
                int(item.get("event_observed", 0)),
            )
            for item in predictions
        ],
        key=lambda row: row[1],
    )
    total = len(ranked)
    cut1 = total // 3
    cut2 = (2 * total) // 3
    groups = {
        "low_risk": ranked[:cut1],
        "mid_risk": ranked[cut1:cut2],
        "high_risk": ranked[cut2:],
    }
    flat_assignments: list[tuple[str, float, int, int]] = []
    code_map = {"low_risk": 0, "mid_risk": 1, "high_risk": 2}
    for name, rows in groups.items():
        flat_assignments.extend((sample_id, survival_time, event_observed, code_map[name]) for sample_id, _, survival_time, event_observed in rows)
    return groups, flat_assignments


def kaplan_meier_curve(rows: list[tuple[str, float, float, int]]) -> dict[str, Any]:
    n = len(rows)
    event_count = int(sum(int(row[3]) for row in rows))
    if not rows:
        return {"times": [], "events": [], "survival": [], "n": 0, "event_count": 0}

    event_times = sorted({float(row[2]) for row in rows if int(row[3]) == 1})
    at_risk = n
    survival_prob = 1.0
    times: list[float] = []
    events: list[int] = []
    survival: list[float] = []

    for event_time in event_times:
        d_i = sum(1 for _, _, time, event in rows if int(event) == 1 and float(time) == event_time)
        n_i = sum(1 for _, _, time, _ in rows if float(time) >= event_time)
        if n_i <= 0:
            continue
        survival_prob *= (1.0 - (d_i / n_i))
        times.append(round(float(event_time), 4))
        events.append(int(d_i))
        survival.append(round(float(survival_prob), 6))

    return {
        "times": times,
        "events": events,
        "survival": survival,
        "n": n,
        "event_count": event_count,
    }


def logrank_p_value(assignments: list[tuple[str, float, int, int]]) -> float | None:
    if len(assignments) < 3:
        return None

    groups = sorted({group for _, _, _, group in assignments})
    if len(groups) < 2:
        return None

    event_times = sorted({time for _, time, event, _ in assignments if event == 1})
    if not event_times:
        return None

    observed = {group: 0.0 for group in groups}
    expected = {group: 0.0 for group in groups}
    variances = {group: 0.0 for group in groups}

    for event_time in event_times:
        at_risk = {group: 0 for group in groups}
        events = {group: 0 for group in groups}
        for _, time, event, group in assignments:
            if time >= event_time:
                at_risk[group] += 1
            if event == 1 and time == event_time:
                events[group] += 1
        total_at_risk = sum(at_risk.values())
        total_events = sum(events.values())
        if total_at_risk <= 1 or total_events == 0:
            continue
        for group in groups:
            observed[group] += events[group]
            expected_events = at_risk[group] * total_events / total_at_risk
            expected[group] += expected_events
            if total_at_risk > 1:
                variances[group] += (
                    at_risk[group]
                    * (total_at_risk - at_risk[group])
                    * total_events
                    * (total_at_risk - total_events)
                    / (total_at_risk**2 * (total_at_risk - 1))
                )

    chi_square = 0.0
    valid_groups = 0
    for group in groups:
        variance = variances[group]
        if variance > 0:
            chi_square += (observed[group] - expected[group]) ** 2 / variance
            valid_groups += 1
    if valid_groups < 2:
        return None

    degrees_of_freedom = valid_groups - 1
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(chi_square, degrees_of_freedom))
    except Exception:
        if degrees_of_freedom == 2:
            return float(math.exp(-chi_square / 2.0))
        return None


def build_km_data(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups, assignments = risk_group_tertiles(predictions)
    payload = {name: kaplan_meier_curve(rows) for name, rows in groups.items()}
    payload["log_rank_p"] = logrank_p_value(assignments)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaplan-Meier curve data from prediction artifacts")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json from a 5-fold CV run")
    parser.add_argument("--output", default="reports/paper/km_data.json", help="Output JSON path")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_path = Path(args.output)
    predictions = read_json(predictions_path)
    payload = build_km_data(predictions)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
