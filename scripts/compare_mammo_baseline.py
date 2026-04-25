#!/usr/bin/env python3
"""
Compare a mammography run summary/history against the canonical historical baseline.

This is intended for Stage 1 reproducibility work so new runs can be checked
against the retained 224px ConvNeXt benchmark without ad hoc notebook work.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_BASELINE_SUMMARY = Path("outputs/mammography/summary_baseline_224.json")
DEFAULT_BASELINE_HISTORY = Path("outputs/mammography/history_baseline_224.json")


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_delta(current, baseline):
    if current is None or baseline is None:
        return "n/a"
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.6f}"


def close_enough(current, baseline, tolerance):
    if current is None or baseline is None:
        return False
    return abs(current - baseline) <= tolerance


def best_epoch_entry(history):
    if not history:
        return None
    return max(history, key=lambda row: row.get("val_auroc", float("-inf")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Summary JSON for the run under evaluation")
    parser.add_argument("--history", help="Optional history JSON for the run under evaluation")
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--baseline-history", default=str(DEFAULT_BASELINE_HISTORY))
    parser.add_argument("--auroc-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    summary = load_json(Path(args.summary))
    baseline_summary = load_json(Path(args.baseline_summary))

    history = load_json(Path(args.history)) if args.history else None
    baseline_history = load_json(Path(args.baseline_history)) if args.baseline_history else None

    current_test_auroc = maybe_float(summary.get("test_auroc"))
    baseline_test_auroc = maybe_float(baseline_summary.get("test_auroc"))
    current_val_auroc = maybe_float(summary.get("best_val_auroc"))
    baseline_val_auroc = maybe_float(baseline_summary.get("best_val_auroc"))

    print("Stage 1 Baseline Comparison")
    print(f"run_summary={Path(args.summary)}")
    print(f"baseline_summary={Path(args.baseline_summary)}")
    print("")
    print(
        f"best_val_auroc: run={current_val_auroc:.6f} "
        f"baseline={baseline_val_auroc:.6f} "
        f"delta={format_delta(current_val_auroc, baseline_val_auroc)}"
    )
    print(
        f"test_auroc:     run={current_test_auroc:.6f} "
        f"baseline={baseline_test_auroc:.6f} "
        f"delta={format_delta(current_test_auroc, baseline_test_auroc)}"
    )
    print(
        f"best_epoch:     run={summary.get('best_epoch')} "
        f"baseline={baseline_summary.get('best_epoch')}"
    )
    print(
        f"image_size:     run={summary.get('image_size')} "
        f"baseline={baseline_summary.get('image_size')}"
    )

    if history and baseline_history:
        run_best = best_epoch_entry(history)
        baseline_best = best_epoch_entry(baseline_history)
        if run_best and baseline_best:
            print("")
            print("Best-Epoch Trace")
            print(
                f"run_epoch={run_best.get('epoch')} "
                f"train_loss={run_best.get('train_loss'):.6f} "
                f"val_loss={run_best.get('val_loss'):.6f} "
                f"val_auroc={run_best.get('val_auroc'):.6f}"
            )
            print(
                f"baseline_epoch={baseline_best.get('epoch')} "
                f"train_loss={baseline_best.get('train_loss'):.6f} "
                f"val_loss={baseline_best.get('val_loss'):.6f} "
                f"val_auroc={baseline_best.get('val_auroc'):.6f}"
            )

    print("")
    if close_enough(current_test_auroc, baseline_test_auroc, args.auroc_tolerance):
        print(f"status=PASS test_auroc within ±{args.auroc_tolerance:.3f} of baseline")
    else:
        print(f"status=FAIL test_auroc differs by more than ±{args.auroc_tolerance:.3f}")


if __name__ == "__main__":
    main()
