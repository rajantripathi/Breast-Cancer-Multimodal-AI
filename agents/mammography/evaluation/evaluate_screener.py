"""
Evaluation script for mammography screening model.

Computes:
- AUROC
- Sensitivity at 90% specificity
- Specificity at 90% sensitivity
- Confusion matrix
- Per-view attention analysis
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="reports/mammography")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Evaluating screener: {args}")
    print("Implementation pending: train model first")


if __name__ == "__main__":
    main()

