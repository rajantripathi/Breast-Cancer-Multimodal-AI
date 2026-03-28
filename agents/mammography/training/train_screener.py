"""
Training script for mammography screening model.

Usage:
  python -m agents.mammography.training.train_screener \
    --data-dir data/mammography/vindr-mammo/processed \
    --output-dir outputs/mammography \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 8 \
    --device auto
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/mammography")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    # Training implementation goes here
    # This is the scaffold; full implementation after data download
    print(f"Training screener: {args}")
    print("Implementation pending: download VinDr-Mammo first")
    print("Access: https://physionet.org/content/vindr-mammo/1.0.0/")


if __name__ == "__main__":
    main()

