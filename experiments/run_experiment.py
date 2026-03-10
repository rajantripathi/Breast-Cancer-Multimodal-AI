from __future__ import annotations

import argparse
import subprocess
import sys


TRAINERS = {
    "vision": "training.vision_trainer",
    "ehr": "training.ehr_trainer",
    "genomics": "training.genomics_trainer",
    "literature": "training.literature_trainer",
    "verifier": "training.verifier_trainer",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured experiment")
    parser.add_argument("modality", choices=TRAINERS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    command = [sys.executable, "-m", TRAINERS[args.modality], "--config", args.config]
    if args.smoke_test:
        command.append("--smoke-test")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

