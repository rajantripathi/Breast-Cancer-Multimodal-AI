from __future__ import annotations

from pathlib import Path

from .tcga_simple_fusion import build_parser, train_simple_fusion


def main() -> None:
    args = build_parser().parse_args()
    path = train_simple_fusion(args, Path(args.output_dir))
    print(f"simple fusion artifact written to {path}", flush=True)


if __name__ == "__main__":
    main()
