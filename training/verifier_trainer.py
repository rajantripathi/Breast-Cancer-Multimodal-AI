from .utils import build_parser, train_verifier


def main() -> None:
    args = build_parser("verifier").parse_args()
    path = train_verifier(args)
    print(f"verifier artifact written to {path}", flush=True)


if __name__ == "__main__":
    main()
