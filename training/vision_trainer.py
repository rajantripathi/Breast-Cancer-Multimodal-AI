from .utils import build_parser, train_text_classifier


def main() -> None:
    args = build_parser("vision").parse_args()
    path = train_text_classifier("vision", args)
    print(f"vision artifact written to {path}")


if __name__ == "__main__":
    main()
