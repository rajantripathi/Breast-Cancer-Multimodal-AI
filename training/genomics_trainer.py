from .utils import build_parser, train_text_classifier


def main() -> None:
    args = build_parser("genomics").parse_args()
    path = train_text_classifier("genomics", args)
    print(f"genomics artifact written to {path}")


if __name__ == "__main__":
    main()
