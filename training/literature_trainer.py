from .utils import build_parser, train_text_classifier


def main() -> None:
    args = build_parser("literature").parse_args()
    path = train_text_classifier("literature", args)
    print(f"literature artifact written to {path}")


if __name__ == "__main__":
    main()
