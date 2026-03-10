from .utils import build_parser, train_text_classifier


def main() -> None:
    args = build_parser("ehr").parse_args()
    path = train_text_classifier("ehr", args)
    print(f"ehr artifact written to {path}")


if __name__ == "__main__":
    main()
