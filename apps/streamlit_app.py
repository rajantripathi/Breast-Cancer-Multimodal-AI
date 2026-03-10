from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from apps.utils import load_sample_cases
from orchestrator.run import run_case


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    st.title("Breast Cancer Multimodal AI Demo")
    cases = load_sample_cases(repo_root / "sample_cases")
    case_names = {case["sample_id"]: case for case in cases}
    selected = st.selectbox("Sample case", options=list(case_names))
    if st.button("Run fusion"):
        result = run_case(case_names[selected])
        st.json(result)
        export_path = repo_root / "outputs" / f"{selected}_result.json"
        export_path.write_text(json.dumps(result, indent=2))
        st.caption(f"Saved result to {export_path}")


if __name__ == "__main__":
    main()

