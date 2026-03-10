from __future__ import annotations

import json

from config import load_settings
from agents.vision.foundation_models import list_model_specs

MODELS = {
    "ehr": "emilyalsentzer/Bio_ClinicalBERT",
    "genomics": "zhihan1996/DNA_bert_6",
    "literature": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


def main() -> None:
    """Download configured model snapshots to the Isambard cache."""
    settings = load_settings()
    manifest = {"models": {}, "cache_dir": str(settings.model_cache_dir)}

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None

    for spec in list_model_specs():
        target_dir = spec.cache_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "repo_id": spec.repo_id,
            "target_dir": str(target_dir),
            "status": "manifest_only",
            "gated": spec.gated,
            "embedding_dim": spec.embedding_dim,
        }
        if snapshot_download is not None and settings.hf_token and spec.repo_id != "local/benchmark-stub":
            try:
                snapshot_download(
                    repo_id=spec.repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    token=settings.hf_token,
                    resume_download=True,
                )
                entry["status"] = "downloaded"
            except Exception as exc:  # pragma: no cover
                entry["status"] = "failed"
                entry["error"] = str(exc)
        manifest["models"][f"vision::{spec.key}"] = entry

    for key, repo_id in MODELS.items():
        target_dir = settings.model_cache_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)
        entry = {"repo_id": repo_id, "target_dir": str(target_dir), "status": "manifest_only"}
        if snapshot_download is not None and settings.hf_token:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    token=settings.hf_token,
                    resume_download=True,
                )
                entry["status"] = "downloaded"
            except Exception as exc:  # pragma: no cover
                entry["status"] = "failed"
                entry["error"] = str(exc)
        manifest["models"][key] = entry

    manifest_path = settings.model_cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
