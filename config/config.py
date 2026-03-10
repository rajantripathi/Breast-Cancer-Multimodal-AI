from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


@dataclass
class Settings:
    repo_root: Path
    project_root: Path
    data_root: Path
    raw_data_root: Path
    processed_data_root: Path
    split_root: Path
    model_cache_dir: Path
    hf_cache_dir: Path
    run_root: Path
    artifact_root: Path
    checkpoint_root: Path
    log_root: Path
    output_root: Path
    slurm_account: str = "brics.u6ef"
    hf_token: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _default_root() -> Path:
    """Return the default project root, preferring scratch on clusters."""
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "breast-cancer-multimodal-ai"
    return Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating the input mappings."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load runtime settings from env vars and optional YAML config files."""
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    default_config = repo_root / "config" / "default.yaml"

    project_root = Path(os.getenv("PROJECT_ROOT", _default_root()))
    data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
    settings = Settings(
        repo_root=repo_root,
        project_root=project_root,
        data_root=data_root,
        raw_data_root=data_root / "raw",
        processed_data_root=data_root / "processed",
        split_root=data_root / "splits",
        model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", project_root / "cache" / "models")),
        hf_cache_dir=Path(os.getenv("HF_HOME", project_root / "cache" / "huggingface")),
        run_root=Path(os.getenv("RUN_ROOT", project_root / "runs")),
        artifact_root=Path(os.getenv("ARTIFACT_ROOT", project_root / "artifacts")),
        checkpoint_root=repo_root / "checkpoints",
        log_root=repo_root / "logs",
        output_root=repo_root / "outputs",
        slurm_account=os.getenv("SLURM_ACCOUNT", "brics.u6ef"),
        hf_token=os.getenv("HF_TOKEN", ""),
        kaggle_username=os.getenv("KAGGLE_USERNAME", ""),
        kaggle_key=os.getenv("KAGGLE_KEY", ""),
    )

    loaded_config: dict[str, Any] = {}
    if yaml is not None and default_config.exists():
        loaded_config = yaml.safe_load(default_config.read_text()) or {}
    if config_path:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load experiment configs")
        loaded = yaml.safe_load(Path(config_path).read_text()) or {}
        loaded_config = _deep_merge(loaded_config, loaded)
    settings.extras.update(loaded_config)

    for path in (
        settings.raw_data_root,
        settings.processed_data_root,
        settings.split_root,
        settings.model_cache_dir,
        settings.hf_cache_dir,
        settings.run_root,
        settings.artifact_root,
        settings.checkpoint_root,
        settings.log_root,
        settings.output_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return settings
