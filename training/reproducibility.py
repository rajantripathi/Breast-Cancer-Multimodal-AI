from __future__ import annotations

"""Shared helpers for deterministic training and run-manifest capture."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and Torch, and optionally enable deterministic backends."""
    random.seed(seed)
    np.random.seed(seed)

    state: dict[str, Any] = {
        "seed": int(seed),
        "deterministic_requested": bool(deterministic),
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    if torch is None:
        state["torch_available"] = False
        return state

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    state["torch_available"] = True

    if not deterministic:
        return state

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        state["torch_deterministic_algorithms"] = True
    except Exception as exc:  # pragma: no cover - environment dependent
        state["torch_deterministic_algorithms_error"] = str(exc)

    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        except Exception as exc:  # pragma: no cover - environment dependent
            state["cudnn_error"] = str(exc)
    return state


def get_git_commit(repo_root: str | Path | None = None) -> str:
    target = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        return (
            subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_status(repo_root: str | Path | None = None) -> list[str]:
    target = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        output = subprocess.check_output(
            ["git", "-C", str(target), "status", "--short"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def hash_path(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if not target.exists():
        return "missing"
    if target.is_file():
        digest.update(target.read_bytes())
        return digest.hexdigest()
    for item in sorted(p for p in target.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(target)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def summarize_inputs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for raw_path in paths:
        target = Path(raw_path)
        summary.append(
            {
                "path": str(target),
                "exists": target.exists(),
                "is_file": target.is_file(),
                "sha256": hash_path(target) if target.exists() else "missing",
            }
        )
    return summary


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch_version": getattr(torch, "__version__", "unavailable") if torch is not None else "unavailable",
        "numpy_version": getattr(np, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
        "cuda_device_count": int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0,
    }
    if torch is not None and torch.cuda.is_available():
        snapshot["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return snapshot


def _normalise_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return {key: value for key, value in vars(args).items()}
    raise TypeError("args must be a dict or argparse Namespace-like object")


def build_run_manifest(
    *,
    task: str,
    args: Any,
    input_paths: Iterable[str | Path],
    split_counts: dict[str, int] | None = None,
    seed_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    target_repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    manifest = {
        "task": task,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(target_repo),
        "git_status_short": get_git_status(target_repo),
        "repo_root": str(target_repo),
        "args": _normalise_args(args),
        "input_paths": summarize_inputs(input_paths),
        "split_counts": split_counts or {},
        "seed_state": seed_state or {},
        "environment": environment_snapshot(),
    }
    if extra:
        manifest.update(extra)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return manifest
