from __future__ import annotations

"""Train a real aligned TCGA verifier on vision, genomics, and clinical features."""

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from agents.vision.mil import AttentionMILPool, TransformerMILPool
from data.common import read_json, write_json
from evaluation.statistics import binary_brier_score, calibration_bins, expected_calibration_error
from training.reproducibility import build_run_manifest, set_global_seed

POSITIVE_VITAL_STATUS = {"dead", "deceased", "1", "true", "yes"}
NEGATIVE_VITAL_STATUS = {"alive", "living", "0", "false", "no"}
DEFAULT_SURVIVAL_HORIZON_DAYS = 1825.0
DEFAULT_CDR_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "tcga_cdr.csv"
DEFAULT_CDR_XLSX_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "tcga_cdr.xlsx"
CLINICAL_EXCLUDE = {
    "clinical_row_idx",
    "days_to_death",
    "days_to_last_followup",
    "vital_status",
}
VALID_MODALITIES = {"vision", "genomics", "clinical"}
VALID_VISION_AGGREGATIONS = {"mean", "abmil", "transmil"}
VALID_GENOMICS_AGGREGATIONS = {"flat", "pathway_tokens"}
VALID_CLINICAL_AGGREGATIONS = {"flat", "embedded"}
CLINICAL_CATEGORICAL_COLUMNS = (
    "gender",
    "tumor_stage",
    "pathologic_stage",
    "er_status_by_ihc",
    "pr_status_by_ihc",
    "her2_status_by_ihc",
    "histological_type",
)
CLINICAL_UNKNOWN_TOKEN = "__UNK__"


def cox_nll_loss(risk_scores: torch.Tensor, survival_times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative log partial likelihood for a Cox proportional hazards model."""
    if risk_scores.numel() == 0:
        return torch.zeros((), device=risk_scores.device, requires_grad=True)
    order = torch.argsort(survival_times, descending=True)
    risk = risk_scores[order]
    event_tensor = events[order].float()
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.mean((risk - log_cumsum) * event_tensor)
    if torch.isnan(loss):
        return torch.zeros((), device=risk.device, requires_grad=True)
    return loss


def _load_tensor_payload(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, dict):
        if "embedding" in payload:
            payload = payload["embedding"]
        elif "tensor" in payload:
            payload = payload["tensor"]
    if isinstance(payload, np.ndarray):
        payload = torch.from_numpy(payload)
    if not isinstance(payload, torch.Tensor):
        payload = torch.tensor(payload)
    return payload.detach().cpu().float()


def _load_tensor(path: str | Path) -> torch.Tensor:
    return _load_tensor_payload(path).reshape(-1)


def _fixed_width(values: torch.Tensor, width: int) -> torch.Tensor:
    if values.numel() == width:
        return values
    if values.numel() == 0:
        return torch.zeros(width, dtype=torch.float32)
    pooled = torch.nn.functional.adaptive_avg_pool1d(values.reshape(1, 1, -1), width)
    return pooled.reshape(width)


def _parse_vision_aggregation(value: Any) -> str:
    aggregation = str(value or "mean").strip().lower()
    if aggregation not in VALID_VISION_AGGREGATIONS:
        raise ValueError(f"Unsupported vision aggregation requested: {aggregation}")
    return aggregation


def _parse_genomics_aggregation(value: Any) -> str:
    aggregation = str(value or "flat").strip().lower()
    if aggregation not in VALID_GENOMICS_AGGREGATIONS:
        raise ValueError(f"Unsupported genomics aggregation requested: {aggregation}")
    return aggregation


def _parse_clinical_aggregation(value: Any) -> str:
    aggregation = str(value or "flat").strip().lower()
    if aggregation not in VALID_CLINICAL_AGGREGATIONS:
        raise ValueError(f"Unsupported clinical aggregation requested: {aggregation}")
    return aggregation


def _infer_patch_vision_path(vision_path: str | Path) -> Path:
    resolved = str(Path(vision_path))
    marker = f"{os.sep}embeddings{os.sep}"
    replacement = f"{os.sep}patch_embeddings{os.sep}"
    if marker not in resolved:
        raise FileNotFoundError(f"Cannot infer patch embedding path from vision path: {vision_path}")
    return Path(resolved.replace(marker, replacement, 1))


def _load_patch_tensor(path: str | Path, width: int) -> torch.Tensor:
    payload = _load_tensor_payload(path)
    if payload.ndim == 1:
        payload = payload.unsqueeze(0)
    elif payload.ndim > 2:
        payload = payload.reshape(payload.shape[0], -1)
    if payload.shape[-1] != width:
        payload = torch.nn.functional.adaptive_avg_pool1d(payload.unsqueeze(1), width).squeeze(1)
    return payload.detach().cpu().float()


def _subsample_bag_instances(bag: torch.Tensor, max_instances: int | None) -> torch.Tensor:
    if max_instances is None or max_instances < 1 or bag.shape[0] <= max_instances:
        return bag
    indices = torch.linspace(0, bag.shape[0] - 1, steps=max_instances).round().long().unique(sorted=True)
    if indices.numel() < max_instances:
        indices = torch.arange(max_instances, dtype=torch.long)
    return bag.index_select(0, indices.clamp(max=bag.shape[0] - 1))


def _normalize_vital_status(value: Any) -> int:
    text = str(value).strip().lower()
    if text in POSITIVE_VITAL_STATUS:
        return 1
    if text in NEGATIVE_VITAL_STATUS:
        return 0
    return 0


def _binary_auroc(labels: list[int], scores: list[float]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    concordant = 0.0
    total = 0
    for pos_score in positives:
        for neg_score in negatives:
            total += 1
            if pos_score > neg_score:
                concordant += 1.0
            elif pos_score == neg_score:
                concordant += 0.5
    return concordant / total if total else 0.0


def _harrell_c_index(survival_times: list[float], risk_scores: list[float], event_observed: list[int]) -> float:
    concordant = 0.0
    admissible = 0
    total = len(survival_times)
    for i in range(total):
        for j in range(i + 1, total):
            t_i, t_j = survival_times[i], survival_times[j]
            e_i, e_j = event_observed[i], event_observed[j]
            r_i, r_j = risk_scores[i], risk_scores[j]
            if t_i == t_j and not (e_i or e_j):
                continue
            if t_i < t_j and e_i:
                admissible += 1
                if r_i > r_j:
                    concordant += 1.0
                elif r_i == r_j:
                    concordant += 0.5
            elif t_j < t_i and e_j:
                admissible += 1
                if r_j > r_i:
                    concordant += 1.0
                elif r_i == r_j:
                    concordant += 0.5
    return concordant / admissible if admissible else 0.0


def _binary_endpoint_label(row: pd.Series, endpoint: str, survival_horizon_days: float) -> int:
    vital_status = str(row.get("vital_status", "")).strip().lower()
    days_to_death = row.get("days_to_death")
    days_to_last_followup = row.get("days_to_last_followup")

    if endpoint == "overall_survival":
        return _normalize_vital_status(vital_status)

    if vital_status in POSITIVE_VITAL_STATUS:
        if pd.notna(days_to_death):
            try:
                return 1 if float(days_to_death) <= survival_horizon_days else 0
            except (TypeError, ValueError):
                return -1
        return -1

    if vital_status in NEGATIVE_VITAL_STATUS:
        if pd.notna(days_to_last_followup):
            try:
                return 0 if float(days_to_last_followup) >= survival_horizon_days else -1
            except (TypeError, ValueError):
                return -1
        return -1

    return -1


def _survival_time(row: pd.Series) -> float:
    if pd.notna(row.get("survival_time")):
        try:
            return max(float(row.get("survival_time")), 0.0)
        except (TypeError, ValueError):
            pass
    for key in (
        "days_to_death",
        "days_to_last_followup",
        "days_to_last_follow_up",
        "days_to_last_followup.1",
    ):
        value = row.get(key)
        if pd.notna(value):
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
    return 0.0


def _clinical_feature_columns(clinical: pd.DataFrame) -> list[str]:
    numeric_cols = clinical.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [column for column in numeric_cols if column not in CLINICAL_EXCLUDE]


def _normalize_clinical_stage(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    if "IV" in text:
        return "STAGE_IV"
    if "III" in text:
        return "STAGE_III"
    if "II" in text:
        return "STAGE_II"
    if "I" in text:
        return "STAGE_I"
    return text.replace(" ", "_")


def _normalize_clinical_receptor(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    if "POS" in text:
        return "POSITIVE"
    if "NEG" in text:
        return "NEGATIVE"
    if "EQUIV" in text:
        return "EQUIVOCAL"
    return text.replace(" ", "_")


def _normalize_clinical_category(column: str, value: Any) -> str:
    if column in {"tumor_stage", "pathologic_stage"}:
        return _normalize_clinical_stage(value)
    if column in {"er_status_by_ihc", "pr_status_by_ihc", "her2_status_by_ihc"}:
        return _normalize_clinical_receptor(value)
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    return text.replace(" ", "_")


def _build_clinical_category_schema(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    schema: dict[str, dict[str, int]] = {}
    for column in CLINICAL_CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            schema[column] = {CLINICAL_UNKNOWN_TOKEN: 0}
            continue
        values = sorted(
            {
                _normalize_clinical_category(column, value)
                for value in frame[column].tolist()
            }
        )
        ordered = [CLINICAL_UNKNOWN_TOKEN] + [value for value in values if value != CLINICAL_UNKNOWN_TOKEN]
        schema[column] = {value: index for index, value in enumerate(ordered)}
    return schema


def _parse_modalities(value: Any) -> set[str]:
    requested = {item.strip().lower() for item in str(value).split(",") if item.strip()}
    if not requested:
        return set(VALID_MODALITIES)
    invalid = requested - VALID_MODALITIES
    if invalid:
        raise ValueError(f"Unsupported modalities requested: {sorted(invalid)}")
    return requested


@dataclass
class TCGASample:
    sample_id: str
    vision: torch.Tensor
    vision_length: int
    genomics: torch.Tensor
    clinical: torch.Tensor
    label: int
    survival_time: float
    event_observed: int
    clinical_categories: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))


class TCGAAlignedDataset(Dataset[TCGASample]):
    def __init__(self, samples: list[TCGASample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TCGASample:
        return self.samples[index]


class PathwayTokenEncoder(nn.Module):
    def __init__(self, num_pathways: int, hidden_dim: int = 256):
        super().__init__()
        self.num_pathways = num_pathways
        self.pathway_embed = nn.Embedding(num_pathways, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, pathway_values: torch.Tensor) -> torch.Tensor:
        batch_size, pathway_count = pathway_values.shape
        if pathway_count != self.num_pathways:
            raise ValueError(f"Expected {self.num_pathways} pathway features, got {pathway_count}")
        pathway_indices = torch.arange(pathway_count, device=pathway_values.device)
        tokens = self.pathway_embed(pathway_indices).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = tokens + self.value_proj(pathway_values.unsqueeze(-1))
        logits = self.attn(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        all_missing = pathway_values.abs().sum(dim=1, keepdim=True) == 0
        pooled = torch.where(all_missing, torch.zeros_like(pooled), pooled)
        return self.output_norm(pooled)


class EmbeddedClinicalEncoder(nn.Module):
    def __init__(self, numeric_dim: int, category_cardinalities: list[int], hidden_dim: int = 256):
        super().__init__()
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden_dim), nn.GELU())
        self.category_embeddings = nn.ModuleList([nn.Embedding(cardinality, hidden_dim) for cardinality in category_cardinalities])
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, numeric_values: torch.Tensor, category_indices: torch.Tensor) -> torch.Tensor:
        tokens = [self.numeric_proj(numeric_values).unsqueeze(1)]
        if self.category_embeddings and category_indices.numel() > 0:
            for index, embedding in enumerate(self.category_embeddings):
                tokens.append(embedding(category_indices[:, index]).unsqueeze(1))
        token_tensor = torch.cat(tokens, dim=1)
        logits = self.attn(token_tensor).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), token_tensor).squeeze(1)
        return self.output_norm(pooled)


class TCGAVerifier(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        genomics_dim: int,
        clinical_dim: int,
        hidden_dim: int = 256,
        vision_aggregation: str = "mean",
        genomics_aggregation: str = "flat",
        clinical_aggregation: str = "flat",
        clinical_category_cardinalities: list[int] | None = None,
    ):
        super().__init__()
        self.vision_aggregation = _parse_vision_aggregation(vision_aggregation)
        self.genomics_aggregation = _parse_genomics_aggregation(genomics_aggregation)
        self.clinical_aggregation = _parse_clinical_aggregation(clinical_aggregation)
        if self.vision_aggregation == "abmil":
            self.vision_pool = AttentionMILPool(vision_dim, hidden_dim=hidden_dim, attention_dim=max(64, hidden_dim // 2))
            self.vision_proj = None
        elif self.vision_aggregation == "transmil":
            self.vision_pool = TransformerMILPool(vision_dim, hidden_dim=hidden_dim, num_heads=4, num_layers=2, dropout=0.1)
            self.vision_proj = None
        else:
            self.vision_pool = None
            self.vision_proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU())
        if self.genomics_aggregation == "pathway_tokens":
            self.genomics_token_encoder = PathwayTokenEncoder(genomics_dim, hidden_dim=hidden_dim)
            self.genomics_proj = None
        else:
            self.genomics_token_encoder = None
            self.genomics_proj = nn.Sequential(nn.LayerNorm(genomics_dim), nn.Linear(genomics_dim, hidden_dim), nn.GELU())
        if self.clinical_aggregation == "embedded":
            cardinalities = clinical_category_cardinalities or [1 for _ in CLINICAL_CATEGORICAL_COLUMNS]
            self.clinical_token_encoder = EmbeddedClinicalEncoder(clinical_dim, cardinalities, hidden_dim=hidden_dim)
            self.clinical_proj = None
        else:
            self.clinical_token_encoder = None
            self.clinical_proj = nn.Sequential(nn.LayerNorm(clinical_dim), nn.Linear(clinical_dim, hidden_dim), nn.GELU())
        self.gate = nn.Linear(hidden_dim, 1)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _project_modalities(
        self,
        vision: torch.Tensor,
        genomics: torch.Tensor,
        clinical: torch.Tensor,
        vision_lengths: torch.Tensor | None = None,
        clinical_categories: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.vision_pool is None:
            vision_token = self.vision_proj(vision)
        else:
            vision_token = self.vision_pool(vision, vision_lengths)
        if self.genomics_token_encoder is None:
            genomics_token = self.genomics_proj(genomics)
        else:
            genomics_token = self.genomics_token_encoder(genomics)
        if self.clinical_token_encoder is None:
            clinical_token = self.clinical_proj(clinical)
        else:
            clinical_categories = clinical_categories if clinical_categories is not None else torch.zeros(
                clinical.shape[0],
                len(self.clinical_token_encoder.category_embeddings),
                dtype=torch.long,
                device=clinical.device,
            )
            clinical_token = self.clinical_token_encoder(clinical, clinical_categories)
        return vision_token, genomics_token, clinical_token

    def _vision_mask(self, vision: torch.Tensor, vision_lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.vision_pool is None:
            return (vision.abs().sum(dim=-1) > 0).float()
        if vision_lengths is None:
            return (vision.abs().sum(dim=(1, 2)) > 0).float()
        return (vision_lengths > 0).float()

    def _clinical_mask(self, clinical: torch.Tensor, clinical_categories: torch.Tensor | None = None) -> torch.Tensor:
        numeric_mask = clinical.abs().sum(dim=-1) > 0
        if clinical_categories is None or clinical_categories.numel() == 0:
            return numeric_mask.float()
        categorical_mask = clinical_categories.sum(dim=-1) > 0
        return (numeric_mask | categorical_mask).float()

    def forward(
        self,
        vision: torch.Tensor,
        genomics: torch.Tensor,
        clinical: torch.Tensor,
        vision_lengths: torch.Tensor | None = None,
        clinical_categories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vision_token, genomics_token, clinical_token = self._project_modalities(
            vision,
            genomics,
            clinical,
            vision_lengths,
            clinical_categories,
        )
        tokens = torch.stack([vision_token, genomics_token, clinical_token], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        masks = torch.stack(
            [
                self._vision_mask(vision, vision_lengths),
                (genomics.abs().sum(dim=-1) > 0).float(),
                self._clinical_mask(clinical, clinical_categories),
            ],
            dim=1,
        )
        gate_logits = self.gate(attn_out).squeeze(-1)
        gate_logits = gate_logits.masked_fill(masks == 0, float("-inf"))
        all_missing = masks.sum(dim=1, keepdim=True) == 0
        gate_logits = torch.where(all_missing, torch.zeros_like(gate_logits), gate_logits)
        gates = torch.softmax(gate_logits, dim=1)
        fused = (attn_out * gates.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused).squeeze(-1)

    def predict_per_modality(
        self,
        vision: torch.Tensor,
        genomics: torch.Tensor,
        clinical: torch.Tensor,
        vision_lengths: torch.Tensor | None = None,
        clinical_categories: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        vision_token, genomics_token, clinical_token = self._project_modalities(
            vision,
            genomics,
            clinical,
            vision_lengths,
            clinical_categories,
        )
        return {
            "vision": torch.sigmoid(self.classifier(vision_token).squeeze(-1)),
            "genomics": torch.sigmoid(self.classifier(genomics_token).squeeze(-1)),
            "clinical": torch.sigmoid(self.classifier(clinical_token).squeeze(-1)),
        }


def _binary_modality_prediction(score: float) -> dict[str, float | str]:
    predicted_label = "high_concern" if score >= 0.5 else "monitor"
    confidence = score if score >= 0.5 else 1.0 - score
    return {
        "class": predicted_label,
        "confidence": round(float(confidence), 4),
    }


def _binary_prediction(score: float, threshold: float) -> tuple[str, float]:
    predicted_label = "high_concern" if score >= threshold else "monitor"
    confidence = score if predicted_label == "high_concern" else 1.0 - score
    return predicted_label, float(confidence)


def _balanced_accuracy_from_scores(labels: list[int], scores: list[float], threshold: float) -> float:
    positives = [index for index, label in enumerate(labels) if label == 1]
    negatives = [index for index, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    predictions = [1 if score >= threshold else 0 for score in scores]
    tpr = sum(predictions[index] == 1 for index in positives) / len(positives)
    tnr = sum(predictions[index] == 0 for index in negatives) / len(negatives)
    return (tpr + tnr) / 2.0


@torch.no_grad()
def _select_classification_threshold(model: TCGAVerifier, loader: DataLoader | None, device: torch.device) -> float:
    if loader is None:
        return 0.5
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for batch in loader:
        vision = batch["vision"].to(device)
        vision_lengths = batch["vision_lengths"].to(device) if batch["vision_lengths"] is not None else None
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        clinical_categories = batch["clinical_categories"].to(device)
        batch_scores = torch.sigmoid(
            model(
                vision,
                genomics,
                clinical,
                vision_lengths=vision_lengths,
                clinical_categories=clinical_categories,
            )
        ).detach().cpu().tolist()
        scores.extend(float(score) for score in batch_scores)
        labels.extend(int(item) for item in batch["label"].detach().cpu().tolist())
    if len(set(labels)) < 2 or not scores:
        return 0.5
    candidates = sorted({round(float(score), 6) for score in scores} | {0.5})
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        bal_acc = _balanced_accuracy_from_scores(labels, scores, threshold)
        if bal_acc > best_score or (bal_acc == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = bal_acc
            best_threshold = threshold
    return float(best_threshold)


def _collate(samples: list[TCGASample]) -> dict[str, Any]:
    is_bag_batch = bool(samples and samples[0].vision.ndim == 2)
    if is_bag_batch:
        vision = pad_sequence([sample.vision for sample in samples], batch_first=True)
        vision_lengths = torch.tensor([sample.vision_length for sample in samples], dtype=torch.long)
    else:
        vision = torch.stack([sample.vision for sample in samples])
        vision_lengths = None
    return {
        "sample_id": [sample.sample_id for sample in samples],
        "vision": vision,
        "vision_lengths": vision_lengths,
        "genomics": torch.stack([sample.genomics for sample in samples]),
        "clinical": torch.stack([sample.clinical for sample in samples]),
        "clinical_categories": torch.stack([sample.clinical_categories for sample in samples]).long(),
        "label": torch.tensor([sample.label for sample in samples], dtype=torch.float32),
        "survival_time": torch.tensor([sample.survival_time for sample in samples], dtype=torch.float32),
        "event_observed": torch.tensor([sample.event_observed for sample in samples], dtype=torch.float32),
    }


def _load_aligned_frame(
    crosswalk_path: Path,
    clinical_csv: Path,
    endpoint: str,
    survival_horizon_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    crosswalk = pd.read_csv(crosswalk_path)
    clinical = pd.read_csv(clinical_csv).copy()
    clinical["clinical_row_idx"] = clinical.index.astype(int)
    feature_columns = _clinical_feature_columns(clinical)
    merged = crosswalk.merge(clinical, on="clinical_row_idx", how="inner", suffixes=("", "_clinical"))
    if {"label", "survival_time", "event_observed"}.issubset(merged.columns):
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
        merged["survival_time"] = pd.to_numeric(merged["survival_time"], errors="coerce")
        merged["event_observed"] = pd.to_numeric(merged["event_observed"], errors="coerce")
        merged = merged[
            merged["label"].notna()
            & merged["survival_time"].notna()
            & merged["event_observed"].notna()
        ].reset_index(drop=True)
        merged["label"] = merged["label"].astype(int)
        merged["event_observed"] = merged["event_observed"].astype(int)
        merged["survival_time"] = merged["survival_time"].astype(float).clip(lower=0.0)
        merged = merged[merged["label"] >= 0].reset_index(drop=True)
        return merged, clinical, feature_columns
    if endpoint == "pfi":
        if DEFAULT_CDR_CSV_PATH.exists():
            cdr = pd.read_csv(DEFAULT_CDR_CSV_PATH)
        elif DEFAULT_CDR_XLSX_PATH.exists():
            cdr = pd.read_excel(DEFAULT_CDR_XLSX_PATH)
        else:
            raise FileNotFoundError(f"Missing TCGA-CDR file: {DEFAULT_CDR_CSV_PATH} or {DEFAULT_CDR_XLSX_PATH}")
        cdr = cdr[cdr["type"] == "BRCA"].copy()
        cdr["bcr_patient_barcode"] = cdr["bcr_patient_barcode"].astype(str).str.upper().str.slice(0, 12)
        cdr = cdr[["bcr_patient_barcode", "PFI", "PFI.time"]].rename(
            columns={"bcr_patient_barcode": "patient_barcode", "PFI": "event_observed", "PFI.time": "survival_time"}
        )
        cdr["event_observed"] = pd.to_numeric(cdr["event_observed"], errors="coerce")
        cdr["survival_time"] = pd.to_numeric(cdr["survival_time"], errors="coerce")
        merged["patient_barcode"] = merged["patient_barcode"].astype(str).str.upper().str.slice(0, 12)
        merged = merged.merge(cdr, on="patient_barcode", how="inner")
        merged = merged[merged["event_observed"].notna() & merged["survival_time"].notna()].reset_index(drop=True)
        merged["label"] = merged["event_observed"].astype(int)
        merged["event_observed"] = merged["event_observed"].astype(int)
        merged["survival_time"] = merged["survival_time"].astype(float).clip(lower=0.0)
    else:
        merged["label"] = merged.apply(lambda row: _binary_endpoint_label(row, endpoint, survival_horizon_days), axis=1).astype(int)
        merged["survival_time"] = merged.apply(_survival_time, axis=1)
        merged["event_observed"] = merged["vital_status"].map(_normalize_vital_status).astype(int)
    merged = merged[merged["label"] >= 0].reset_index(drop=True)
    return merged, clinical, feature_columns


def _split_frame(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame["label"].nunique() < 2 or len(frame) < 10:
        return frame.copy(), frame.iloc[0:0].copy(), frame.iloc[0:0].copy()
    try:
        train_frame, temp_frame = train_test_split(
            frame,
            test_size=0.3,
            random_state=seed,
            stratify=frame["label"],
        )
        val_frame, test_frame = train_test_split(
            temp_frame,
            test_size=0.5,
            random_state=seed,
            stratify=temp_frame["label"],
        )
    except ValueError:
        pos = frame[frame["label"] == 1].sample(frac=1.0, random_state=seed)
        neg = frame[frame["label"] == 0].sample(frac=1.0, random_state=seed)

        n_pos_train = max(1, int(len(pos) * 0.7))
        n_pos_val = max(1, int(len(pos) * 0.15))
        n_neg_train = max(1, int(len(neg) * 0.7))
        n_neg_val = max(1, int(len(neg) * 0.15))

        train_frame = pd.concat([pos.iloc[:n_pos_train], neg.iloc[:n_neg_train]])
        val_frame = pd.concat(
            [
                pos.iloc[n_pos_train : n_pos_train + n_pos_val],
                neg.iloc[n_neg_train : n_neg_train + n_neg_val],
            ]
        )
        test_frame = pd.concat(
            [
                pos.iloc[n_pos_train + n_pos_val :],
                neg.iloc[n_neg_train + n_neg_val :],
            ]
        )

    print(f"Train: {train_frame['label'].value_counts().to_dict()}", flush=True)
    print(f"Val: {val_frame['label'].value_counts().to_dict()}", flush=True)
    print(f"Test: {test_frame['label'].value_counts().to_dict()}", flush=True)
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def _clinical_scaler(train_frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    if not feature_columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    values = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    means = values.mean(axis=0).fillna(0.0)
    stds = values.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return means, stds


def _resolve_device(requested_device: str) -> torch.device:
    requested = str(requested_device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested)


def _last_fold_reference_frames(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    last_fold_index = 0
    dev_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None
    for fold_index, split in enumerate(splitter.split(frame, frame["label"]), start=1):
        last_fold_index = fold_index
        dev_idx, test_idx = split
    if dev_idx is None or test_idx is None:
        raise ValueError("Unable to reconstruct reference fold split")
    dev_frame = frame.iloc[dev_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    if dev_frame["label"].nunique() < 2:
        train_frame = dev_frame.copy()
        val_frame = dev_frame.iloc[0:0].copy()
    else:
        train_frame, val_frame = train_test_split(
            dev_frame,
            test_size=max(0.1, min(0.2, 32 / max(len(dev_frame), 1))),
            random_state=int(seed) + last_fold_index,
            stratify=dev_frame["label"],
        )
        train_frame = train_frame.reset_index(drop=True)
        val_frame = val_frame.reset_index(drop=True)
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def _prediction_rows_to_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        row: dict[str, Any] = {}
        for key, value in prediction.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, sort_keys=True)
            else:
                row[key] = value
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _prediction_calibration(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if item.get("true_label") == "high_concern" else 0 for item in predictions]
    scores = [float(item.get("risk_score", item.get("probabilities", {}).get("high_concern", 0.0))) for item in predictions]
    hard_predictions = [1 if score >= 0.5 else 0 for score in scores]
    confidences = [score if predicted == 1 else 1.0 - score for score, predicted in zip(scores, hard_predictions)]
    return {
        "brier_score": round(float(binary_brier_score(labels, scores)), 6),
        "ece": round(float(expected_calibration_error(labels, confidences, hard_predictions)), 6),
        "bins": calibration_bins(labels, scores),
    }


def _encode_clinical_categories(row: dict[str, Any], category_schema: dict[str, dict[str, int]] | None) -> torch.Tensor:
    if not category_schema:
        return torch.zeros(len(CLINICAL_CATEGORICAL_COLUMNS), dtype=torch.long)
    indices: list[int] = []
    for column in CLINICAL_CATEGORICAL_COLUMNS:
        mapping = category_schema.get(column, {CLINICAL_UNKNOWN_TOKEN: 0})
        normalized = _normalize_clinical_category(column, row.get(column))
        indices.append(int(mapping.get(normalized, 0)))
    return torch.tensor(indices, dtype=torch.long)


def _resolve_patch_vision_path(row: dict[str, Any]) -> Path:
    explicit = row.get("vision_patch_path")
    if explicit is not None and not pd.isna(explicit) and str(explicit).strip():
        return Path(str(explicit))
    return _infer_patch_vision_path(str(row["vision_path"]))


def _build_vision_tensor(
    row: dict[str, Any],
    vision_dim: int,
    modalities: set[str],
    vision_aggregation: str,
    max_vision_instances: int | None,
) -> tuple[torch.Tensor, int]:
    if vision_aggregation == "mean":
        vision_tensor = _fixed_width(_load_tensor(str(row["vision_path"])), vision_dim)
        if "vision" not in modalities:
            vision_tensor = torch.zeros_like(vision_tensor)
        return vision_tensor, 0

    if "vision" not in modalities:
        return torch.zeros((1, vision_dim), dtype=torch.float32), 0

    patch_path = _resolve_patch_vision_path(row)
    patch_tensor = _load_patch_tensor(patch_path, vision_dim)
    patch_tensor = _subsample_bag_instances(patch_tensor, max_vision_instances)
    return patch_tensor, int(patch_tensor.shape[0])


def _build_samples(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    stds: pd.Series,
    vision_dim: int,
    genomics_dim: int,
    modalities: set[str],
    vision_aggregation: str = "mean",
    max_vision_instances: int | None = None,
    category_schema: dict[str, dict[str, int]] | None = None,
) -> list[TCGASample]:
    samples: list[TCGASample] = []
    for row in frame.to_dict(orient="records"):
        if feature_columns:
            clinical_values = pd.to_numeric(pd.Series({column: row.get(column) for column in feature_columns}), errors="coerce")
            clinical_values = ((clinical_values.fillna(means) - means) / stds).fillna(0.0)
            clinical_tensor = torch.tensor(clinical_values.astype(float).to_numpy(), dtype=torch.float32)
        else:
            clinical_tensor = torch.tensor([0.0], dtype=torch.float32)
        clinical_categories = _encode_clinical_categories(row, category_schema)
        vision_tensor, vision_length = _build_vision_tensor(
            row,
            vision_dim=vision_dim,
            modalities=modalities,
            vision_aggregation=vision_aggregation,
            max_vision_instances=max_vision_instances,
        )
        genomics_tensor = _fixed_width(_load_tensor(str(row["genomics_path"])), genomics_dim)
        if "genomics" not in modalities:
            genomics_tensor = torch.zeros_like(genomics_tensor)
        if "clinical" not in modalities:
            clinical_tensor = torch.zeros_like(clinical_tensor)
            clinical_categories = torch.zeros_like(clinical_categories)
        samples.append(
            TCGASample(
                sample_id=str(row["patient_barcode"]),
                vision=vision_tensor,
                vision_length=vision_length,
                genomics=genomics_tensor,
                clinical=clinical_tensor,
                label=int(row["label"]),
                survival_time=float(row["survival_time"]),
                event_observed=int(row["event_observed"]),
                clinical_categories=clinical_categories,
            )
        )
    return samples


def _genomics_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"representation": "unknown", "feature_count": 0}
    genomics_path = Path(str(frame.iloc[0]["genomics_path"]))
    metadata_path = genomics_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            payload = read_json(metadata_path)
            return {
                "representation": payload.get("representation", "unknown"),
                "feature_count": int(payload.get("num_features", 0)),
                "metadata_path": str(metadata_path),
            }
        except Exception:
            pass
    return {"representation": "flat_genes", "feature_count": int(_load_tensor(genomics_path).numel())}


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, float(variance ** 0.5)


def _fold_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(item.get("risk_score", item.get("probabilities", {}).get("high_concern", 0.0))) for item in predictions]
    labels = [1 if item.get("true_label") == "high_concern" else 0 for item in predictions]
    survival_times = [float(item.get("survival_time", 0.0)) for item in predictions]
    event_observed = [int(item.get("event_observed", 0)) for item in predictions]
    return {
        "c_index": round(_harrell_c_index(survival_times, scores, event_observed), 4),
        "auroc": round(_binary_auroc(labels, scores), 4),
        "num_predictions": len(predictions),
        "num_events": int(sum(event_observed)),
    }


def _run_epoch(
    model: TCGAVerifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total_loss = 0.0
    total_examples = 0
    is_train = optimizer is not None
    model.train(is_train)
    for batch in loader:
        vision = batch["vision"].to(device)
        vision_lengths = batch["vision_lengths"].to(device) if batch["vision_lengths"] is not None else None
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        clinical_categories = batch["clinical_categories"].to(device)
        survival_time = batch["survival_time"].to(device)
        event_observed = batch["event_observed"].to(device)
        logits = model(
            vision,
            genomics,
            clinical,
            vision_lengths=vision_lengths,
            clinical_categories=clinical_categories,
        )
        loss = cox_nll_loss(logits, survival_time, event_observed)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_size = len(survival_time)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return total_loss / total_examples if total_examples else 0.0


@torch.no_grad()
def _predict(model: TCGAVerifier, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        vision = batch["vision"].to(device)
        vision_lengths = batch["vision_lengths"].to(device) if batch["vision_lengths"] is not None else None
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        clinical_categories = batch["clinical_categories"].to(device)
        logits = model(
            vision,
            genomics,
            clinical,
            vision_lengths=vision_lengths,
            clinical_categories=clinical_categories,
        )
        modality_scores = model.predict_per_modality(
            vision,
            genomics,
            clinical,
            vision_lengths=vision_lengths,
            clinical_categories=clinical_categories,
        )
        scores = torch.sigmoid(logits).detach().cpu().tolist()
        for index, score in enumerate(scores):
            predicted_label, confidence = _binary_prediction(float(score), threshold)
            true_label = "high_concern" if int(batch["label"][index].item()) == 1 else "monitor"
            modality_predictions = {
                modality: _binary_modality_prediction(float(values[index].detach().cpu().item()))
                for modality, values in modality_scores.items()
            }
            predictions.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "probabilities": {
                        "monitor": round(1.0 - float(score), 6),
                        "high_concern": round(float(score), 6),
                    },
                    "risk_score": round(float(score), 6),
                    "classification_threshold": round(float(threshold), 6),
                    "prediction_confidence": round(float(confidence), 6),
                    "modality_predictions": modality_predictions,
                    "survival_time": float(batch["survival_time"][index].item()),
                    "event_observed": int(batch["event_observed"][index].item()),
                }
            )
    return predictions


def train_tcga_verifier(args: Any, output_dir: Path) -> Path:
    seed_state = set_global_seed(int(args.seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    crosswalk_path = Path(args.crosswalk)
    clinical_csv = Path(args.clinical_csv)
    endpoint = str(getattr(args, "endpoint", "pfi"))
    survival_horizon_days = float(getattr(args, "survival_horizon_days", DEFAULT_SURVIVAL_HORIZON_DAYS))
    vision_aggregation = _parse_vision_aggregation(getattr(args, "vision_aggregation", "mean"))
    genomics_aggregation = _parse_genomics_aggregation(getattr(args, "genomics_aggregation", "flat"))
    clinical_aggregation = _parse_clinical_aggregation(getattr(args, "clinical_aggregation", "flat"))
    max_vision_instances = int(getattr(args, "max_vision_instances", 256))
    if max_vision_instances < 1:
        raise ValueError("--max-vision-instances must be >= 1")
    frame, _clinical, feature_columns = _load_aligned_frame(crosswalk_path, clinical_csv, endpoint, survival_horizon_days)
    clinical_category_schema = _build_clinical_category_schema(frame)
    genomics_metadata = _genomics_metadata(frame)
    if genomics_aggregation == "pathway_tokens" and genomics_metadata.get("representation") != "hallmark_pathways":
        raise ValueError("genomics pathway tokenization requires hallmark_pathways tensors")
    modalities = _parse_modalities(getattr(args, "modalities", "vision,clinical,genomics"))

    if frame.empty:
        raise ValueError("TCGA crosswalk produced no aligned samples")
    first_vision = _load_tensor(str(frame.iloc[0]["vision_path"]))
    vision_dim = int(first_vision.numel())
    first_genomics = _load_tensor(str(frame.iloc[0]["genomics_path"]))
    genomics_dim = min(max(128, int(first_genomics.numel())), 1024)
    device = _resolve_device(str(args.device))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed))
    fold_predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    validation_fold_predictions: list[dict[str, Any]] = []
    validation_fold_metrics: list[dict[str, Any]] = []
    final_state = None
    final_threshold = 0.5
    final_train = final_val = final_test = 0
    for fold_index, (dev_idx, test_idx) in enumerate(splitter.split(frame, frame["label"]), start=1):
        dev_frame = frame.iloc[dev_idx].reset_index(drop=True)
        test_frame = frame.iloc[test_idx].reset_index(drop=True)
        if dev_frame["label"].nunique() < 2:
            train_frame = dev_frame.copy()
            val_frame = dev_frame.iloc[0:0].copy()
        else:
            train_frame, val_frame = train_test_split(
                dev_frame,
                test_size=max(0.1, min(0.2, 32 / max(len(dev_frame), 1))),
                random_state=int(args.seed) + fold_index,
                stratify=dev_frame["label"],
            )
            train_frame = train_frame.reset_index(drop=True)
            val_frame = val_frame.reset_index(drop=True)
        print(f"Fold {fold_index}", flush=True)
        print(f"Train: {train_frame['label'].value_counts().to_dict()}", flush=True)
        print(f"Val: {val_frame['label'].value_counts().to_dict()}", flush=True)
        print(f"Test: {test_frame['label'].value_counts().to_dict()}", flush=True)

        means, stds = _clinical_scaler(train_frame if not train_frame.empty else dev_frame, feature_columns)
        train_samples = _build_samples(
            train_frame,
            feature_columns,
            means,
            stds,
            vision_dim,
            genomics_dim,
            modalities,
            vision_aggregation=vision_aggregation,
            max_vision_instances=max_vision_instances,
            category_schema=clinical_category_schema,
        )
        val_samples = _build_samples(
            val_frame,
            feature_columns,
            means,
            stds,
            vision_dim,
            genomics_dim,
            modalities,
            vision_aggregation=vision_aggregation,
            max_vision_instances=max_vision_instances,
            category_schema=clinical_category_schema,
        )
        test_samples = _build_samples(
            test_frame,
            feature_columns,
            means,
            stds,
            vision_dim,
            genomics_dim,
            modalities,
            vision_aggregation=vision_aggregation,
            max_vision_instances=max_vision_instances,
            category_schema=clinical_category_schema,
        )
        if not train_samples:
            raise ValueError(f"No TCGA training samples available for fold {fold_index}")

        batch_cap = 4 if vision_aggregation != "mean" else 16
        train_loader = DataLoader(TCGAAlignedDataset(train_samples), batch_size=min(batch_cap, len(train_samples)), shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(TCGAAlignedDataset(val_samples), batch_size=min(batch_cap, max(1, len(val_samples))), shuffle=False, collate_fn=_collate) if val_samples else None
        test_loader = DataLoader(TCGAAlignedDataset(test_samples), batch_size=min(batch_cap, max(1, len(test_samples))), shuffle=False, collate_fn=_collate)

        model = TCGAVerifier(
            vision_dim=vision_dim,
            genomics_dim=genomics_dim,
            clinical_dim=len(feature_columns) or 1,
            vision_aggregation=vision_aggregation,
            genomics_aggregation=genomics_aggregation,
            clinical_aggregation=clinical_aggregation,
            clinical_category_cardinalities=[len(clinical_category_schema[column]) for column in CLINICAL_CATEGORICAL_COLUMNS],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
        best_state = None
        best_val_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, int(args.epochs) + 1):
            train_loss = _run_epoch(model, train_loader, device, optimizer)
            val_loss = _run_epoch(model, val_loader, device, None) if val_loader is not None and val_samples else train_loss
            print(f"fold={fold_index} epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(args.patience):
                    print(f"fold={fold_index} early stopping at epoch={epoch}", flush=True)
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        classification_threshold = _select_classification_threshold(model, val_loader if val_loader is not None and val_samples else None, device)
        val_predictions = _predict(model, val_loader, device, threshold=classification_threshold) if val_loader is not None and val_samples else []
        for item in val_predictions:
            item["fold"] = fold_index
        if val_predictions:
            val_metrics = _fold_metrics(val_predictions)
            val_metrics["fold"] = fold_index
            val_metrics["classification_threshold"] = round(float(classification_threshold), 6)
            validation_fold_metrics.append(val_metrics)
            validation_fold_predictions.extend(val_predictions)

        predictions = _predict(model, test_loader, device, threshold=classification_threshold)
        for item in predictions:
            item["fold"] = fold_index
        metrics = _fold_metrics(predictions)
        metrics["fold"] = fold_index
        metrics["classification_threshold"] = round(float(classification_threshold), 6)
        fold_metrics.append(metrics)
        fold_predictions.extend(predictions)
        final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        final_threshold = classification_threshold
        final_train = len(train_samples)
        final_val = len(val_samples)
        final_test = len(test_samples)

    c_index_mean, c_index_std = _mean_std([float(item["c_index"]) for item in fold_metrics])
    auroc_mean, auroc_std = _mean_std([float(item["auroc"]) for item in fold_metrics])
    val_c_index_mean, val_c_index_std = _mean_std([float(item["c_index"]) for item in validation_fold_metrics])
    val_auroc_mean, val_auroc_std = _mean_std([float(item["auroc"]) for item in validation_fold_metrics])

    checkpoint_path = output_dir / "model.pt"
    if final_state is not None:
        torch.save(final_state, checkpoint_path)
    manifest_input_paths = [crosswalk_path, clinical_csv]
    cdr_path = DEFAULT_CDR_CSV_PATH if DEFAULT_CDR_CSV_PATH.exists() else DEFAULT_CDR_XLSX_PATH
    if endpoint == "pfi" and cdr_path.exists():
        manifest_input_paths.append(cdr_path)
    genomics_metadata_path = genomics_metadata.get("metadata_path")
    if genomics_metadata_path:
        manifest_input_paths.append(genomics_metadata_path)
    manifest = build_run_manifest(
        task="tcga_aligned_cross_attention_verifier",
        args=args,
        input_paths=manifest_input_paths,
        split_counts={
            "aligned_samples": int(len(frame)),
            "train_last_fold": int(final_train),
            "val_last_fold": int(final_val),
            "test_last_fold": int(final_test),
        },
        seed_state=seed_state,
        extra={
            "checkpoint_path": str(checkpoint_path),
            "modalities": sorted(modalities),
            "endpoint": endpoint,
            "survival_horizon_days": survival_horizon_days,
            "vision_aggregation": vision_aggregation,
            "genomics_aggregation": genomics_aggregation,
            "clinical_aggregation": clinical_aggregation,
            "max_vision_instances": max_vision_instances,
            "clinical_categorical_columns": list(CLINICAL_CATEGORICAL_COLUMNS),
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    artifact = {
        "task": "verifier",
        "model_name": "tcga_aligned_cross_attention_verifier",
        "device": str(device),
        "alignment_status": "patient_aligned_tcga",
        "aligned_sample_count": int(len(frame)),
        "loss_function": "cox_nll",
        "missing_modality_handling": "zero_mask_gate",
        "endpoint": endpoint,
        "survival_horizon_days": survival_horizon_days,
        "classification_threshold": round(float(classification_threshold), 6),
        "genomics_representation": genomics_metadata.get("representation", "unknown"),
        "genomics_feature_count": genomics_metadata.get("feature_count", int(genomics_dim)),
        "genomics_aggregation": genomics_aggregation,
        "clinical_aggregation": clinical_aggregation,
        "clinical_categorical_columns": list(CLINICAL_CATEGORICAL_COLUMNS),
        "clinical_category_cardinalities": {
            column: len(clinical_category_schema[column]) for column in CLINICAL_CATEGORICAL_COLUMNS
        },
        "vision_feature_count": int(vision_dim),
        "vision_aggregation": vision_aggregation,
        "max_vision_instances": int(max_vision_instances),
        "modalities": sorted(modalities),
        "crosswalk_path": str(crosswalk_path),
        "clinical_csv": str(clinical_csv),
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(output_dir / "manifest.json"),
        "seed_state": seed_state,
        "metrics": {
            "c_index_mean": round(c_index_mean, 4),
            "c_index_std": round(c_index_std, 4),
            "auroc_mean": round(auroc_mean, 4),
            "auroc_std": round(auroc_std, 4),
            "validation_c_index_mean": round(val_c_index_mean, 4),
            "validation_c_index_std": round(val_c_index_std, 4),
            "validation_auroc_mean": round(val_auroc_mean, 4),
            "validation_auroc_std": round(val_auroc_std, 4),
            "num_samples": int(len(frame)),
            "num_folds": 5,
            "num_train_last_fold": int(final_train),
            "num_val_last_fold": int(final_val),
            "num_test_last_fold": int(final_test),
            "alignment_status": "patient_aligned_tcga",
            "aligned_sample_count": int(len(frame)),
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "endpoint": endpoint,
            "survival_horizon_days": survival_horizon_days,
            "vision_aggregation": vision_aggregation,
            "genomics_aggregation": genomics_aggregation,
            "clinical_aggregation": clinical_aggregation,
            "max_vision_instances": int(max_vision_instances),
            "loss_function": "cox_nll",
            "missing_modality_handling": "zero_mask_gate",
            "classification_threshold": round(float(final_threshold), 6),
            "cv_folds": 5,
        },
        "fold_metrics": fold_metrics,
        "validation_fold_metrics": validation_fold_metrics,
        "predictions": fold_predictions,
        "validation_predictions": validation_fold_predictions,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "artifact.json", artifact)
    write_json(output_dir / "summary.json", artifact["metrics"])
    write_json(output_dir / "predictions.json", fold_predictions)
    return output_dir / "artifact.json"


def run_tcga_verifier_inference(args: Any, output_dir: Path) -> Path:
    seed_state = set_global_seed(int(args.seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(str(args.device))
    reference_endpoint = str(getattr(args, "reference_endpoint", args.endpoint))
    reference_horizon = float(getattr(args, "reference_survival_horizon_days", args.survival_horizon_days))
    reference_frame, _, feature_columns = _load_aligned_frame(
        Path(args.reference_crosswalk),
        Path(args.reference_clinical_csv),
        reference_endpoint,
        reference_horizon,
    )
    if reference_frame.empty:
        raise ValueError("Reference TCGA frame produced no samples for inference calibration")
    train_frame, val_frame, _ = _last_fold_reference_frames(reference_frame, int(args.seed))
    means, stds = _clinical_scaler(train_frame if not train_frame.empty else reference_frame, feature_columns)
    clinical_category_schema = _build_clinical_category_schema(reference_frame)
    target_frame, _, _ = _load_aligned_frame(
        Path(args.crosswalk),
        Path(args.clinical_csv),
        str(args.endpoint),
        float(args.survival_horizon_days),
    )
    if target_frame.empty:
        raise ValueError("Target external frame produced no aligned samples")
    modalities = _parse_modalities(getattr(args, "modalities", "vision,clinical,genomics"))
    vision_aggregation = _parse_vision_aggregation(getattr(args, "vision_aggregation", "mean"))
    genomics_aggregation = _parse_genomics_aggregation(getattr(args, "genomics_aggregation", "flat"))
    clinical_aggregation = _parse_clinical_aggregation(getattr(args, "clinical_aggregation", "flat"))
    max_vision_instances = int(getattr(args, "max_vision_instances", 256))

    first_reference_vision = _load_tensor(str(reference_frame.iloc[0]["vision_path"]))
    vision_dim = int(first_reference_vision.numel())
    first_reference_genomics = _load_tensor(str(reference_frame.iloc[0]["genomics_path"]))
    genomics_dim = min(max(128, int(first_reference_genomics.numel())), 1024)

    model = TCGAVerifier(
        vision_dim=vision_dim,
        genomics_dim=genomics_dim,
        clinical_dim=len(feature_columns) or 1,
        vision_aggregation=vision_aggregation,
        genomics_aggregation=genomics_aggregation,
        clinical_aggregation=clinical_aggregation,
        clinical_category_cardinalities=[len(clinical_category_schema[column]) for column in CLINICAL_CATEGORICAL_COLUMNS],
    ).to(device)
    checkpoint_state = torch.load(Path(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint_state)

    val_samples = _build_samples(
        val_frame,
        feature_columns,
        means,
        stds,
        vision_dim,
        genomics_dim,
        modalities,
        vision_aggregation=vision_aggregation,
        max_vision_instances=max_vision_instances,
        category_schema=clinical_category_schema,
    )
    batch_cap = 4 if vision_aggregation != "mean" else 16
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            TCGAAlignedDataset(val_samples),
            batch_size=min(batch_cap, max(1, len(val_samples))),
            shuffle=False,
            collate_fn=_collate,
        )
    classification_threshold = _select_classification_threshold(model, val_loader, device)

    target_samples = _build_samples(
        target_frame,
        feature_columns,
        means,
        stds,
        vision_dim,
        genomics_dim,
        modalities,
        vision_aggregation=vision_aggregation,
        max_vision_instances=max_vision_instances,
        category_schema=clinical_category_schema,
    )
    target_loader = DataLoader(
        TCGAAlignedDataset(target_samples),
        batch_size=min(batch_cap, max(1, len(target_samples))),
        shuffle=False,
        collate_fn=_collate,
    )
    predictions = _predict(model, target_loader, device, threshold=classification_threshold)
    summary = _fold_metrics(predictions)
    summary.update(
        {
            "classification_threshold": round(float(classification_threshold), 6),
            "num_samples": len(predictions),
            "seed": int(args.seed),
        }
    )
    calibration = _prediction_calibration(predictions)
    artifact = {
        "task": "verifier_external_inference",
        "model_name": "tcga_aligned_cross_attention_verifier",
        "device": str(device),
        "checkpoint_path": str(Path(args.checkpoint)),
        "reference_crosswalk": str(Path(args.reference_crosswalk)),
        "reference_clinical_csv": str(Path(args.reference_clinical_csv)),
        "crosswalk_path": str(Path(args.crosswalk)),
        "clinical_csv": str(Path(args.clinical_csv)),
        "endpoint": str(args.endpoint),
        "survival_horizon_days": float(args.survival_horizon_days),
        "modalities": sorted(modalities),
        "vision_aggregation": vision_aggregation,
        "genomics_aggregation": genomics_aggregation,
        "clinical_aggregation": clinical_aggregation,
        "max_vision_instances": max_vision_instances,
        "metrics": summary,
        "calibration": calibration,
        "predictions": predictions,
        "seed_state": seed_state,
    }
    write_json(output_dir / "manifest.json", build_run_manifest(
        task="tcga_verifier_external_inference",
        args=args,
        input_paths=[
            Path(args.checkpoint),
            Path(args.reference_crosswalk),
            Path(args.reference_clinical_csv),
            Path(args.crosswalk),
            Path(args.clinical_csv),
        ],
        split_counts={"reference_samples": int(len(reference_frame)), "target_samples": int(len(target_frame))},
        seed_state=seed_state,
        extra={
            "classification_threshold": round(float(classification_threshold), 6),
            "modalities": sorted(modalities),
            "vision_aggregation": vision_aggregation,
            "genomics_aggregation": genomics_aggregation,
            "clinical_aggregation": clinical_aggregation,
        },
        repo_root=Path(__file__).resolve().parents[1],
    ))
    write_json(output_dir / "artifact.json", artifact)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "calibration.json", calibration)
    write_json(output_dir / "predictions.json", predictions)
    _prediction_rows_to_csv(output_dir / "predictions.csv", predictions)
    return output_dir / "artifact.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCGA cross-attention verifier trainer")
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--clinical-csv", required=True)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference-crosswalk", default=None)
    parser.add_argument("--reference-clinical-csv", default=None)
    parser.add_argument("--reference-endpoint", choices=["overall_survival", "5yr_survival", "pfi"], default=None)
    parser.add_argument("--reference-survival-horizon-days", type=float, default=None)
    parser.add_argument("--modalities", default="vision,clinical,genomics")
    parser.add_argument("--endpoint", choices=["overall_survival", "5yr_survival", "pfi"], default="pfi")
    parser.add_argument("--survival-horizon-days", type=float, default=DEFAULT_SURVIVAL_HORIZON_DAYS)
    parser.add_argument("--vision-aggregation", choices=sorted(VALID_VISION_AGGREGATIONS), default="mean")
    parser.add_argument("--genomics-aggregation", choices=sorted(VALID_GENOMICS_AGGREGATIONS), default="flat")
    parser.add_argument("--clinical-aggregation", choices=sorted(VALID_CLINICAL_AGGREGATIONS), default="flat")
    parser.add_argument("--max-vision-instances", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inference_only:
        if not args.checkpoint or not args.reference_crosswalk or not args.reference_clinical_csv:
            raise ValueError("--checkpoint, --reference-crosswalk, and --reference-clinical-csv are required for --inference-only")
        path = run_tcga_verifier_inference(args, Path(args.output_dir))
    else:
        path = train_tcga_verifier(args, Path(args.output_dir))
    print(f"tcga verifier artifact written to {path}", flush=True)


if __name__ == "__main__":
    main()
