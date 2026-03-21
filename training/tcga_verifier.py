from __future__ import annotations

"""Train a real aligned TCGA verifier on vision, genomics, and clinical features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.common import read_json, write_json

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


def _load_tensor(path: str | Path) -> torch.Tensor:
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
    return payload.detach().cpu().float().reshape(-1)


def _fixed_width(values: torch.Tensor, width: int) -> torch.Tensor:
    if values.numel() == width:
        return values
    if values.numel() == 0:
        return torch.zeros(width, dtype=torch.float32)
    pooled = torch.nn.functional.adaptive_avg_pool1d(values.reshape(1, 1, -1), width)
    return pooled.reshape(width)


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
    genomics: torch.Tensor
    clinical: torch.Tensor
    label: int
    survival_time: float
    event_observed: int


class TCGAAlignedDataset(Dataset[TCGASample]):
    def __init__(self, samples: list[TCGASample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TCGASample:
        return self.samples[index]


class TCGAVerifier(nn.Module):
    def __init__(self, vision_dim: int, genomics_dim: int, clinical_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.vision_proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU())
        self.genomics_proj = nn.Sequential(nn.LayerNorm(genomics_dim), nn.Linear(genomics_dim, hidden_dim), nn.GELU())
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vision_proj(vision), self.genomics_proj(genomics), self.clinical_proj(clinical)

    def forward(self, vision: torch.Tensor, genomics: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        vision_token, genomics_token, clinical_token = self._project_modalities(vision, genomics, clinical)
        tokens = torch.stack([vision_token, genomics_token, clinical_token], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        masks = torch.stack(
            [
                (vision.abs().sum(dim=-1) > 0).float(),
                (genomics.abs().sum(dim=-1) > 0).float(),
                (clinical.abs().sum(dim=-1) > 0).float(),
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
    ) -> dict[str, torch.Tensor]:
        vision_token, genomics_token, clinical_token = self._project_modalities(vision, genomics, clinical)
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
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        batch_scores = torch.sigmoid(model(vision, genomics, clinical)).detach().cpu().tolist()
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
    return {
        "sample_id": [sample.sample_id for sample in samples],
        "vision": torch.stack([sample.vision for sample in samples]),
        "genomics": torch.stack([sample.genomics for sample in samples]),
        "clinical": torch.stack([sample.clinical for sample in samples]),
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


def _build_samples(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    stds: pd.Series,
    genomics_dim: int,
    modalities: set[str],
) -> list[TCGASample]:
    samples: list[TCGASample] = []
    for row in frame.to_dict(orient="records"):
        if feature_columns:
            clinical_values = pd.to_numeric(pd.Series({column: row.get(column) for column in feature_columns}), errors="coerce")
            clinical_values = ((clinical_values.fillna(means) - means) / stds).fillna(0.0)
            clinical_tensor = torch.tensor(clinical_values.astype(float).to_numpy(), dtype=torch.float32)
        else:
            clinical_tensor = torch.tensor([0.0], dtype=torch.float32)
        vision_tensor = _fixed_width(_load_tensor(str(row["vision_path"])), 1536)
        genomics_tensor = _fixed_width(_load_tensor(str(row["genomics_path"])), genomics_dim)
        if "vision" not in modalities:
            vision_tensor = torch.zeros_like(vision_tensor)
        if "genomics" not in modalities:
            genomics_tensor = torch.zeros_like(genomics_tensor)
        if "clinical" not in modalities:
            clinical_tensor = torch.zeros_like(clinical_tensor)
        samples.append(
            TCGASample(
                sample_id=str(row["patient_barcode"]),
                vision=vision_tensor,
                genomics=genomics_tensor,
                clinical=clinical_tensor,
                label=int(row["label"]),
                survival_time=float(row["survival_time"]),
                event_observed=int(row["event_observed"]),
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
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        survival_time = batch["survival_time"].to(device)
        event_observed = batch["event_observed"].to(device)
        logits = model(vision, genomics, clinical)
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
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        logits = model(vision, genomics, clinical)
        modality_scores = model.predict_per_modality(vision, genomics, clinical)
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
    crosswalk_path = Path(args.crosswalk)
    clinical_csv = Path(args.clinical_csv)
    endpoint = str(getattr(args, "endpoint", "pfi"))
    survival_horizon_days = float(getattr(args, "survival_horizon_days", DEFAULT_SURVIVAL_HORIZON_DAYS))
    frame, _clinical, feature_columns = _load_aligned_frame(crosswalk_path, clinical_csv, endpoint, survival_horizon_days)
    genomics_metadata = _genomics_metadata(frame)
    modalities = _parse_modalities(getattr(args, "modalities", "vision,clinical,genomics"))

    if frame.empty:
        raise ValueError("TCGA crosswalk produced no aligned samples")
    first_genomics = _load_tensor(str(frame.iloc[0]["genomics_path"]))
    genomics_dim = min(max(128, int(first_genomics.numel())), 1024)
    requested_device = str(args.device).lower()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA requested but unavailable")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed))
    fold_predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
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
        train_samples = _build_samples(train_frame, feature_columns, means, stds, genomics_dim, modalities)
        val_samples = _build_samples(val_frame, feature_columns, means, stds, genomics_dim, modalities)
        test_samples = _build_samples(test_frame, feature_columns, means, stds, genomics_dim, modalities)
        if not train_samples:
            raise ValueError(f"No TCGA training samples available for fold {fold_index}")

        train_loader = DataLoader(TCGAAlignedDataset(train_samples), batch_size=min(16, len(train_samples)), shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(TCGAAlignedDataset(val_samples), batch_size=min(16, max(1, len(val_samples))), shuffle=False, collate_fn=_collate) if val_samples else None
        test_loader = DataLoader(TCGAAlignedDataset(test_samples), batch_size=min(16, max(1, len(test_samples))), shuffle=False, collate_fn=_collate)

        model = TCGAVerifier(vision_dim=1536, genomics_dim=genomics_dim, clinical_dim=len(feature_columns) or 1).to(device)
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

    checkpoint_path = output_dir / "model.pt"
    if final_state is not None:
        torch.save(final_state, checkpoint_path)
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
        "modalities": sorted(modalities),
        "crosswalk_path": str(crosswalk_path),
        "clinical_csv": str(clinical_csv),
        "checkpoint_path": str(checkpoint_path),
        "metrics": {
            "c_index_mean": round(c_index_mean, 4),
            "c_index_std": round(c_index_std, 4),
            "auroc_mean": round(auroc_mean, 4),
            "auroc_std": round(auroc_std, 4),
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
            "loss_function": "cox_nll",
            "missing_modality_handling": "zero_mask_gate",
            "classification_threshold": round(float(final_threshold), 6),
            "cv_folds": 5,
        },
        "fold_metrics": fold_metrics,
        "predictions": fold_predictions,
    }
    write_json(output_dir / "artifact.json", artifact)
    write_json(output_dir / "summary.json", artifact["metrics"])
    write_json(output_dir / "predictions.json", predictions)
    return output_dir / "artifact.json"
