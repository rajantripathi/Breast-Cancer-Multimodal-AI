from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from agents.vision.mil import AttentionMILPool, TransformerMILPool
from data.common import write_json
from training.reproducibility import build_run_manifest, set_global_seed
from training.tcga_verifier import (
    _binary_auroc,
    _build_samples,
    _build_clinical_category_schema,
    _clinical_scaler,
    _collate,
    _fold_metrics,
    _genomics_metadata,
    _load_aligned_frame,
    _load_tensor,
    _mean_std,
    _parse_clinical_aggregation,
    _parse_genomics_aggregation,
    _parse_modalities,
    _parse_vision_aggregation,
    cox_nll_loss,
    CLINICAL_CATEGORICAL_COLUMNS,
    EmbeddedClinicalEncoder,
    PathwayTokenEncoder,
    TCGAAlignedDataset,
    VALID_CLINICAL_AGGREGATIONS,
    VALID_GENOMICS_AGGREGATIONS,
    VALID_VISION_AGGREGATIONS,
)


class TCGASimpleFusion(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        genomics_dim: int,
        clinical_dim: int,
        hidden_dim: int = 64,
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
            self.vision_pool = AttentionMILPool(vision_dim, hidden_dim=hidden_dim, attention_dim=max(32, hidden_dim // 2))
            self.vision_proj = None
        elif self.vision_aggregation == "transmil":
            self.vision_pool = TransformerMILPool(vision_dim, hidden_dim=hidden_dim, num_heads=4, num_layers=2, dropout=0.1)
            self.vision_proj = None
        else:
            self.vision_pool = None
            self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        if self.genomics_aggregation == "pathway_tokens":
            self.genomics_token_encoder = PathwayTokenEncoder(genomics_dim, hidden_dim=hidden_dim)
            self.genomics_proj = None
        else:
            self.genomics_token_encoder = None
            self.genomics_proj = nn.Linear(genomics_dim, hidden_dim)
        if self.clinical_aggregation == "embedded":
            cardinalities = clinical_category_cardinalities or [1 for _ in CLINICAL_CATEGORICAL_COLUMNS]
            self.clinical_token_encoder = EmbeddedClinicalEncoder(clinical_dim, cardinalities, hidden_dim=hidden_dim)
            self.clinical_proj = None
        else:
            self.clinical_token_encoder = None
            self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def _vision_features(self, vision: torch.Tensor, vision_lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.vision_pool is None:
            return self.vision_proj(vision)
        return self.vision_pool(vision, vision_lengths)

    def _genomics_features(self, genomics: torch.Tensor) -> torch.Tensor:
        if self.genomics_token_encoder is None:
            return self.genomics_proj(genomics)
        return self.genomics_token_encoder(genomics)

    def _clinical_features(self, clinical: torch.Tensor, clinical_categories: torch.Tensor | None = None) -> torch.Tensor:
        if self.clinical_token_encoder is None:
            return self.clinical_proj(clinical)
        clinical_categories = clinical_categories if clinical_categories is not None else torch.zeros(
            clinical.shape[0],
            len(self.clinical_token_encoder.category_embeddings),
            dtype=torch.long,
            device=clinical.device,
        )
        return self.clinical_token_encoder(clinical, clinical_categories)

    def forward(
        self,
        vision: torch.Tensor,
        genomics: torch.Tensor,
        clinical: torch.Tensor,
        vision_lengths: torch.Tensor | None = None,
        clinical_categories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                self._vision_features(vision, vision_lengths),
                self._genomics_features(genomics),
                self._clinical_features(clinical, clinical_categories),
            ],
            dim=-1,
        )
        return self.head(fused).squeeze(-1)


def _run_epoch(model: TCGASimpleFusion, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
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
        logits = model(vision, genomics, clinical, vision_lengths=vision_lengths, clinical_categories=clinical_categories)
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
def _predict(model: TCGASimpleFusion, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        vision = batch["vision"].to(device)
        vision_lengths = batch["vision_lengths"].to(device) if batch["vision_lengths"] is not None else None
        genomics = batch["genomics"].to(device)
        clinical = batch["clinical"].to(device)
        clinical_categories = batch["clinical_categories"].to(device)
        scores = torch.sigmoid(
            model(
                vision,
                genomics,
                clinical,
                vision_lengths=vision_lengths,
                clinical_categories=clinical_categories,
            )
        ).detach().cpu().tolist()
        for index, score in enumerate(scores):
            predictions.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "true_label": "high_concern" if int(batch["label"][index].item()) == 1 else "monitor",
                    "predicted_label": "high_concern" if float(score) >= 0.5 else "monitor",
                    "risk_score": round(float(score), 6),
                    "survival_time": float(batch["survival_time"][index].item()),
                    "event_observed": int(batch["event_observed"][index].item()),
                    "probabilities": {
                        "monitor": round(1.0 - float(score), 6),
                        "high_concern": round(float(score), 6),
                    },
                }
            )
    return predictions


def train_simple_fusion(args: argparse.Namespace, output_dir: Path) -> Path:
    seed_state = set_global_seed(int(args.seed))
    vision_aggregation = _parse_vision_aggregation(getattr(args, "vision_aggregation", "mean"))
    genomics_aggregation = _parse_genomics_aggregation(getattr(args, "genomics_aggregation", "flat"))
    clinical_aggregation = _parse_clinical_aggregation(getattr(args, "clinical_aggregation", "flat"))
    max_vision_instances = int(getattr(args, "max_vision_instances", 256))
    if max_vision_instances < 1:
        raise ValueError("--max-vision-instances must be >= 1")
    frame, _clinical, feature_columns = _load_aligned_frame(
        Path(args.crosswalk), Path(args.clinical_csv), str(args.endpoint), float(args.survival_horizon_days)
    )
    clinical_category_schema = _build_clinical_category_schema(frame)
    if frame.empty:
        raise ValueError("TCGA crosswalk produced no aligned samples")
    modalities = _parse_modalities(args.modalities)
    genomics_metadata = _genomics_metadata(frame)
    if genomics_aggregation == "pathway_tokens" and genomics_metadata.get("representation") != "hallmark_pathways":
        raise ValueError("genomics pathway tokenization requires hallmark_pathways tensors")
    first_vision = _load_tensor(str(frame.iloc[0]["vision_path"]))
    vision_dim = int(first_vision.numel())
    first_genomics = _load_tensor(str(frame.iloc[0]["genomics_path"]))
    genomics_dim = int(first_genomics.numel())
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed))

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

    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    final_state = None
    final_train = final_val = final_test = 0

    for fold_index, (dev_idx, test_idx) in enumerate(splitter.split(frame, frame["label"]), start=1):
        dev_frame = frame.iloc[dev_idx].reset_index(drop=True)
        test_frame = frame.iloc[test_idx].reset_index(drop=True)
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

        means, stds = _clinical_scaler(train_frame, feature_columns)
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

        batch_cap = 4 if vision_aggregation != "mean" else 16
        train_loader = DataLoader(TCGAAlignedDataset(train_samples), batch_size=min(batch_cap, len(train_samples)), shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(TCGAAlignedDataset(val_samples), batch_size=min(batch_cap, len(val_samples)), shuffle=False, collate_fn=_collate)
        test_loader = DataLoader(TCGAAlignedDataset(test_samples), batch_size=min(batch_cap, len(test_samples)), shuffle=False, collate_fn=_collate)

        model = TCGASimpleFusion(
            vision_dim,
            genomics_dim,
            len(feature_columns) or 1,
            vision_aggregation=vision_aggregation,
            genomics_aggregation=genomics_aggregation,
            clinical_aggregation=clinical_aggregation,
            clinical_category_cardinalities=[len(clinical_category_schema[column]) for column in CLINICAL_CATEGORICAL_COLUMNS],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_state = None
        best_val_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, int(args.epochs) + 1):
            train_loss = _run_epoch(model, train_loader, device, optimizer)
            val_loss = _run_epoch(model, val_loader, device, None)
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

        fold_predictions = _predict(model, test_loader, device)
        for item in fold_predictions:
            item["fold"] = fold_index
        metrics = _fold_metrics(fold_predictions)
        metrics["fold"] = fold_index
        fold_metrics.append(metrics)
        predictions.extend(fold_predictions)
        final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        final_train = len(train_samples)
        final_val = len(val_samples)
        final_test = len(test_samples)

    c_index_mean, c_index_std = _mean_std([float(item["c_index"]) for item in fold_metrics])
    auroc_mean, auroc_std = _mean_std([float(item["auroc"]) for item in fold_metrics])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'model.pt'
    if final_state is not None:
        torch.save(final_state, checkpoint_path)
    manifest = build_run_manifest(
        task='tcga_simple_late_fusion',
        args=args,
        input_paths=[Path(args.crosswalk), Path(args.clinical_csv)],
        split_counts={
            'aligned_samples': int(len(frame)),
            'train_last_fold': int(final_train),
            'val_last_fold': int(final_val),
            'test_last_fold': int(final_test),
        },
        seed_state=seed_state,
        extra={
            'checkpoint_path': str(checkpoint_path),
            'modalities': sorted(modalities),
            'endpoint': str(args.endpoint),
            'survival_horizon_days': float(args.survival_horizon_days),
            'vision_aggregation': vision_aggregation,
            'genomics_aggregation': genomics_aggregation,
            'clinical_aggregation': clinical_aggregation,
            'max_vision_instances': max_vision_instances,
            'clinical_categorical_columns': list(CLINICAL_CATEGORICAL_COLUMNS),
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    artifact = {
        'task': 'simple_fusion',
        'model_name': 'tcga_simple_late_fusion',
        'device': str(device),
        'endpoint': str(args.endpoint),
        'survival_horizon_days': float(args.survival_horizon_days),
        'genomics_representation': genomics_metadata.get('representation', 'unknown'),
        'genomics_feature_count': genomics_metadata.get('feature_count', int(genomics_dim)),
        'genomics_aggregation': genomics_aggregation,
        'clinical_aggregation': clinical_aggregation,
        'clinical_categorical_columns': list(CLINICAL_CATEGORICAL_COLUMNS),
        'clinical_category_cardinalities': {
            column: len(clinical_category_schema[column]) for column in CLINICAL_CATEGORICAL_COLUMNS
        },
        'vision_feature_count': int(vision_dim),
        'vision_aggregation': vision_aggregation,
        'max_vision_instances': int(max_vision_instances),
        'modalities': sorted(modalities),
        'manifest_path': str(output_dir / 'manifest.json'),
        'seed_state': seed_state,
        'metrics': {
            'c_index_mean': round(c_index_mean, 4),
            'c_index_std': round(c_index_std, 4),
            'auroc_mean': round(auroc_mean, 4),
            'auroc_std': round(auroc_std, 4),
            'num_samples': int(len(frame)),
            'num_folds': 5,
            'num_train_last_fold': int(final_train),
            'num_val_last_fold': int(final_val),
            'num_test_last_fold': int(final_test),
        },
        'hyperparameters': {
            'epochs': int(args.epochs),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'patience': int(args.patience),
            'seed': int(args.seed),
            'vision_aggregation': vision_aggregation,
            'genomics_aggregation': genomics_aggregation,
            'clinical_aggregation': clinical_aggregation,
            'max_vision_instances': int(max_vision_instances),
        },
        'fold_metrics': fold_metrics,
        'predictions': predictions,
    }
    write_json(output_dir / 'manifest.json', manifest)
    write_json(output_dir / 'artifact.json', artifact)
    write_json(output_dir / 'summary.json', artifact['metrics'])
    write_json(output_dir / 'predictions.json', predictions)
    return output_dir / 'artifact.json'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='TCGA simple fusion trainer')
    parser.add_argument('--crosswalk', required=True)
    parser.add_argument('--clinical-csv', required=True)
    parser.add_argument('--modalities', default='vision,clinical,genomics')
    parser.add_argument('--endpoint', choices=['overall_survival', '5yr_survival', 'pfi'], default='pfi')
    parser.add_argument('--survival-horizon-days', type=float, default=1825.0)
    parser.add_argument('--vision-aggregation', choices=sorted(VALID_VISION_AGGREGATIONS), default='mean')
    parser.add_argument('--genomics-aggregation', choices=sorted(VALID_GENOMICS_AGGREGATIONS), default='flat')
    parser.add_argument('--clinical-aggregation', choices=sorted(VALID_CLINICAL_AGGREGATIONS), default='flat')
    parser.add_argument('--max-vision-instances', type=int, default=256)
    parser.add_argument('--cv-folds', type=int, default=5, help='Compatibility flag; simple fusion currently runs 5-fold CV')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = train_simple_fusion(args, Path(args.output_dir))
    print(f'simple fusion artifact written to {path}', flush=True)


if __name__ == '__main__':
    main()
