"""
Train a breast-wise multi-view classifier from frozen Mammo-CLIP embeddings.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class BreastEmbeddingDataset(Dataset):
    def __init__(self, metadata_csv, embedding_dir, split="train", embedding_dim=2048):
        self.rows = []
        with open(metadata_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                self.rows.append(row)
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.rows)

    def _load_embedding(self, image_id):
        if image_id is None:
            return torch.zeros(self.embedding_dim, dtype=torch.float32), False
        image_id = str(image_id).strip()
        if not image_id or image_id.lower() == "nan":
            return torch.zeros(self.embedding_dim, dtype=torch.float32), False

        emb_path = self.embedding_dir / f"{image_id}.pt"
        if not emb_path.exists():
            return torch.zeros(self.embedding_dim, dtype=torch.float32), False

        emb = torch.load(emb_path, map_location="cpu").float()
        if emb.ndim > 1:
            emb = emb.flatten()
        return emb, True

    def __getitem__(self, idx):
        row = self.rows[idx]
        cc_emb, cc_valid = self._load_embedding(row.get("cc_image_id"))
        mlo_emb, mlo_valid = self._load_embedding(row.get("mlo_image_id"))
        return {
            "cc_embedding": cc_emb,
            "mlo_embedding": mlo_emb,
            "mask": torch.tensor([cc_valid, mlo_valid], dtype=torch.bool),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "study_id": row["study_id"],
            "breast_id": row["breast_id"],
            "laterality": row["laterality"],
        }


def collate_batch(batch):
    return {
        "cc_embedding": torch.stack([item["cc_embedding"] for item in batch], dim=0),
        "mlo_embedding": torch.stack([item["mlo_embedding"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "study_ids": [item["study_id"] for item in batch],
        "breast_ids": [item["breast_id"] for item in batch],
        "lateralities": [item["laterality"] for item in batch],
    }


class BreastMammoClipClassifier(nn.Module):
    def __init__(self, input_dim=2048, proj_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.view_proj = nn.Linear(input_dim, proj_dim)
        self.attention = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cc_embedding, mlo_embedding, mask):
        cc_proj = self.view_proj(cc_embedding)
        mlo_proj = self.view_proj(mlo_embedding)
        diff_proj = torch.abs(cc_proj - mlo_proj)

        tokens = torch.stack([cc_proj, mlo_proj, diff_proj], dim=1)
        both_views = mask[:, 0] & mask[:, 1]
        token_mask = torch.stack([mask[:, 0], mask[:, 1], both_views], dim=1)

        attn_logits = self.attention(tokens).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~token_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = attn_weights * token_mask.float()
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        fused = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused).squeeze(-1)


def compute_exam_auroc(study_ids, labels, probs):
    exam_rows = {}
    for study_id, label, prob in zip(study_ids, labels, probs):
        item = exam_rows.setdefault(study_id, {"label": label, "prob": prob})
        item["label"] = max(item["label"], label)
        item["prob"] = max(item["prob"], prob)
    exam_labels = [item["label"] for item in exam_rows.values()]
    exam_probs = [item["prob"] for item in exam_rows.values()]
    return float(roc_auc_score(exam_labels, exam_probs))


def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, labels, study_ids = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            cc_embedding = batch["cc_embedding"].to(device)
            mlo_embedding = batch["mlo_embedding"].to(device)
            mask = batch["mask"].to(device)
            label = batch["labels"].to(device)

            logits = model(cc_embedding, mlo_embedding, mask)
            if criterion is not None:
                total_loss += criterion(logits, label).item() * label.size(0)

            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
            study_ids.extend(batch["study_ids"])

    metrics = {
        "breast_auroc": float(roc_auc_score(labels, preds)),
        "exam_auroc": compute_exam_auroc(study_ids, labels, preds),
    }
    if criterion is not None:
        metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/mammography/mammoclip_breast_multiview")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    print("STAGE parsed_args", flush=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("STAGE build_train_dataset", flush=True)
    train_ds = BreastEmbeddingDataset(args.metadata, args.embedding_dir, split="train")
    print("STAGE build_val_dataset", flush=True)
    val_ds = BreastEmbeddingDataset(args.metadata, args.embedding_dir, split="val")
    print("STAGE build_test_dataset", flush=True)
    test_ds = BreastEmbeddingDataset(args.metadata, args.embedding_dir, split="test")
    print("STAGE datasets_ready", flush=True)

    print("STAGE compute_class_balance", flush=True)
    train_labels = [float(row["label"]) for row in train_ds.rows]
    n_neg = sum(1 for label in train_labels if label == 0)
    n_pos = sum(1 for label in train_labels if label == 1)
    pos_weight_value = n_neg / max(n_pos, 1)
    print(f"Class balance: {n_neg} neg, {n_pos} pos, weight={pos_weight_value:.2f}")

    sample_weights = [1.0 if label == 0 else pos_weight_value for label in train_labels]
    print("STAGE build_sampler", flush=True)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("STAGE build_loaders", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    print("STAGE build_model", flush=True)
    model = BreastMammoClipClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print("STAGE start_train", flush=True)

    best_val_exam_auroc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    first_batch_logged = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_preds, train_labels_epoch, train_studies = [], [], []

        for batch in train_loader:
            if not first_batch_logged:
                print("STAGE first_batch_loaded", flush=True)
            cc_embedding = batch["cc_embedding"].to(device)
            mlo_embedding = batch["mlo_embedding"].to(device)
            mask = batch["mask"].to(device)
            label = batch["labels"].to(device)

            if not first_batch_logged:
                print("STAGE first_forward", flush=True)
            logits = model(cc_embedding, mlo_embedding, mask)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            if not first_batch_logged:
                print("STAGE first_backward", flush=True)
            loss.backward()
            optimizer.step()
            if not first_batch_logged:
                print("STAGE first_step_done", flush=True)
                first_batch_logged = True

            total_train_loss += loss.item() * label.size(0)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            train_labels_epoch.extend(label.detach().cpu().numpy().tolist())
            train_studies.extend(batch["study_ids"])

        train_breast_auroc = float(roc_auc_score(train_labels_epoch, train_preds))
        train_exam_auroc = compute_exam_auroc(train_studies, train_labels_epoch, train_preds)
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": total_train_loss / max(len(train_loader.dataset), 1),
            "train_breast_auroc": train_breast_auroc,
            "train_exam_auroc": train_exam_auroc,
            "val_loss": val_metrics["loss"],
            "val_breast_auroc": val_metrics["breast_auroc"],
            "val_exam_auroc": val_metrics["exam_auroc"],
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

        if val_metrics["exam_auroc"] > best_val_exam_auroc:
            best_val_exam_auroc = val_metrics["exam_auroc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False))
    test_metrics = evaluate(model, test_loader, device, criterion=criterion)

    summary = {
        "model": "Breast-wise Mammo-CLIP attention classifier",
        "best_epoch": best_epoch,
        "best_val_exam_auroc": best_val_exam_auroc,
        "test_breast_auroc": test_metrics["breast_auroc"],
        "test_exam_auroc": test_metrics["exam_auroc"],
        "train_breasts": len(train_ds),
        "val_breasts": len(val_ds),
        "test_breasts": len(test_ds),
        "pos_weight": pos_weight_value,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
