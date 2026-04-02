"""
Train an exam-level Mammo-CLIP classifier from frozen image embeddings.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


VIEW_KEYS = ["lcc", "rcc", "lmlo", "rmlo"]


def normalize_view_name(laterality, view_name):
    laterality = str(laterality or "").strip().lower()
    view_name = str(view_name or "").strip().lower()
    if laterality not in {"l", "r"}:
        return None
    if "cc" in view_name:
        return f"{laterality}cc"
    if "mlo" in view_name:
        return f"{laterality}mlo"
    return None


class EmbeddingDataset(Dataset):
    """Load Mammo-CLIP embeddings grouped at exam level."""

    def __init__(self, metadata_csv, embedding_dir, split="train", embedding_dim=2048):
        df = pd.read_csv(metadata_csv)
        df = df[df["split"] == split].copy()
        df["view_key"] = df.apply(
            lambda row: normalize_view_name(
                row.get("laterality"),
                row.get("view_position") or row.get("view"),
            ),
            axis=1,
        )
        df = df[df["view_key"].notna()].reset_index(drop=True)
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dim = embedding_dim

        self.exams = []
        for study_id, exam_df in df.groupby("study_id"):
            row_by_view = {row["view_key"]: row for _, row in exam_df.iterrows()}
            if not any(view in row_by_view for view in VIEW_KEYS):
                continue
            self.exams.append(
                {
                    "study_id": study_id,
                    "label": float(exam_df.iloc[0]["label"]),
                    "rows": row_by_view,
                }
            )

    def __len__(self):
        return len(self.exams)

    def __getitem__(self, idx):
        exam = self.exams[idx]
        view_embeddings = []
        view_mask = []

        for view in VIEW_KEYS:
            row = exam["rows"].get(view)
            if row is None:
                view_embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
                view_mask.append(False)
                continue

            emb_path = self.embedding_dir / f"{row['image_id']}.pt"
            if emb_path.exists():
                emb = torch.load(emb_path, map_location="cpu").float()
                if emb.ndim > 1:
                    emb = emb.flatten()
                view_embeddings.append(emb)
                view_mask.append(True)
            else:
                view_embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
                view_mask.append(False)

        return {
            "embeddings": torch.stack(view_embeddings, dim=0),
            "mask": torch.tensor(view_mask, dtype=torch.bool),
            "label": torch.tensor(exam["label"], dtype=torch.float32),
            "study_id": exam["study_id"],
        }


class MammoClipAttentionClassifier(nn.Module):
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

    def forward(self, embeddings, mask):
        projected = self.view_proj(embeddings)

        lcc, rcc, lmlo, rmlo = [projected[:, i, :] for i in range(4)]
        cc_diff = torch.abs(lcc - rcc)
        mlo_diff = torch.abs(lmlo - rmlo)

        cc_valid = mask[:, 0] & mask[:, 1]
        mlo_valid = mask[:, 2] & mask[:, 3]

        tokens = torch.cat([projected, cc_diff.unsqueeze(1), mlo_diff.unsqueeze(1)], dim=1)
        token_mask = torch.cat([mask, cc_valid.unsqueeze(1), mlo_valid.unsqueeze(1)], dim=1)

        attn_logits = self.attention(tokens).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~token_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = attn_weights * token_mask.float()
        denom = attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attn_weights = attn_weights / denom

        fused = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits


def collate_batch(batch):
    return {
        "embeddings": torch.stack([item["embeddings"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "study_ids": [item["study_id"] for item in batch],
    }


def evaluate(model, loader, device, criterion=None):
    preds, labels = [], []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            label = batch["labels"].to(device)
            logits = model(embeddings, mask)
            if criterion is not None:
                total_loss += criterion(logits, label).item() * label.size(0)
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())

    metrics = {"auroc": float(roc_auc_score(labels, preds))}
    if criterion is not None:
        metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/mammography/mammoclip_attn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EmbeddingDataset(args.metadata, args.embedding_dir, split="train")
    val_ds = EmbeddingDataset(args.metadata, args.embedding_dir, split="val")
    test_ds = EmbeddingDataset(args.metadata, args.embedding_dir, split="test")

    train_labels = [exam["label"] for exam in train_ds.exams]
    n_neg = sum(1 for label in train_labels if label == 0)
    n_pos = sum(1 for label in train_labels if label == 1)
    pos_weight_value = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Class balance: {n_neg} neg, {n_pos} pos, weight={pos_weight_value:.2f}")

    sample_weights = [1.0 if label == 0 else pos_weight_value for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = MammoClipAttentionClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_auroc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []
        for batch in train_loader:
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            label = batch["labels"].to(device)

            logits = model(embeddings, mask)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * label.size(0)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            train_targets.extend(label.detach().cpu().numpy().tolist())

        train_auroc = float(roc_auc_score(train_targets, train_preds))
        val_metrics = evaluate(model, val_loader, device, criterion=criterion)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": total_train_loss / max(len(train_loader.dataset), 1),
            "train_auroc": train_auroc,
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
        }
        history.append(epoch_summary)

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}: val_auroc={val_metrics['auroc']:.4f} "
            f"(best={best_val_auroc:.4f})"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, criterion=criterion)
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")

    summary = {
        "model": "Mammo-CLIP EfficientNet-B5 + attention fusion classifier",
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "test_auroc": test_metrics["auroc"],
        "embedding_dim": 2048,
        "train_exams": len(train_ds),
        "val_exams": len(val_ds),
        "test_exams": len(test_ds),
        "pos_weight": pos_weight_value,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
