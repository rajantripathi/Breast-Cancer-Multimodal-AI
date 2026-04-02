"""
Train a lightweight classifier on pre-extracted Mammo-CLIP embeddings.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


class EmbeddingDataset(Dataset):
    """Load pre-extracted Mammo-CLIP embeddings at exam level."""

    def __init__(self, metadata_csv, embedding_dir, split="train"):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.embedding_dir = Path(embedding_dir)
        self.exams = self.df.groupby("study_id").first().reset_index()

    def __len__(self):
        return len(self.exams)

    def __getitem__(self, idx):
        exam = self.exams.iloc[idx]
        study_id = exam["study_id"]
        label = float(exam["label"])

        exam_rows = self.df[self.df["study_id"] == study_id]
        embeddings = []
        for _, row in exam_rows.iterrows():
            emb_path = self.embedding_dir / f"{row['image_id']}.pt"
            if emb_path.exists():
                embeddings.append(torch.load(emb_path, map_location="cpu"))

        if embeddings:
            exam_emb = torch.stack(embeddings).mean(dim=0)
        else:
            exam_emb = torch.zeros(2048, dtype=torch.float32)

        return exam_emb.float(), torch.tensor(label, dtype=torch.float32)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate(model, loader, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for emb, label in loader:
            emb = emb.to(device)
            logit = model(emb)
            preds.extend(torch.sigmoid(logit).cpu().numpy().tolist())
            labels.extend(label.numpy().tolist())
    return float(roc_auc_score(labels, preds))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/mammography/mammoclip")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    labels = [float(train_ds.exams.iloc[i]["label"]) for i in range(len(train_ds))]
    n_neg = sum(1 for label in labels if label == 0)
    n_pos = sum(1 for label in labels if label == 1)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Class balance: {n_neg} neg, {n_pos} pos, weight={pos_weight.item():.2f}")

    model = SimpleClassifier(input_dim=2048).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_auroc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for emb, label in train_loader:
            emb = emb.to(device)
            label = label.to(device)
            loss = criterion(model(emb), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auroc = evaluate(model, val_loader, device)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: val_auroc={val_auroc:.4f} (best={best_val_auroc:.4f})")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_auroc = evaluate(model, test_loader, device)
    print(f"Test AUROC: {test_auroc:.4f}")

    summary = {
        "model": "Mammo-CLIP EfficientNet-B5 + simple classifier",
        "best_val_auroc": best_val_auroc,
        "test_auroc": test_auroc,
        "embedding_dim": 2048,
        "train_exams": len(train_ds),
        "val_exams": len(val_ds),
        "test_exams": len(test_ds),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
