"""
Train a breast-wise multi-view mammography classifier and derive exam-level AUROC.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from agents.mammography.models.breast_multiview_model import BreastMultiViewClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BreastDataset(Dataset):
    def __init__(self, metadata_csv, split="train", image_size=456, training=False):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.training = training
        self.image_size = image_size
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _load_png(self, path_str):
        path = Path(str(path_str or ""))
        if not path.exists():
            return None
        img = Image.open(path).convert("RGB")
        return img

    def _augment(self, image):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        angle = random.uniform(-10.0, 10.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        arr *= brightness
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cc_img = self._load_png(row.get("cc_png_path"))
        mlo_img = self._load_png(row.get("mlo_png_path"))
        mask = [cc_img is not None, mlo_img is not None]

        blank = Image.new("RGB", (self.image_size, self.image_size))
        cc_img = blank if cc_img is None else cc_img
        mlo_img = blank if mlo_img is None else mlo_img

        if self.training:
            cc_img = self._augment(cc_img)
            mlo_img = self._augment(mlo_img)

        cc = self.base_transform(cc_img)
        mlo = self.base_transform(mlo_img)

        return {
            "cc": cc,
            "mlo": mlo,
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "study_id": row["study_id"],
            "breast_id": row["breast_id"],
        }


def collate_batch(batch):
    return {
        "cc": torch.stack([item["cc"] for item in batch], dim=0),
        "mlo": torch.stack([item["mlo"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "study_ids": [item["study_id"] for item in batch],
        "breast_ids": [item["breast_id"] for item in batch],
    }


def compute_exam_auroc(study_ids, labels, probs):
    exam_rows = {}
    for study_id, label, prob in zip(study_ids, labels, probs):
        item = exam_rows.setdefault(study_id, {"label": label, "prob": prob})
        item["prob"] = max(item["prob"], prob)
        item["label"] = max(item["label"], label)
    exam_labels = [item["label"] for item in exam_rows.values()]
    exam_probs = [item["prob"] for item in exam_rows.values()]
    return float(roc_auc_score(exam_labels, exam_probs))


def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, labels, study_ids = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            cc = batch["cc"].to(device)
            mlo = batch["mlo"].to(device)
            mask = batch["mask"].to(device)
            target = batch["labels"].to(device)

            logits = model(cc, mlo, mask)
            if criterion is not None:
                total_loss += criterion(logits, target).item() * target.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            preds.extend(probs)
            labels.extend(target.cpu().numpy().tolist())
            study_ids.extend(batch["study_ids"])

    return {
        "breast_auroc": float(roc_auc_score(labels, preds)),
        "exam_auroc": compute_exam_auroc(study_ids, labels, preds),
        "loss": None if criterion is None else total_loss / max(len(loader.dataset), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="outputs/mammography/vindr_breast_multiview")
    parser.add_argument("--backbone-name", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=456)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BreastDataset(args.metadata, split="train", image_size=args.image_size, training=True)
    val_ds = BreastDataset(args.metadata, split="val", image_size=args.image_size, training=False)
    test_ds = BreastDataset(args.metadata, split="test", image_size=args.image_size, training=False)

    train_labels = train_ds.df["label"].astype(float).tolist()
    n_neg = sum(1 for label in train_labels if label == 0)
    n_pos = sum(1 for label in train_labels if label == 1)
    if n_pos == 0:
        raise RuntimeError("No positive breast samples found in training split.")
    pos_weight_value = n_neg / n_pos
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
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = BreastMultiViewClassifier(backbone_name=args.backbone_name, pretrained=True).to(device)
    if args.backbone_checkpoint:
        model.encoder.load_backbone_checkpoint(args.backbone_checkpoint)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_exam_auroc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.freeze_backbone(epoch <= args.freeze_epochs)
        total_train_loss = 0.0
        train_preds, train_labels_epoch, train_studies = [], [], []
        for batch in train_loader:
            cc = batch["cc"].to(device)
            mlo = batch["mlo"].to(device)
            mask = batch["mask"].to(device)
            target = batch["labels"].to(device)

            logits = model(cc, mlo, mask)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * target.size(0)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            train_labels_epoch.extend(target.detach().cpu().numpy().tolist())
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

        if val_metrics["exam_auroc"] > best_exam_auroc:
            best_exam_auroc = val_metrics["exam_auroc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_exam_auroc": best_exam_auroc,
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion=criterion)

    summary = {
        "model": "Breast-wise multi-view classifier",
        "best_epoch": best_epoch,
        "best_val_exam_auroc": best_exam_auroc,
        "test_breast_auroc": test_metrics["breast_auroc"],
        "test_exam_auroc": test_metrics["exam_auroc"],
        "train_breasts": len(train_ds),
        "val_breasts": len(val_ds),
        "test_breasts": len(test_ds),
        "pos_weight": pos_weight_value,
        "backbone_name": args.backbone_name,
        "backbone_checkpoint": args.backbone_checkpoint,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
