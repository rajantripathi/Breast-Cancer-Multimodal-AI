"""
Training script for mammography screening model.

Recovered from commit f209755, which produced the tracked
`outputs/mammography/summary.json` benchmark at test AUROC 0.7407.

Usage:
  python -m agents.mammography.training.train_screener \
    --data-dir data/mammography/vindr-mammo/processed \
    --output-dir outputs/mammography \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 8 \
    --device auto
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from agents.mammography.models.screening_model_legacy import LegacyMammographyScreener as MammographyScreener

try:
    import pydicom
    from PIL import Image
except ImportError:
    pydicom = None
    Image = None


VIEW_KEYS = ["lcc", "rcc", "lmlo", "rmlo"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/mammography")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint to evaluate or resume from. Defaults to <output-dir>/best_model.pt.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate the provided checkpoint on the test split.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


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


def build_exam_records(metadata_path, raw_dir):
    records = {}
    raw_dir = Path(raw_dir)
    archive_root = raw_dir / (
        "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
    )
    for row in csv.DictReader(metadata_path.open()):
        study_id = row["study_id"]
        image_id = row["image_id"]
        split = row["split"]
        label = int(row["label"])
        key = normalize_view_name(row.get("laterality"), row.get("view_position") or row.get("view"))
        if key is None:
            continue

        dicom_path = raw_dir / "images" / study_id / f"{image_id}.dicom"
        if not dicom_path.exists():
            dicom_path = archive_root / "images" / study_id / f"{image_id}.dicom"
        if not dicom_path.exists():
            continue

        exam = records.setdefault(
            study_id,
            {"study_id": study_id, "split": split, "label": label, "views": {}},
        )
        exam["views"][key] = dicom_path

    exams = []
    for exam in records.values():
        if all(k in exam["views"] for k in VIEW_KEYS):
            exams.append(exam)
    return exams


def is_valid_dicom(path):
    try:
        ds = pydicom.dcmread(path)
        _ = ds.pixel_array
        return True
    except Exception:
        return False


def filter_valid_exams(exams):
    valid = []
    dropped = 0
    for exam in exams:
        if all(is_valid_dicom(exam["views"][k]) for k in VIEW_KEYS):
            valid.append(exam)
        else:
            dropped += 1
    return valid, dropped


class MammographyExamDataset(Dataset):
    def __init__(self, exams, image_size):
        if pydicom is None or Image is None:
            raise ImportError("pydicom and Pillow are required for mammography training")
        self.exams = exams
        self.image_size = image_size

    def __len__(self):
        return len(self.exams)

    def _load_view(self, path):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        img = Image.fromarray((arr * 255.0).astype(np.uint8)).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        exam = self.exams[idx]
        views = {k: self._load_view(exam["views"][k]) for k in VIEW_KEYS}
        return {
            "views": views,
            "label": torch.tensor(float(exam["label"]), dtype=torch.float32),
            "study_id": exam["study_id"],
        }


def collate_batch(batch):
    views = {k: torch.stack([item["views"][k] for item in batch], dim=0) for k in VIEW_KEYS}
    labels = torch.stack([item["label"] for item in batch], dim=0)
    study_ids = [item["study_id"] for item in batch]
    return {"views": views, "labels": labels, "study_ids": study_ids}


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["auroc"] = float("nan")

    thresholds = np.unique(probs)
    best_sens_at_90_spec = 0.0
    best_spec_at_90_sens = 0.0
    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if spec >= 0.9:
            best_sens_at_90_spec = max(best_sens_at_90_spec, sens)
        if sens >= 0.9:
            best_spec_at_90_sens = max(best_spec_at_90_sens, spec)
    metrics["sensitivity_at_90_specificity"] = float(best_sens_at_90_spec)
    metrics["specificity_at_90_sensitivity"] = float(best_spec_at_90_sens)
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in loader:
        views = {k: v.to(device) for k, v in batch["views"].items()}
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(training):
            logits, _ = model(views)
            logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        total_loss += loss.item() * labels.size(0)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (output_dir / "best_model.pt")

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    raw_dir = data_dir.parent / "raw"
    exams = build_exam_records(metadata_path, raw_dir)
    if not exams:
        raise RuntimeError(f"No complete 4-view exams found under {raw_dir}")

    exams, dropped = filter_valid_exams(exams)
    if not exams:
        raise RuntimeError("All candidate exams failed DICOM validation")

    split_buckets = {"train": [], "val": [], "test": []}
    for exam in exams:
        split_buckets[exam["split"]].append(exam)

    print(
        f"Loaded exams: train={len(split_buckets['train'])}, "
        f"val={len(split_buckets['val'])}, test={len(split_buckets['test'])}"
    )
    if dropped:
        print(f"Dropped invalid exams: {dropped}")

    train_ds = MammographyExamDataset(split_buckets["train"], args.image_size)
    val_ds = MammographyExamDataset(split_buckets["val"], args.image_size)
    test_ds = MammographyExamDataset(split_buckets["test"], args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = MammographyScreener(pretrained=True)
    if model.encoder.backbone is None:
        raise RuntimeError("timm is required for real mammography training but is not installed")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_epoch = -1
    history = []

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for eval-only run: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = int(checkpoint.get("best_epoch", -1))
        best_val_auroc = float(checkpoint.get("best_val_auroc", float("nan")))
        test_metrics = run_epoch(model, test_loader, criterion, device)
        summary = {
            "best_epoch": best_epoch,
            "best_val_auroc": best_val_auroc,
            "test_auroc": test_metrics["auroc"],
            "test_sensitivity_at_90_specificity": test_metrics["sensitivity_at_90_specificity"],
            "test_specificity_at_90_sensitivity": test_metrics["specificity_at_90_sensitivity"],
            "train_exams": len(train_ds),
            "val_exams": len(val_ds),
            "test_exams": len(test_ds),
            "image_size": args.image_size,
        }
        save_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auroc": train_metrics["auroc"],
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

        if np.isnan(val_metrics["auroc"]):
            continue
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_val_auroc,
                },
                checkpoint_path,
            )

    if best_epoch < 0:
        raise RuntimeError("Validation AUROC was never defined; check label balance.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device)

    summary = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "test_auroc": test_metrics["auroc"],
        "test_sensitivity_at_90_specificity": test_metrics["sensitivity_at_90_specificity"],
        "test_specificity_at_90_sensitivity": test_metrics["specificity_at_90_sensitivity"],
        "train_exams": len(train_ds),
        "val_exams": len(val_ds),
        "test_exams": len(test_ds),
        "image_size": args.image_size,
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "history.json", history)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
