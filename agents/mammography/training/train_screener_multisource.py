"""
Legacy-style mammography screener training with train-only auxiliary metadata.

This keeps the recovered legacy recipe intact:
- ConvNeXt-Base 4-view attention fusion
- BCEWithLogitsLoss
- AdamW at 1e-4
- batch size 8
- no class balancing
- no source weighting

Auxiliary metadata is folded into the training split only. Validation and test
remain driven by the primary dataset metadata.
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
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--aux-metadata-csv", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/mammography")
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


def infer_dataset_source(metadata_path, row):
    source = str(row.get("dataset_source") or "").strip().lower()
    if source:
        return source
    if "cmmd" in str(metadata_path).lower():
        return "cmmd"
    return "vindr"


def resolve_dicom_path(row, raw_dir):
    explicit = str(row.get("raw_path") or row.get("dicom_path") or "").strip()
    if explicit and Path(explicit).exists():
        return Path(explicit)

    study_id = row.get("study_id")
    image_id = row.get("image_id")
    if not study_id or not image_id:
        return None

    raw_dir = Path(raw_dir)
    archive_root = raw_dir / (
        "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
    )
    candidates = [
        raw_dir / "images" / str(study_id) / f"{image_id}.dicom",
        archive_root / "images" / str(study_id) / f"{image_id}.dicom",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_exam_records(metadata_specs):
    records = {}
    for metadata_path, raw_dir, auxiliary_only in metadata_specs:
        for row in csv.DictReader(metadata_path.open()):
            study_id = str(row["study_id"]).strip()
            image_id = str(row["image_id"]).strip()
            split = "train" if auxiliary_only else str(row.get("split", "train")).strip().lower()
            label = int(row["label"])
            key = normalize_view_name(row.get("laterality"), row.get("view_position") or row.get("view"))
            if key is None:
                continue

            dataset_source = infer_dataset_source(metadata_path, row)
            dicom_path = resolve_dicom_path(row, raw_dir)
            if dicom_path is None or not dicom_path.exists():
                continue

            exam_key = (dataset_source, study_id)
            exam = records.setdefault(
                exam_key,
                {
                    "study_id": study_id,
                    "sample_id": f"{dataset_source}:{study_id}",
                    "split": split,
                    "label": label,
                    "dataset_source": dataset_source,
                    "views": {},
                },
            )
            exam["label"] = max(int(exam["label"]), label)
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
            "study_id": exam["sample_id"],
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

    metadata_path = Path(args.metadata_csv) if args.metadata_csv else (data_dir / "metadata.csv")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata_specs = [(metadata_path, data_dir.parent / "raw", False)]
    for item in args.aux_metadata_csv:
        aux_path = Path(item)
        if not aux_path.exists():
            raise FileNotFoundError(f"Missing auxiliary metadata file: {aux_path}")
        metadata_specs.append((aux_path, aux_path.parent.parent / "raw", True))

    exams = build_exam_records(metadata_specs)
    if not exams:
        raise RuntimeError("No complete 4-view exams found across the supplied metadata sources")

    exams, dropped = filter_valid_exams(exams)
    if not exams:
        raise RuntimeError("All candidate exams failed DICOM validation")

    split_buckets = {"train": [], "val": [], "test": []}
    source_counts = {"vindr": 0, "cmmd": 0}
    for exam in exams:
        split_buckets[exam["split"]].append(exam)
        source_counts[exam["dataset_source"]] = source_counts.get(exam["dataset_source"], 0) + 1

    print(
        f"Loaded exams: train={len(split_buckets['train'])}, "
        f"val={len(split_buckets['val'])}, test={len(split_buckets['test'])}"
    )
    if dropped:
        print(f"Dropped invalid exams: {dropped}")
    print(f"Data sources: {source_counts}")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_auroc = -1.0
    best_epoch = -1
    history = []

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
                output_dir / "best_model.pt",
            )

    if best_epoch < 0:
        raise RuntimeError("Validation AUROC was never defined; check label balance.")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
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
        "data_sources": source_counts,
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "history.json", history)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
