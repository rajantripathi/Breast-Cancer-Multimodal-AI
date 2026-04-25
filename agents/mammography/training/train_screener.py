"""
Training script for mammography screening model.

Usage:
  python -m agents.mammography.training.train_screener \
    --data-dir data/mammography/vindr-mammo/processed \
    --output-dir outputs/mammography \
    --epochs 50 \
    --lr 3e-4 \
    --batch-size 2 \
    --device auto
"""

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from agents.mammography.models.screening_model import MammographyScreener

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-sampler", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["smoothed_bce", "weighted_bce", "focal"],
        default="weighted_bce",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--allow-horizontal-flip", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=7.0)
    parser.add_argument("--crop-scale-min", type=float, default=0.95)
    parser.add_argument("--brightness-jitter", type=float, default=0.08)
    parser.add_argument("--contrast-jitter", type=float, default=0.08)
    parser.add_argument(
        "--tta",
        choices=["none", "hflip"],
        default="none",
        help="Validation/test-time augmentation. Horizontal flip is opt-in only.",
    )
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
    def __init__(
        self,
        exams,
        image_size,
        training=False,
        allow_horizontal_flip=False,
        rotation_degrees=7.0,
        crop_scale_min=0.95,
        brightness_jitter=0.08,
        contrast_jitter=0.08,
    ):
        if pydicom is None or Image is None:
            raise ImportError("pydicom and Pillow are required for mammography training")
        self.exams = exams
        self.image_size = image_size
        self.training = training
        self.allow_horizontal_flip = allow_horizontal_flip
        self.rotation_degrees = rotation_degrees
        self.crop_scale_min = crop_scale_min
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __len__(self):
        return len(self.exams)

    def _load_png_or_dicom(self, exam, view_key):
        path = exam["views"][view_key]
        png_path = exam.get("png_views", {}).get(view_key)
        if png_path is not None and png_path.exists():
            img = Image.open(png_path)
            arr = np.asarray(img, dtype=np.float32)
            arr /= 65535.0 if arr.max() > 255 else 255.0
            return arr

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return arr

    def _apply_augmentations(self, img):
        # Horizontal flips can corrupt laterality semantics in mammography, so
        # they are disabled by default and must be explicitly enabled.
        if self.allow_horizontal_flip and random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

        crop_scale = random.uniform(self.crop_scale_min, 1.0)
        width, height = img.size
        crop_w = max(1, int(width * crop_scale))
        crop_h = max(1, int(height * crop_scale))
        left = random.randint(0, max(width - crop_w, 0))
        top = random.randint(0, max(height - crop_h, 0))
        img = img.crop((left, top, left + crop_w, top + crop_h)).resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

        brightness = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
        contrast = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
        arr = np.asarray(img, dtype=np.float32)
        arr *= brightness
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 65535.0).astype(np.uint16), mode="I;16")

    def _load_view(self, exam, view_key):
        arr = self._load_png_or_dicom(exam, view_key)
        img = Image.fromarray((arr * 65535.0).astype(np.uint16), mode="I;16")
        if self.training:
            img = self._apply_augmentations(img)
        arr = np.asarray(img, dtype=np.float32) / 65535.0
        if arr.shape != (self.image_size, self.image_size):
            img = Image.fromarray((arr * 65535.0).astype(np.uint16), mode="I;16").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32) / 65535.0
        arr = np.stack([arr, arr, arr], axis=0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        exam = self.exams[idx]
        views = {k: self._load_view(exam, k) for k in VIEW_KEYS}
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
        metrics["prauc"] = float(average_precision_score(labels, probs))
    else:
        metrics["auroc"] = float("nan")
        metrics["prauc"] = float("nan")

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


def apply_tta(views, mode):
    if mode == "hflip":
        return {k: torch.flip(v, dims=[3]) for k, v in views.items()}
    return views


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, accumulation_steps=1, tta="none"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels = []
    all_probs = []
    if training:
        optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        views = {k: v.to(device) for k, v in batch["views"].items()}
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(training):
            logits, _ = model(views)
            logits = logits.squeeze(1)
            if not training and tta != "none":
                aug_logits, _ = model(apply_tta(views, tta))
                logits = 0.5 * (logits + aug_logits.squeeze(1))
            loss = criterion(logits, labels)
            if training:
                (loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

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


def build_manifest(data_dir, metadata_path, output_dir, args, split_buckets, train_neg, train_pos, pos_weight_value):
    return {
        "git_commit": get_git_commit(),
        "data_dir": str(data_dir),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "config": vars(args),
        "split_counts": {k: len(v) for k, v in split_buckets.items()},
        "class_balance": {
            "train_negatives": train_neg,
            "train_positives": train_pos,
            "pos_weight": pos_weight_value,
        },
    }


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        smooth_labels = labels * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(logits, smooth_labels)


class BinaryFocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, labels):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * labels + (1.0 - probs) * (1.0 - labels)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * bce).mean()


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def build_sampler_and_pos_weight(exams, override_pos_weight=None):
    train_labels = [int(exam["label"]) for exam in exams]
    n_neg = sum(1 for label in train_labels if label == 0)
    n_pos = sum(1 for label in train_labels if label == 1)
    if n_pos == 0:
        raise RuntimeError("No positive exams found in training split.")
    pos_weight_value = override_pos_weight if override_pos_weight is not None else (n_neg / n_pos)
    sample_weights = [1.0 if label == 0 else pos_weight_value for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, float(pos_weight_value), n_neg, n_pos


def build_criterion(args, device, pos_weight_value):
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    if args.loss == "smoothed_bce":
        return SmoothedBCEWithLogitsLoss(args.label_smoothing)
    if args.loss == "focal":
        return BinaryFocalLoss(pos_weight=pos_weight, gamma=args.focal_gamma)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    raw_dir = data_dir.parent / "raw"
    exams = build_exam_records(metadata_path, raw_dir)
    if not exams:
        raise RuntimeError(f"No complete 4-view exams found under {raw_dir}")

    png_root = data_dir / f"png_{args.image_size}"
    fallback_png_root = data_dir / "png_1024"
    for exam in exams:
        exam["png_views"] = {}
        for key, dicom_path in exam["views"].items():
            image_id = dicom_path.stem
            png_path = png_root / exam["study_id"] / f"{image_id}.png"
            if not png_path.exists():
                png_path = fallback_png_root / exam["study_id"] / f"{image_id}.png"
            if png_path.exists():
                exam["png_views"][key] = png_path

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

    train_ds = MammographyExamDataset(
        split_buckets["train"],
        args.image_size,
        training=True,
        allow_horizontal_flip=args.allow_horizontal_flip,
        rotation_degrees=args.rotation_degrees,
        crop_scale_min=args.crop_scale_min,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
    )
    val_ds = MammographyExamDataset(split_buckets["val"], args.image_size, training=False)
    test_ds = MammographyExamDataset(split_buckets["test"], args.image_size, training=False)

    sampler = None
    pos_weight_value = 1.0
    train_neg = train_pos = 0
    if args.balance_sampler or args.loss in {"weighted_bce", "focal"} or args.pos_weight is not None:
        sampler, pos_weight_value, train_neg, train_pos = build_sampler_and_pos_weight(
            split_buckets["train"],
            override_pos_weight=args.pos_weight,
        )
        print(f"Class balance: {train_neg} neg, {train_pos} pos, weight={pos_weight_value:.4f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
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

    model = MammographyScreener(pretrained=True)
    if model.encoder.backbone is None:
        raise RuntimeError("timm is required for real mammography training but is not installed")
    model = model.to(device)
    criterion = build_criterion(args, device, pos_weight_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
    total_optimizer_steps = args.epochs * int(np.ceil(len(train_loader) / accumulation_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_optimizer_steps, 1),
        eta_min=1e-6,
    )

    best_val_auroc = -1.0
    best_epoch = -1
    history = []
    manifest = build_manifest(
        data_dir,
        metadata_path,
        output_dir,
        args,
        split_buckets,
        train_neg,
        train_pos,
        pos_weight_value,
    )
    save_json(output_dir / "manifest.json", manifest)

    for epoch in range(1, args.epochs + 1):
        model.freeze_backbone(epoch <= args.freeze_epochs)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, tta=args.tta)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auroc": train_metrics["auroc"],
            "train_prauc": train_metrics["prauc"],
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_prauc": val_metrics["prauc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))
        save_json(output_dir / "history.json", history)

        if np.isnan(val_metrics["auroc"]):
            continue
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "git_commit": get_git_commit(),
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_val_auroc,
                },
                output_dir / "best_model.pt",
            )

    if best_epoch < 0:
        raise RuntimeError("Validation AUROC was never defined; check label balance.")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, tta=args.tta)

    summary = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "test_auroc": test_metrics["auroc"],
        "test_prauc": test_metrics["prauc"],
        "test_sensitivity_at_90_specificity": test_metrics["sensitivity_at_90_specificity"],
        "test_specificity_at_90_sensitivity": test_metrics["specificity_at_90_sensitivity"],
        "train_exams": len(train_ds),
        "val_exams": len(val_ds),
        "test_exams": len(test_ds),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size,
        "loss": args.loss,
        "balance_sampler": args.balance_sampler,
        "pos_weight": pos_weight_value,
        "tta": args.tta,
        "train_negatives": train_neg,
        "train_positives": train_pos,
        "git_commit": get_git_commit(),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
