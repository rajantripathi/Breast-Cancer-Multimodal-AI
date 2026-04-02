"""
Extract Mammo-CLIP embeddings for VinDr-Mammo images.

This mirrors the pathology embedding flow: a frozen domain-specific
foundation model produces per-image embeddings, and a lightweight
classifier is trained downstream.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class VinDrImageDataset(Dataset):
    """Load VinDr-Mammo images for feature extraction."""

    def __init__(self, metadata_csv, image_dir, transform=None):
        self.df = pd.read_csv(metadata_csv).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((456, 456)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        png_path = str(row.get("png_path", "") or "").strip()
        if png_path:
            candidate = Path(png_path)
            if candidate.exists():
                return candidate

        candidates = [
            self.image_dir / f"{row['image_id']}.png",
            self.image_dir / f"{row['study_id']}" / f"{row['image_id']}.png",
            self.image_dir / "png_1536" / f"{row['study_id']}" / f"{row['image_id']}.png",
            self.image_dir / "png_1024" / f"{row['study_id']}" / f"{row['image_id']}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find PNG for image_id={row['image_id']}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, row["image_id"], row["study_id"]


def load_mammoclip_encoder(mammoclip_root, checkpoint_path):
    """
    Load the Mammo-CLIP image encoder using the repo's own loader.
    """

    mammoclip_root = Path(mammoclip_root)
    sys.path.insert(0, str(mammoclip_root / "src" / "codebase" / "breastclip" / "model"))

    from modules import load_image_encoder

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_config = checkpoint["config"]["model"]["image_encoder"]
    model = load_image_encoder(encoder_config)

    state_dict = checkpoint["model"]
    image_encoder_weights = {}
    for key, value in state_dict.items():
        if key.startswith("image_encoder."):
            image_encoder_weights[".".join(key.split(".")[1:])] = value

    missing, unexpected = model.load_state_dict(image_encoder_weights, strict=False)
    print(f"Loaded encoder config: {encoder_config}")
    print(f"Loaded {len(image_encoder_weights)} image encoder weights")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Mammo-CLIP checkpoint path")
    parser.add_argument("--metadata", required=True, help="VinDr metadata CSV")
    parser.add_argument("--image-dir", required=True, help="VinDr processed image directory")
    parser.add_argument("--output-dir", required=True, help="Output embeddings directory")
    parser.add_argument(
        "--mammoclip-root",
        default="/scratch/u6ef/rajantripathi.u6ef/Mammo-CLIP",
        help="Path to the cloned Mammo-CLIP repo",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Mammo-CLIP vision encoder...")
    model = load_mammoclip_encoder(args.mammoclip_root, args.checkpoint).to(device)

    dataset = VinDrImageDataset(args.metadata, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    print(f"Extracting embeddings for {len(dataset)} images...")
    count = 0
    embedding_dim = None
    with torch.no_grad():
        for imgs, image_ids, _study_ids in loader:
            imgs = imgs.to(device)
            features = model(imgs)
            embedding_dim = int(features.shape[1])

            for i, image_id in enumerate(image_ids):
                torch.save(features[i].cpu(), output_dir / f"{image_id}.pt")
                count += 1

            if count % 500 == 0:
                print(f"  Extracted {count}/{len(dataset)}")

    print(f"Done. {count} embeddings saved to {output_dir}")
    print(f"Embedding dim: {embedding_dim}")


if __name__ == "__main__":
    main()
