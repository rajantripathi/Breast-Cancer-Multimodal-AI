from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import torch

    from agents.vision.mil import AttentionMILPool, TransformerMILPool
    from training.tcga_simple_fusion import TCGASimpleFusion
    from training.tcga_verifier import EmbeddedClinicalEncoder, PathwayTokenEncoder, TCGASample, TCGAVerifier, _build_samples, _collate
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    AttentionMILPool = None
    TransformerMILPool = None
    TCGASimpleFusion = None
    EmbeddedClinicalEncoder = None
    PathwayTokenEncoder = None
    TCGASample = None
    TCGAVerifier = None
    _build_samples = None
    _collate = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for MIL tests")
class TestTCGAMIL(unittest.TestCase):
    def test_attention_mil_pool_output_shape(self) -> None:
        pool = AttentionMILPool(input_dim=8, hidden_dim=16, attention_dim=8)
        bag = torch.randn(2, 5, 8)
        lengths = torch.tensor([5, 3], dtype=torch.long)
        pooled = pool(bag, lengths)
        self.assertEqual(tuple(pooled.shape), (2, 16))

    def test_transformer_mil_pool_output_shape(self) -> None:
        pool = TransformerMILPool(input_dim=8, hidden_dim=16, num_heads=4, num_layers=1, dropout=0.0)
        bag = torch.randn(2, 6, 8)
        lengths = torch.tensor([6, 2], dtype=torch.long)
        pooled = pool(bag, lengths)
        self.assertEqual(tuple(pooled.shape), (2, 16))

    def test_collate_pads_variable_length_bags(self) -> None:
        samples = [
            TCGASample(
                sample_id="TCGA-XX-0001",
                vision=torch.randn(4, 8),
                vision_length=4,
                genomics=torch.randn(6),
                clinical=torch.randn(3),
                label=1,
                survival_time=12.0,
                event_observed=1,
            ),
            TCGASample(
                sample_id="TCGA-XX-0002",
                vision=torch.randn(2, 8),
                vision_length=2,
                genomics=torch.randn(6),
                clinical=torch.randn(3),
                label=0,
                survival_time=8.0,
                event_observed=0,
            ),
        ]
        batch = _collate(samples)
        self.assertEqual(tuple(batch["vision"].shape), (2, 4, 8))
        self.assertEqual(batch["vision_lengths"].tolist(), [4, 2])

    def test_build_samples_uses_inferred_patch_embeddings_for_abmil(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            slide_dir = root / "tcga-brca" / "embeddings" / "conch"
            patch_dir = root / "tcga-brca" / "patch_embeddings" / "conch"
            genomics_dir = root / "tcga-brca" / "genomics"
            slide_dir.mkdir(parents=True)
            patch_dir.mkdir(parents=True)
            genomics_dir.mkdir(parents=True)

            slide_path = slide_dir / "TCGA-XX-0001.pt"
            patch_path = patch_dir / "TCGA-XX-0001.pt"
            genomics_path = genomics_dir / "TCGA-XX-0001.pt"
            torch.save(torch.randn(8), slide_path)
            torch.save(torch.randn(7, 8), patch_path)
            torch.save(torch.randn(4), genomics_path)

            frame = pd.DataFrame(
                [
                    {
                        "patient_barcode": "TCGA-XX-0001",
                        "vision_path": str(slide_path),
                        "genomics_path": str(genomics_path),
                        "label": 1,
                        "survival_time": 12.0,
                        "event_observed": 1,
                    }
                ]
            )
            samples = _build_samples(
                frame=frame,
                feature_columns=[],
                means=pd.Series(dtype=float),
                stds=pd.Series(dtype=float),
                vision_dim=8,
                genomics_dim=4,
                modalities={"vision", "genomics", "clinical"},
                vision_aggregation="abmil",
                max_vision_instances=4,
            )
            self.assertEqual(len(samples), 1)
            self.assertEqual(tuple(samples[0].vision.shape), (4, 8))
            self.assertEqual(samples[0].vision_length, 4)

    def test_verifier_forward_accepts_bag_vision(self) -> None:
        model = TCGAVerifier(vision_dim=8, genomics_dim=4, clinical_dim=3, vision_aggregation="abmil")
        vision = torch.randn(3, 5, 8)
        genomics = torch.randn(3, 4)
        clinical = torch.randn(3, 3)
        lengths = torch.tensor([5, 4, 2], dtype=torch.long)
        logits = model(vision, genomics, clinical, vision_lengths=lengths)
        self.assertEqual(tuple(logits.shape), (3,))

    def test_verifier_forward_accepts_transmil_style_pooling(self) -> None:
        model = TCGAVerifier(vision_dim=8, genomics_dim=4, clinical_dim=3, vision_aggregation="transmil")
        vision = torch.randn(2, 6, 8)
        genomics = torch.randn(2, 4)
        clinical = torch.randn(2, 3)
        lengths = torch.tensor([6, 3], dtype=torch.long)
        logits = model(vision, genomics, clinical, vision_lengths=lengths)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_simple_fusion_forward_accepts_bag_vision(self) -> None:
        model = TCGASimpleFusion(vision_dim=8, genomics_dim=4, clinical_dim=3, vision_aggregation="abmil")
        vision = torch.randn(3, 5, 8)
        genomics = torch.randn(3, 4)
        clinical = torch.randn(3, 3)
        lengths = torch.tensor([5, 3, 2], dtype=torch.long)
        logits = model(vision, genomics, clinical, vision_lengths=lengths)
        self.assertEqual(tuple(logits.shape), (3,))

    def test_pathway_token_encoder_output_shape(self) -> None:
        encoder = PathwayTokenEncoder(num_pathways=6, hidden_dim=16)
        values = torch.randn(4, 6)
        pooled = encoder(values)
        self.assertEqual(tuple(pooled.shape), (4, 16))

    def test_embedded_clinical_encoder_output_shape(self) -> None:
        encoder = EmbeddedClinicalEncoder(numeric_dim=3, category_cardinalities=[3, 4, 2], hidden_dim=16)
        numeric = torch.randn(5, 3)
        categorical = torch.tensor(
            [
                [0, 1, 1],
                [1, 2, 0],
                [2, 3, 1],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        pooled = encoder(numeric, categorical)
        self.assertEqual(tuple(pooled.shape), (5, 16))

    def test_verifier_forward_accepts_pathway_tokens(self) -> None:
        model = TCGAVerifier(
            vision_dim=8,
            genomics_dim=6,
            clinical_dim=3,
            vision_aggregation="mean",
            genomics_aggregation="pathway_tokens",
        )
        vision = torch.randn(2, 8)
        genomics = torch.randn(2, 6)
        clinical = torch.randn(2, 3)
        logits = model(vision, genomics, clinical)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_verifier_forward_accepts_embedded_clinical_features(self) -> None:
        model = TCGAVerifier(
            vision_dim=8,
            genomics_dim=6,
            clinical_dim=3,
            vision_aggregation="mean",
            genomics_aggregation="flat",
            clinical_aggregation="embedded",
            clinical_category_cardinalities=[3, 4, 2],
        )
        vision = torch.randn(2, 8)
        genomics = torch.randn(2, 6)
        clinical = torch.randn(2, 3)
        clinical_categories = torch.tensor([[1, 2, 0], [0, 3, 1]], dtype=torch.long)
        logits = model(vision, genomics, clinical, clinical_categories=clinical_categories)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_simple_fusion_forward_accepts_pathway_tokens(self) -> None:
        model = TCGASimpleFusion(
            vision_dim=8,
            genomics_dim=6,
            clinical_dim=3,
            vision_aggregation="mean",
            genomics_aggregation="pathway_tokens",
        )
        vision = torch.randn(2, 8)
        genomics = torch.randn(2, 6)
        clinical = torch.randn(2, 3)
        logits = model(vision, genomics, clinical)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_simple_fusion_forward_accepts_embedded_clinical_features(self) -> None:
        model = TCGASimpleFusion(
            vision_dim=8,
            genomics_dim=6,
            clinical_dim=3,
            vision_aggregation="mean",
            genomics_aggregation="flat",
            clinical_aggregation="embedded",
            clinical_category_cardinalities=[3, 4, 2],
        )
        vision = torch.randn(2, 8)
        genomics = torch.randn(2, 6)
        clinical = torch.randn(2, 3)
        clinical_categories = torch.tensor([[1, 2, 0], [0, 3, 1]], dtype=torch.long)
        logits = model(vision, genomics, clinical, clinical_categories=clinical_categories)
        self.assertEqual(tuple(logits.shape), (2,))


if __name__ == "__main__":
    unittest.main()
