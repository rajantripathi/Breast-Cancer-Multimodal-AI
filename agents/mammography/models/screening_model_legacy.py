"""
Legacy mammography screening model used for the original 224px ConvNeXt run.

This preserves the simpler 4-view attention fusion path from commit f209755 so
we can reproduce the historical baseline without mixing in later architecture
changes such as bilateral asymmetry tokens.
"""

import torch
import torch.nn as nn


class LegacyViewEncoder(nn.Module):
    """Encode a single mammography view using ConvNeXt-Base."""

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            import timm

            self.backbone = timm.create_model(
                "convnext_base.fb_in22k_ft_in1k",
                pretrained=pretrained,
                num_classes=0,
            )
            self.feat_dim = self.backbone.num_features
        except ImportError:
            self.backbone = None
            self.feat_dim = 768

    def forward(self, x):
        if self.backbone is None:
            return torch.randn(x.size(0), self.feat_dim, device=x.device)
        return self.backbone(x)


class LegacyMammographyScreener(nn.Module):
    """Simple 4-view attention fusion classifier."""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = LegacyViewEncoder(pretrained=pretrained)
        feat_dim = self.encoder.feat_dim

        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, views):
        features = []
        for key in ["lcc", "rcc", "lmlo", "rmlo"]:
            if key in views and views[key] is not None:
                features.append(self.encoder(views[key]))

        if not features:
            raise ValueError("No valid views provided")

        stacked = torch.stack(features, dim=1)
        attn_logits = self.attention(stacked)
        attn_weights = torch.softmax(attn_logits, dim=1)
        aggregated = (stacked * attn_weights).sum(dim=1)
        logit = self.classifier(aggregated)
        return logit, attn_weights.squeeze(-1)
