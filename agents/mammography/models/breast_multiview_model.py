"""
Breast-wise two-view mammography classifier.
"""

from pathlib import Path

import torch
import torch.nn as nn


class BreastViewEncoder(nn.Module):
    def __init__(self, backbone_name="convnext_base.fb_in22k_ft_in1k", pretrained=True):
        super().__init__()
        try:
            import timm

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
            )
            self.feat_dim = self.backbone.num_features
        except ImportError:
            self.backbone = None
            self.feat_dim = 1024

    def load_backbone_checkpoint(self, checkpoint_path):
        if self.backbone is None or not checkpoint_path:
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        candidates = [state_dict]
        for prefix in ["backbone.", "encoder.", "model.", "module."]:
            filtered = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if filtered:
                candidates.append(filtered)

        best_result = None
        best_missing = None
        for candidate in candidates:
            missing, unexpected = self.backbone.load_state_dict(candidate, strict=False)
            score = len(missing) + len(unexpected)
            if best_missing is None or score < best_missing:
                best_missing = score
                best_result = (missing, unexpected)

        if best_result is not None:
            missing, unexpected = best_result
            print(f"Loaded backbone checkpoint {Path(checkpoint_path).name}: missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        if self.backbone is None:
            return torch.randn(x.size(0), self.feat_dim, device=x.device)
        return self.backbone(x)


class BreastMultiViewClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_base.fb_in22k_ft_in1k",
        pretrained=True,
        proj_dim=256,
        hidden_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        self.encoder = BreastViewEncoder(backbone_name=backbone_name, pretrained=pretrained)
        feat_dim = self.encoder.feat_dim

        self.view_proj = nn.Linear(feat_dim, proj_dim)
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

    def freeze_backbone(self, freeze=True):
        if self.encoder.backbone is None:
            return
        for param in self.encoder.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, cc, mlo, mask):
        cc_feat = self.encoder(cc)
        mlo_feat = self.encoder(mlo)

        cc_proj = self.view_proj(cc_feat)
        mlo_proj = self.view_proj(mlo_feat)
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
