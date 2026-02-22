import torch
import torch.nn as nn
from torchvision import models


class PairScorer(nn.Module):
    """Encode two images and output one similarity logit."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        if backbone != "resnet18":
            raise ValueError("Only resnet18 is supported in this minimal template.")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        encoder = models.resnet18(weights=weights)
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder

        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        feat_a = self.encoder(image_a)
        feat_b = self.encoder(image_b)
        pair_feat = torch.cat([feat_a, feat_b], dim=1)
        logits = self.head(pair_feat).squeeze(1)
        return logits
