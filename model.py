from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ParameterCount:
    total: int
    trainable: int
    frozen: int


class SiamesePairRegressor(nn.Module):
    """Build a Siamese regression model for pairwise product deviation."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        freeze_backbone_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))

        backbone = models.resnet50(pretrained=True)
        embedding_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        feature_dim = embedding_dim * 4
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_activation = nn.Sigmoid()

    def set_backbone_trainable(self, epoch: int) -> None:
        """Freeze or unfreeze the backbone based on the current epoch."""
        should_train_backbone = epoch >= self.freeze_backbone_epochs
        for parameter in self.encoder.parameters():
            parameter.requires_grad = should_train_backbone

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        abs_diff = torch.abs(e1 - e2)
        prod = e1 * e2

        feat = torch.cat([e1, e2, abs_diff, prod], dim=1)
        score_0_1 = self.output_activation(self.regression_head(feat)).squeeze(1)

        return {
            "score": score_0_1,
            "deviation_percent": score_0_1 * 100.0,
        }


def count_parameters(model: nn.Module) -> ParameterCount:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    return ParameterCount(total=total, trainable=trainable, frozen=frozen)


def print_model_summary(model: nn.Module) -> None:
    params = count_parameters(model)
    print("Model summary")
    print(f"- Model class: {model.__class__.__name__}")
    print(f"- Total parameters: {params.total:,}")
    print(f"- Trainable parameters: {params.trainable:,}")
    print(f"- Frozen parameters: {params.frozen:,}")
