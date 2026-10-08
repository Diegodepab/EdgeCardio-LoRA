"""Compact 1D Convolutional Neural Network for ECG arrhythmia classification.

Optimized for Edge AI deployment (< 150k parameters), minimizing computational
complexity while capturing temporal morphological features (QRS, ST segments).
"""

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """1D Convolutional Block: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECG1DCNN(nn.Module):
    """Compact 1D-CNN architecture for 5-class AAMI EC57 ECG classification.

    Input shape: (batch_size, 1, 360)
    Output shape: (batch_size, 5)
    Total parameters: ~44,000 (< 150k budget).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extractor: 3 Convolutional Blocks
        # Input: (B, 1, 360) -> Output: (B, 128, 45)
        self.features = nn.Sequential(
            ConvBlock1D(
                in_channels=in_channels, out_channels=32, kernel_size=7, padding=3, pool_size=2
            ),
            ConvBlock1D(in_channels=32, out_channels=64, kernel_size=5, padding=2, pool_size=2),
            ConvBlock1D(in_channels=64, out_channels=128, kernel_size=3, padding=1, pool_size=2),
        )

        # Global Average Pooling: (B, 128, 45) -> (B, 128, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Linear Classifier: Head suitable for direct fine-tuning and LoRA injection
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 360).

        Returns:
            Logits tensor of shape (batch_size, 5).
        """
        feats = self.features(x)
        pooled = self.global_pool(feats)
        flattened = torch.flatten(pooled, start_dim=1)
        logits = self.classifier(flattened)
        return logits


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Returns total and trainable parameter counts of the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
