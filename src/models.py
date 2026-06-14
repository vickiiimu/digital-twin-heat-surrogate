"""Neural surrogate models for 2D heat diffusion."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution block that preserves spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HeatCNN(nn.Module):
    """Basic CNN baseline for final-temperature prediction.

    Input:
        x: [batch, 3, height, width]
           channels are initial temperature, heater/source, alpha grid.

    Output:
        y_hat: [batch, 1, height, width]
               predicted final temperature field.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 64,
        depth: int = 5,
        residual_initial: bool = True,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        layers: list[nn.Module] = [ConvBlock(in_channels, hidden_channels)]
        for _ in range(depth - 2):
            layers.append(ConvBlock(hidden_channels, hidden_channels))

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.residual_initial = residual_initial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        delta = self.head(features)

        if self.residual_initial:
            return x[:, :1] + delta
        return delta


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    model = HeatCNN()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(model)
    print("Output shape:", tuple(y.shape))
    print("Trainable parameters:", count_parameters(model))
