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


class SpectralConv2d(nn.Module):
    """2D Fourier layer used by a Fourier Neural Operator.

    The layer keeps only a fixed number of low-frequency Fourier modes. This is
    the core operator-learning idea: learn global spatial interactions in the
    frequency domain instead of only local convolutional stencils.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def complex_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes1 = min(self.modes1, height)
        modes2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :modes1, :modes2] = self.complex_mul2d(
            x_ft[:, :, :modes1, :modes2],
            self.weights_pos[:, :, :modes1, :modes2],
        )
        out_ft[:, :, -modes1:, :modes2] = self.complex_mul2d(
            x_ft[:, :, -modes1:, :modes2],
            self.weights_neg[:, :, :modes1, :modes2],
        )

        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    """One Fourier layer plus a learned pointwise linear correction."""

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class HeatFNO(nn.Module):
    """Small Fourier Neural Operator for 2D heat-diffusion prediction.

    Input:
        x: [batch, 3, height, width]

    Output:
        y_hat: [batch, 1, height, width]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        use_grid: bool = True,
        residual_initial: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.use_grid = use_grid
        self.residual_initial = residual_initial
        lift_channels = in_channels + 2 if use_grid else in_channels

        self.lift = nn.Conv2d(lift_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock(width, modes1, modes2) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def coordinate_grid(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        grid_y = torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.repeat(batch_size, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_input = torch.cat([x, self.coordinate_grid(x)], dim=1) if self.use_grid else x
        features = self.lift(model_input)
        features = self.blocks(features)
        out = self.project(features)
        if self.residual_initial:
            return x[:, :1] + out
        return out


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    model = HeatFNO()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(model)
    print("Output shape:", tuple(y.shape))
    print("Trainable parameters:", count_parameters(model))
