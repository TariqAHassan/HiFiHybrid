"""

    Layers

"""
import warnings

import torch
from numpy import pi
from scipy.signal import firwin
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=UserWarning)


def _torch_firwin(
    n: int,
    cutoff_freq: float,
    f_h: float,
    fs: float,  # sampling frequency
) -> torch.Tensor:
    A = 2.285 * ((n / 2) - 1) * pi * (4 * f_h) + 7.95
    beta = 0.1102 * (A - 8.7)
    filter_fn = firwin(
        numtaps=n,
        cutoff=cutoff_freq,
        width=f_h,
        fs=fs,
        window=("kaiser", beta),
    )
    return torch.from_numpy(filter_fn).float()


class LowPassConv1d(nn.Module):
    low_pass: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        sr: int,
        m: int = 2,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.sr = sr
        self.m = m
        self.trainable = trainable

        low_pass = (
            _torch_firwin(6 * m, cutoff_freq=sr / (2 * m), f_h=0.6 / m, fs=sr * m)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(in_channels, 1, 1)
        )
        if trainable:
            self.register_parameter("low_pass", nn.Parameter(low_pass))
        else:
            self.register_buffer("low_pass", low_pass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            input=x,
            weight=self.low_pass,
            padding="same",
            groups=self.in_channels,
        )


class Snake1d(nn.Module):
    def __init__(self, in_channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        self.alpha = nn.Parameter(torch.randn(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha[None, :, None]
        act = (1 / alpha.add(self.eps)) * torch.sin(alpha * x).pow(2)
        return x + act


class UpDownSnake1d(Snake1d):
    def __init__(
        self,
        in_channels: int,
        sr: int,
        m: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(in_channels, eps=eps)
        self.sr = sr
        self.m = m

        self.lowpass_conv0 = LowPassConv1d(in_channels, sr=sr, m=m)
        self.lowpass_conv1 = LowPassConv1d(in_channels, sr=sr, m=m)
        self.snake1d = Snake1d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lowpass_conv0(x)
        x = self.snake1d(x)
        x = self.lowpass_conv1(x)
        return x


if __name__ == "__main__":
    from hifihybrid.utils.training import count_parameters

    IN_CHANNELS: int = 128

    x = torch.randn(2, IN_CHANNELS, 1024)

    self = Snake1d(IN_CHANNELS)
    print(f"Snake1d() Params: {count_parameters(self):,}")

    y = self(x)
    assert y.shape == x.shape

    self = UpDownSnake1d(IN_CHANNELS, sr=16_000)
    print(f"UpDownSnake1d() Params: {count_parameters(self):,}")
    y = self(x)
    assert y.shape == x.shape
