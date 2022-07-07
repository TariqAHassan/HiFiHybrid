"""

    Generator

    References:
        * https://arxiv.org/abs/2010.05646
        * https://github.com/jik876/hifi-gan

"""
from __future__ import annotations

from math import prod

import torch
from torch import nn

from hifihybrid.generator.layers.snake import UpDownSnake1d


class AmpBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.activation = UpDownSnake1d(
            in_channels=channels,
            sr=width,
        )

        self.convs = nn.ModuleList([])
        for d in dilations:
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=d,
                padding="same",
                bias=False,
            )
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(x)
        for conv in self.convs:
            y = conv(y) + y
        return y


class AmpComplex(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int,
        kernel_sizes: tuple[int, ...],
        dilation_sizes: tuple[list[int], ...],
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.kernel_sizes = kernel_sizes
        self.dilation_sizes = dilation_sizes

        self.amps = nn.ModuleList([])
        for k, d in zip(kernel_sizes, dilation_sizes):
            amp = AmpBlock(
                channels=channels,
                width=width,
                kernel_size=k,
                dilations=d,
            )
            self.amps.append(amp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 0
        for amp in self.amps:
            y = y + amp(x)
        return y


class GeneralBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        width: int,
        amp_kernel_sizes: tuple[int, ...],
        amp_dilation_sizes: tuple[list[int], ...],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.width = width
        self.amp_kernel_sizes = amp_kernel_sizes
        self.amp_dilation_sizes = amp_dilation_sizes

        self.tconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2,
        )
        self.amp_complex = AmpComplex(
            channels=out_channels,
            width=width,
            kernel_sizes=amp_kernel_sizes,
            dilation_sizes=amp_dilation_sizes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tconv(x)
        y = self.amp_complex(y)
        return y


class FinalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        width: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.width = width

        self.activation = UpDownSnake1d(
            in_channels=in_channels,
            sr=width,
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(x)
        y = self.conv(y)
        return y


class Generator(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_mels: int = 128,
        initial_channels: int = 512,
        stride_sizes: tuple[int, ...] = (8, 8, 2, 2),
        amp_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        amp_dilation_sizes: tuple[list[int], ...] = (
            # Note: Sizes are given by [kernel_size * 2 + 1, ...]
            [1, 1],
            [3, 1],
            [5, 1],
        ),
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.initial_channels = initial_channels
        self.stride_sizes = stride_sizes
        self.amp_kernel_sizes = amp_kernel_sizes
        self.amp_dilation_sizes = amp_dilation_sizes

        self.initial_block = nn.Conv1d(
            in_channels=n_mels,
            out_channels=initial_channels,
            kernel_size=7,
            bias=False,
            padding="same",
        )

        width = n_frames
        self.blocks = nn.ModuleList()
        for i, u in enumerate(stride_sizes):
            block = GeneralBlock(
                in_channels=initial_channels // (2**i),
                out_channels=initial_channels // (2 ** (i + 1)),
                kernel_size=2 * u,  # see fig. 1 of HiFi paper
                stride=u,
                width=width,
                amp_kernel_sizes=amp_kernel_sizes,
                amp_dilation_sizes=amp_dilation_sizes,
            )
            self.blocks.append(block)
            width *= u

        self.final_block = FinalBlock(
            in_channels=self.blocks[-1].out_channels,
            out_channels=1,
            kernel_size=7,
            width=width,
        )

    @property
    def final_seq_len(self) -> int:
        return self.n_frames * prod(self.stride_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_block(x)
        for block in self.blocks:
            x = block(x)
        return self.final_block(x).squeeze(1)


if __name__ == "__main__":
    from hifihybrid.utils.training import count_parameters

    self = g = Generator(32)
    print(f"G(z) Params: {count_parameters(g):,}")
    print(f"G(z) [Blocks] Params: {count_parameters(g.blocks):,}")

    x = torch.randn(2, g.n_mels, self.n_frames)
    y = g(x)
    assert y.shape == (x.shape[0], g.final_seq_len)
