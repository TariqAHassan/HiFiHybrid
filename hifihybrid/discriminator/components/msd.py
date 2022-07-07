"""

    Multi-Scale Discriminator

    References:
        * https://arxiv.org/abs/2010.05646
        * https://github.com/jik876/hifi-gan

"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _avgpooling1d_size(l_in: int, kernel_size: int, stride: int, padding: int) -> int:
    return (l_in + 2 * padding - kernel_size) // stride + 1


class ScaleDiscriminator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pooling_params: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pooling_params = pooling_params

        self.pool = nn.AvgPool1d(**pooling_params) if pooling_params else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(1, 128, 15, stride=1, padding=7),
                nn.Conv1d(128, 128, 41, stride=2, groups=4, padding=20),
                nn.Conv1d(128, 256, 41, stride=2, groups=16, padding=20),
                nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20),
                nn.Conv1d(512, 1024, 41, stride=4, groups=16, padding=20),
                nn.Conv1d(1024, 1024, 41, stride=1, groups=16, padding=20),
                nn.Conv1d(1024, 1024, 5, stride=1, padding=2),
            ]
        )
        self.conv_post = nn.Conv1d(
            in_channels=self.blocks[-1].out_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = list()
        y = self.pool(x.unsqueeze(1) if x.ndim == 2 else x)
        for block in self.blocks:
            y = F.leaky_relu(block(y), negative_slope=0.2)
            fmap.append(y)
        y = self.conv_post(y)
        fmap.append(y)
        return y.squeeze(1), fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        seq_len: int = 2**16,
        pooling: tuple[dict[str, int] | None] = (
            None,
            dict(kernel_size=4, stride=2, padding=2),
        ),
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pooling = pooling

        self.discriminators = nn.ModuleList([])
        for p in pooling:
            self.discriminators.append(ScaleDiscriminator(seq_len, pooling_params=p))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        ys = list()
        fmaps = list()
        for d in self.discriminators:
            y, fmap = d(x)
            ys.append(y)
            fmaps.append(fmap)
        return ys, fmaps


if __name__ == "__main__":
    from hifihybrid.utils.training import count_parameters

    self = msd = MultiScaleDiscriminator()
    print(f"MSD(x) Params: {count_parameters(msd):,}")

    x = torch.randn(2, 2**16)
    ys, fmaps = msd(x)

    assert [i.shape for i in ys] == [
        torch.Size([2, 1024]),
        torch.Size([2, 513]),
    ]
    assert all(all(isinstance(i, torch.Tensor) for i in fmap) for fmap in fmaps)
