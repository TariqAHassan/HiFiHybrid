"""

    Discriminator Types

"""
from typing import NamedTuple

import torch


class MsdOutput(NamedTuple):
    ys: list[torch.Tensor]
    fmaps: list[list[torch.Tensor]]


class MpdOutput(NamedTuple):
    ys: list[torch.Tensor]
    fmaps: list[list[torch.Tensor]]


class UniOutput(NamedTuple):
    msd: MsdOutput
    mpd: MpdOutput


class DiscriminatorOutput(NamedTuple):
    fake: UniOutput
    real: UniOutput
