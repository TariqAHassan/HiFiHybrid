"""

    Discriminator

"""
import torch
from torch import nn

from hifihybrid.discriminator._types import (
    DiscriminatorOutput,
    MpdOutput,
    MsdOutput,
    UniOutput,
)
from hifihybrid.discriminator.components.mpd import MultiPeriodDiscriminator
from hifihybrid.discriminator.components.msd import MultiScaleDiscriminator


class Discriminator(nn.Module):
    def __init__(self, seq_len: int = 2**16) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.msd = MultiScaleDiscriminator(seq_len=seq_len)
        self.mdp = MultiPeriodDiscriminator(seq_len=seq_len)

    def _package(self, x: torch.Tensor) -> UniOutput:
        return UniOutput(
            msd=MsdOutput(*self.msd(x)),
            mpd=MpdOutput(*self.mdp(x)),
        )

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> DiscriminatorOutput:
        return DiscriminatorOutput(
            fake=self._package(fake),
            real=self._package(real),
        )


if __name__ == "__main__":
    from hifihybrid.utils.training import count_parameters

    self = d = Discriminator(2**16)
    print(f"D(x) Params: {count_parameters(d):,}")

    fake, real = torch.randn(2, 1, d.seq_len)  # [fake/real, batch, seq_len]
    output = d(fake, real)
    assert isinstance(output, DiscriminatorOutput)
