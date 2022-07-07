"""

    Least Squares Loss

"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.nn import functional as F

from hifihybrid.discriminator.model import Discriminator


def mel_loss(
    real_mel: torch.Tensor,
    fake_mel: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(fake_mel, target=real_mel)


def feature_loss(
    fmaps_real: list[[list[torch.Tensor]]],
    fmaps_fake: list[[list[torch.Tensor]]],
) -> torch.Tensor:
    loss = 0
    for dr, dg in zip(fmaps_real, fmaps_fake):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


class LeastSquaresLoss:
    def __init__(
        self,
        discriminator: Discriminator,
        to_mel: Callable[[torch.Tensor], torch.Tensor],
        g_mel_lambda: int | float = 45,
        g_feat_lambda: int | float = 2,
        log_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.discriminator = discriminator
        self.log_fn = log_fn
        self.to_mel = to_mel
        self.g_mel_lambda = g_mel_lambda
        self.g_feat_lambda = g_feat_lambda

    def log(self, value: str, name: str, prog_bar: bool = False) -> None:
        if callable(self.log_fn):
            self.log_fn(name, value, prog_bar=prog_bar)  # noqa

    @staticmethod
    def _subdiscriminator_loss(
        ys_real: list[torch.Tensor],
        ys_fake: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(ys_real, ys_fake):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
        return loss

    def discriminator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        global_step: int,  # noqa
    ) -> torch.Tensor:
        d_outputs = self.discriminator(fake_batch.detach(), real=real_batch)
        loss_msd = self._subdiscriminator_loss(
            ys_real=d_outputs.real.msd.ys,
            ys_fake=d_outputs.fake.msd.ys,
        )
        loss_mpd = self._subdiscriminator_loss(
            ys_real=d_outputs.real.mpd.ys,
            ys_fake=d_outputs.fake.mpd.ys,
        )
        self.log(loss_msd, name="d_loss_msd")
        self.log(loss_mpd, name="d_loss_mpd")
        return loss_msd + loss_mpd

    def generator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        global_step: int,  # noqa
    ) -> torch.Tensor:
        d_outputs = self.discriminator(fake_batch, real=real_batch)

        # 1. Adversarial loss
        loss_adv = 0
        for model in (d_outputs.fake.msd, d_outputs.fake.mpd):
            for dg in model.ys:
                loss_adv += torch.mean((1 - dg) ** 2)

        # 2. L1 Mel Loss
        loss_mel = mel_loss(
            real_mel=self.to_mel(real_batch),
            fake_mel=self.to_mel(fake_batch),
        )

        # 3. Feature Loss
        loss_feat = 0
        for model in ("msd", "mpd"):
            loss_feat += feature_loss(
                fmaps_real=getattr(d_outputs.real, model).fmaps,
                fmaps_fake=getattr(d_outputs.fake, model).fmaps,
            )

        self.log(loss_adv, name="g_loss_adv")
        self.log(loss_mel, name="g_loss_mel", prog_bar=True)
        self.log(loss_feat, name="g_loss_feat")
        return (
            loss_adv
            + loss_mel.mul(self.g_mel_lambda)
            + loss_feat.mul(self.g_feat_lambda)  # noqa
        )
