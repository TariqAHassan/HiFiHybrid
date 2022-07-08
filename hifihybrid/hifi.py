"""

    Trainer

"""
import copy
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (  # noqa
    _METRIC_COLLECTION as METRIC_COLLECTION,
)
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from hifihybrid.discriminator.model import Discriminator
from hifihybrid.dsp import MelSpec
from hifihybrid.generator.model import Generator
from hifihybrid.loss import LeastSquaresLoss
from hifihybrid.utils.metrics import BestMetricTracker
from hifihybrid.utils.saving import audio_saver, spec_grid_saver  # noqa
from hifihybrid.utils.training import compute_total_norm, update_average

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)

PROGRESS_BAR_STATS: tuple[str, ...] = (
    "loss",
    "d_loss",
    "g_loss",
    "g_loss_mel",
    "val_loss_mel",
)


class HiFiHybrid(pl.LightningModule):
    def __init__(
        self,
        seq_len: int,
        n_mels: int,
        sample_rate: int,
        start_time: str,
        base_dir: str,
        # Data
        dataset: str,
        # Training
        ema_decay: float = 0.999,
        lr: float = 1e-4,
        lr_decay: float = 0.999,
        adam_betas: tuple[float, float] = (0.8, 0.99),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            Namespace(
                seq_len=seq_len,
                n_mels=n_mels,
                sample_rate=sample_rate,
                start_time=start_time,
                base_dir=base_dir,
                dataset=dataset,
                ema_decay=ema_decay,
                lr=lr,
                lr_decay=lr_decay,
                adam_betas=adam_betas,
            )
        )

        self.melspec = MelSpec(
            seq_len=self.hparams.seq_len,
            n_mels=self.hparams.n_mels,
            sr=self.hparams.sample_rate,
        )

        self.g = Generator(
            n_frames=self.melspec.n_frames,
            n_mels=self.hparams.n_mels,
        )
        self.d = Discriminator(self.hparams.seq_len)

        # Generator EMA
        self.g_ema = copy.deepcopy(self.g)
        update_average(self.g_ema, model_src=self.g, beta=0)

        # Define loss ---
        self.loss = LeastSquaresLoss(
            discriminator=self.d,
            to_mel=self.melspec,
            log_fn=self.log,
        )
        self.best_tracker = BestMetricTracker(
            dict(val_loss_mel="min"),
        )

        self.val_batch = None

    @property
    def name(self) -> str:
        return "hifi_output"

    def get_epoch(self) -> int:
        return self.current_epoch

    @property
    def storage(self) -> Path:
        directory = (
            Path(self.hparams.base_dir)
            .joinpath(self.name)
            .joinpath(self.hparams.start_time)
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def checkpoints_dir(self) -> Path:
        directory = self.storage.joinpath("checkpoints")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def samples_dir(self) -> Path:
        directory = self.storage.joinpath("samples")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def image_samples_dir(self) -> Path:
        directory = self.samples_dir.joinpath("images")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def audio_samples_dir(self) -> Path:
        directory = self.samples_dir.joinpath("audio")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def get_progress_bar_dict(self) -> dict[str, Any]:
        tqdm_dict = super().get_progress_bar_dict()
        to_drop = {k for k in tqdm_dict if k not in PROGRESS_BAR_STATS}
        for k in to_drop:
            tqdm_dict.pop(k, None)
        return tqdm_dict

    def log(self, name: str, value: METRIC_COLLECTION, **kwargs: Any) -> None:
        super().log(name, value=value, **kwargs)

        if name in self.best_tracker:
            value = float(value)
            if self.best_tracker.update(name, value, current_step=self.global_step):
                self.log(f"{name}_best", value=value)

    def forward(self, x: torch.Tensor, ema: bool = True) -> torch.Tensor:
        g = self.g_ema if ema else self.g
        return g(x)

    def d_loss_calculator(
        self,
        x_audio: torch.Tensor,
        X_spec: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss.discriminator_loss(
            real_batch=x_audio,
            fake_batch=self.g(X_spec).detach(),
            global_step=self.global_step,
        )

    def g_loss_calculator(
        self,
        x_audio: torch.Tensor,
        X_spec: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss.generator_loss(
            real_batch=x_audio,
            fake_batch=self.g(X_spec),
            global_step=self.global_step,
        )

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        X_spec = self.melspec(batch)
        if optimizer_idx == 0:  # Train Discriminator
            d_loss = self.d_loss_calculator(batch, X_spec=X_spec)
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss
        elif optimizer_idx == 1:  # Train Generator
            g_loss = self.g_loss_calculator(batch, X_spec=X_spec)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss
        else:
            raise IndexError(f"Invalid optimizer_idx {optimizer_idx}")

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
    ) -> None:
        if optimizer_idx == 0:  # discriminator
            self.log("d_total_norm", compute_total_norm(self.d.parameters()))
        elif optimizer_idx == 1:  # generator
            self.log(f"g_total_norm", compute_total_norm(self.g.parameters()))
        else:
            raise IndexError(f"Invalid optimizer_idx {optimizer_idx}")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Any,
        optimizer_idx: int = 0,
        optimizer_closure: Any = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
        )
        if optimizer_idx == 1:
            update_average(self.g_ema, model_src=self.g, beta=self.hparams.ema_decay)

    def _write_samples(self, x: torch.Tensor) -> None:  # real data as input
        X_real = self.melspec(x)
        x_inv = self.forward(X_real, ema=True)
        X_inv = self.melspec(x_inv)

        # Save Raw Audio
        audio_saver(
            tracks=x_inv.cpu(),
            sample_rate=self.hparams.sample_rate,
            directory=self.audio_samples_dir.joinpath(f"epoch_{self.current_epoch}"),
        )

        # Save Images
        spec_grid_saver(
            F.interpolate(
                torch.cat([X_real, X_inv], dim=1).unsqueeze(1).cpu(),
                size=(self.melspec.n_mels * 2, 256),
                mode="bilinear",
            ).squeeze(1),
            nrow=8,
            n_chunks=1,
            path=self.image_samples_dir.joinpath(f"epoch_{self.current_epoch}.png"),
            color_map="turbo",
        )

    @torch.inference_mode()
    def save_samples(self) -> None:
        if self.val_batch is None:
            self.val_batch = next(iter(self.trainer.val_dataloaders[0]))

        self._write_samples(self.val_batch.to(self.device))

    def validate(self) -> bool:
        return self.current_epoch % 2 == 0

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Optional[torch.Tensor]:
        if self.validate():
            X_real = self.melspec(batch)
            x_inv = self.forward(X_real, ema=True)
            X_inv = self.melspec(x_inv)
            return F.l1_loss(X_inv, target=X_real)

    def validation_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        if self.validate():
            self.log("val_loss_mel", torch.mean(torch.stack(outputs)), prog_bar=True)
        self.save_samples()

    def configure_optimizers(self) -> tuple[list[torch.optim.AdamW], list[Any]]:
        # Discriminator
        opt_d = torch.optim.AdamW(
            self.d.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.adam_betas,
        )
        scheduler_d = ExponentialLR(
            opt_d,
            gamma=self.hparams.lr_decay,
            last_epoch=self.current_epoch - 1,
        )
        # Generator
        opt_g = torch.optim.AdamW(
            self.g.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.adam_betas,
        )
        scheduler_g = ExponentialLR(
            opt_g,
            gamma=self.hparams.lr_decay,
            last_epoch=self.current_epoch - 1,
        )
        return [opt_d, opt_g], [scheduler_d, scheduler_g]
