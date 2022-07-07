"""

    Run HiFi

"""
from pathlib import Path

import torch

from datetime import datetime
from multiprocessing import cpu_count

import fire
from typing import Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hifihybrid.data.datasets.chunk import AudioChunkDataset
from hifihybrid.data.datasets.sized import SizedDataset

from hifihybrid.utils.reproducibility import SEED_VALUE, seed_everything
from hifihybrid.hifi import HiFiHybrid

seed_everything(deterministic_cudnn=False)

START_TIME = datetime.utcnow().strftime("%d-%B-%Y-%Ih-%Mm-%Ss-%p")


def _ensure_exists(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def _get_datasets(
    data_path: Path,
    file_ext: str,
    val_prop: float,
    sample_rate: int,
    train_duration: float,
    val_duration: float,
    sample_per_epoch: int,
    max_sample_per_epoch_val: int = 2**12,
) -> tuple[SizedDataset, SizedDataset]:
    train_files, val_files = train_test_split(
        list(data_path.rglob(f"*.{file_ext}")),
        test_size=val_prop,
        random_state=SEED_VALUE,
    )

    train_datasets = SizedDataset(
        AudioChunkDataset(
            src_files=train_files,
            sr=sample_rate,
            duration=train_duration,
        ),
        size=sample_per_epoch,
    )
    val_datasets = SizedDataset(
        AudioChunkDataset(
            src_files=val_files,
            sr=sample_rate,
            duration=val_duration,
        ),
        size=min(sample_per_epoch, max_sample_per_epoch_val),
    )
    return train_datasets, val_datasets


def main(
    # Data
    data_path: str,
    file_ext: str = "wav",
    val_prop: float = 0.1,
    max_epochs: int = 3_200,
    # Logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    # Training
    batch_size: int = 32,
    batch_ceil: int = 2**10,
    sample_rate: int = 22_050,
    n_mels: int = 128,
    train_seq_len: int = 2**13,
    # Validation
    val_seq_len=2**17,
    # Output
    output_path: Optional[Path] = None,
) -> None:
    """Train Model.

    Args:
        data_path (str): system path where audio samples exist.
        file_ext (str): file extension to filter for in ``data_path``.
        val_prop (float): proportion of files in ``data_path`` to use for validation
        max_epochs (int): the maximum number of epochs to train the model for
        wandb_project (str, optional): the name of the Weights&Biases project to log to.
            If ``None``, Weights&Biases logging will be disabled.
        wandb_entity (str, optional): username of the Weights&Biases account to log to.
            Note: this value will be ignored if ``wandb_project=None``.
        batch_size (int): number of elements to use in each mini-batch
        batch_ceil (int): maximum number of mini-batches to use per epoch
        sample_rate (int): sample rate of audio
        n_mels (int): number of mel bands to use when vocoding
        train_seq_len (int): number of audio samples to use during training the model
        val_seq_len (int): number of audio samples to use when validating the model
        output_path (Path, optional): path to persist samples, checkpoints, etc.
            If ``None``, the current working directory will be used.

    Returns:
        None

    """
    train_dataset, val_dataset = _get_datasets(
        data_path=Path(data_path),
        file_ext=file_ext,
        val_prop=val_prop,
        sample_rate=sample_rate,
        train_duration=train_seq_len / sample_rate,
        val_duration=val_seq_len / sample_rate,
        sample_per_epoch=batch_ceil * batch_size,
    )
    model = HiFiHybrid(
        seq_len=train_seq_len,
        n_mels=n_mels,
        sample_rate=sample_rate,
        start_time=START_TIME,
        base_dir=_ensure_exists(output_path or Path.cwd().joinpath("hifi")),
        dataset=Path(data_path).name,
    )
    Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=max_epochs,
        precision=32,
        limit_train_batches=batch_ceil,
        log_every_n_steps=5,
        logger=(
            WandbLogger(
                project=wandb_project,
                name=START_TIME,
                entity=wandb_entity,
                anonymous="never",
            )
            if wandb_project
            else None
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(model.checkpoints_dir),
                filename="{epoch:02d}-{val_loss_mel:.2f}",
                monitor="val_loss_mel",
                mode="min",
                save_top_k=3,
                save_last=True,
                every_n_epochs=2,
            )
        ],
    ).fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(SEED_VALUE),
        ),
    )


if __name__ == "__main__":
    fire.Fire(main)
