"""

    Saving

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import soundfile
import torch
from matplotlib import cm as matplotlib_cm
from pathos.threading import ThreadPool
from PIL import Image
from torchvision.utils import make_grid


def normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = (data.max() - data.min()) + eps  # noqa
    return (data - data.min()) / denom  # noqa


def to_image(
    data: Union[np.ndarray, torch.Tensor],
    color_map: str = "viridis",
    asarray: bool = False,
) -> Image.Image | np.ndarray:
    # See: https://stackoverflow.com/a/14877059
    data = np.asarray(data)
    cmap = matplotlib_cm.get_cmap(color_map)  # noqa
    data_normalized = 255 * cmap(normalize(data))
    image = Image.fromarray(data_normalized.astype("uint8"))
    return np.asarray(image) if asarray else image


def audio_saver(
    tracks: torch.Tensor,
    sample_rate: int,
    directory: Path,
    make_filename: Callable[[int], str] = lambda e: f"{e}.wav",
) -> Image:
    if tracks.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tracks.ndim}D")
    directory.mkdir(parents=True, exist_ok=True)

    def _save_file(data: tuple[int, torch.Tensor]) -> None:
        idx, tensor = data
        soundfile.write(
            file=directory.joinpath(make_filename(idx)),
            data=tensor.numpy(),
            samplerate=sample_rate,
        )

    pool = ThreadPool()
    for _ in pool.imap(_save_file, enumerate(tracks)):
        pass


def spec2grid(
    images: torch.Tensor,
    nrow: int = 8,
    n_chunks: int = 1,
    color_map: str = "viridis",
    **kwargs: Any,
) -> torch.Tensor:
    chunks = list()
    for i in images:
        cmapped_images = np.concatenate(
            [
                to_image(c, asarray=True, color_map=color_map)
                for c in i.chunk(n_chunks, dim=0)
            ]
        )
        chunks.append(torch.as_tensor(cmapped_images))

    images = torch.stack(chunks).permute(0, 3, 1, 2)
    return make_grid(images, nrow=nrow, normalize=False, **kwargs)


def spec2pilgrid(
    images: torch.Tensor,
    nrow: int = 8,
    n_chunks: int = 1,
    **kwargs: Any,
) -> Image:
    image_grid = spec2grid(images, nrow=nrow, n_chunks=n_chunks, **kwargs)
    img = Image.fromarray(image_grid.permute(1, 2, 0).numpy().astype("uint8"))
    return img


def spec_grid_saver(
    images: torch.Tensor,
    path: Path,
    nrow: int = 8,
    n_chunks: int = 1,
    **kwargs: Any,
) -> Image:
    img = spec2pilgrid(images, nrow=nrow, n_chunks=n_chunks, **kwargs)
    img.save(str(path))
    return img
