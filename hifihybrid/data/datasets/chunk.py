"""

    Chunk Audio Dataset

"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import librosa
import torch
from librosa.util import normalize
from numpy.random import uniform as rand_uniform
from torch.nn import functional as F


class AudioChunkDataset:
    """Dataset for randomly intercepting audio files.

    Args:
        src_files (list[Path]): path to the files
        src2duration (dict, optional): a file -> duration mapping.
            If None, durations will be determined using librosa.
        sr (int): sample rate
        duration (int): duration of audio chunks
        res_type (str): resampling algorithm to use
        random_offset (bool): randomly choose the offset when generating
            an audio chunk. If ``False``, no offset will be used.
    """

    def __init__(
        self,
        src_files: list[Path],
        src2duration: Optional[dict[Path, float]] = None,
        sr: int = 22_050,
        duration: Optional[int | float] = 10,
        res_type: str = "kaiser_fast",
        random_offset: bool = True,
    ) -> None:
        self.src_files = src_files
        self.src2duration = src2duration or dict()
        self.sr = sr
        self.duration = duration
        self.res_type = res_type
        self.random_offset = random_offset

    def _gaurentee_length(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        if self.duration is None:
            return x

        segment_size = round(self.sr * self.duration)
        if len(x) < segment_size:
            return F.pad(
                input=x,
                pad=(0, segment_size - len(x)),
                mode="constant",
                value=0.0,
            )
        elif len(x) > segment_size:
            return x[:segment_size]
        else:
            return x

    def _load(self, file: Path, offset: float = 0) -> torch.Tensor:
        wav, _ = librosa.load(
            path=file,
            sr=self.sr,
            mono=True,
            res_type=self.res_type,
            offset=offset,
            duration=self.duration,
        )
        return self._gaurentee_length(torch.from_numpy(normalize(wav)))

    def __len__(self) -> int:
        return len(self.src_files)

    @lru_cache()
    def _get_duration(self, file: Path) -> float:
        try:
            return self.src2duration[file]
        except KeyError:
            return librosa.get_duration(filename=file)

    def _get_offset(self, file: Path) -> float:
        if self.random_offset:
            return rand_uniform(0, max(0, self._get_duration(file) - self.duration))
        else:
            return 0.0

    def __getitem__(self, item: int) -> torch.Tensor:
        file = self.src_files[item]
        return self._load(file, offset=self._get_offset(file))
