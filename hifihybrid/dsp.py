"""

    DSP

"""
from functools import cached_property
from typing import Optional

import torch
from librosa.filters import mel as mel_filter
from torch import nn
from torch.nn import functional as F


def dynamic_range_compression(
    y: torch.Tensor,
    C: int = 1,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    return torch.log(torch.clamp(y, min=clip_val) * C)


class MelSpec(nn.Module):
    mel_basis: torch.Tensor
    stft_window: torch.Tensor

    def __init__(
        self,
        seq_len: int,
        n_mels: int = 128,
        n_fft: int = 1024,
        sr: int = 22_050,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        center: bool = False,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.eps = eps

        self.register_buffer(
            "stft_window",
            torch.hann_window(self.win_length),
        )
        self.register_buffer(
            "mel_basis",
            torch.from_numpy(
                mel_filter(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            ),
        )

    @cached_property
    def shape(self) -> tuple[int, int]:
        y = self.forward(torch.randn(1, self.seq_len))
        return tuple(y.shape[1:])

    @property
    def n_frames(self) -> int:
        return self.shape[-1]

    def _pad(self, y: torch.Tensor) -> torch.Tensor:
        padding = self.n_fft - self.hop_length
        return F.pad(y, (padding // 2, padding // 2), mode="reflect")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, *_ = y.shape
        spec = torch.stft(
            input=self._pad(y),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.abs().pow(2).add(self.eps)
        spec = torch.bmm(self.mel_basis[None, ...].repeat(B, 1, 1), spec)
        return dynamic_range_compression(spec)


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    PLOT: bool = False
    SEQ_LEN: int = 2**16

    self = MelSpec(SEQ_LEN)

    y, _ = librosa.load(librosa.ex("trumpet"), sr=self.sr, duration=SEQ_LEN / self.sr)
    y = torch.from_numpy(y).unsqueeze(0).repeat(8, 1)

    X = self(y)
    assert X.shape == (y.shape[0], self.n_mels, self.n_frames)

    if PLOT:
        plt.imshow(X[0], aspect="auto", cmap="turbo")
        plt.show()
