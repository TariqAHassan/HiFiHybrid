"""

    Sized

"""
from typing import Any

from torch.utils.data import Dataset
from hifihybrid.data.datasets.chunk import AudioChunkDataset


class SizedDataset(Dataset):
    """Adapt a finite AudioChunkDataset into one of arbitrary size."""
    def __init__(self, dataset: AudioChunkDataset, size: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int) -> Any:
        # See https://stackoverflow.com/a/22122635
        return self.dataset.__getitem__(item % len(self.dataset.src_files))
