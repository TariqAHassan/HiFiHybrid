"""

    Reproducibility

    References:
        * https://www.kaggle.com/code/bminixhofer/deterministic-neural-networks-using-pytorch/notebook
        * https://pytorch.org/docs/stable/notes/randomness.html

"""
import os
import random

import numpy as np
import torch

SEED_VALUE: int = 42


def seed_everything(
    random_lib: bool = True,
    python: bool = True,
    numpy_lib: bool = True,
    torch_lib: bool = True,
    deterministic_cudnn: bool = True,  # faster but slower
) -> None:
    if random_lib:
        random.seed(SEED_VALUE)
    if python:
        os.environ["PYTHONHASHSEED"] = str(SEED_VALUE)
    if numpy_lib:
        np.random.seed(SEED_VALUE)
    if torch_lib:
        torch.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed(SEED_VALUE)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = deterministic_cudnn  # noqa
