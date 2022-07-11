"""

    Training

"""
import math
from typing import Iterable, Optional, Union

import torch
from torch import nn


def count_parameters(model: nn.Module, grad_req: bool = True) -> int:
    # See https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad is grad_req)


def toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(
    model_tgt: nn.Module,
    model_src: nn.Module,
    beta: Optional[Union[int, float]],
) -> None:
    # Adapted from https://github.com/akanimax/BMSG-GAN
    toggle_grad(model_tgt, requires_grad=False)
    toggle_grad(model_src, requires_grad=False)

    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        if p_src is p_tgt:
            raise ValueError("p_src == p_tgt")

        if beta is None:
            p_tgt.copy_(p_src)
        else:
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

    toggle_grad(model_tgt, requires_grad=True)
    toggle_grad(model_src, requires_grad=True)


def compute_total_norm(
    parameters: Iterable[torch.Tensor],
    norm_type: float = 2.0,
) -> torch.Tensor:
    # From torch.nn.utils.clip_grad.clip_grad_norm_.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    if norm_type == math.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            p=norm_type,
        )
    return total_norm
