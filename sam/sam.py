from typing import Iterable

import torch
from torch.optim import Optimizer

from sam.utils import compute_sam

__all__ = ["SAM"]


class SAM(Optimizer):
    """ SAM wrapper for optimizers

    All credits: https://github.com/moskomule/sam.pytorch
    Args:
        params (Iterable): tensors to be optimized
        optim (torch.optim.Optimizer): PyTorch optimizer
        rho (Float, optional): Neighbourhood size, default=0.05
    """

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 optim: Optimizer,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.optim = optim

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns: the loss value evaluated on the original point
        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            compute_sam(group, closure)

        self.optim.step()
        return loss
