from typing import Iterable, Callable

import torch
from torch.optim import Optimizer


def compute_sam(group: dict, closure: Callable):
    grads = []
    params_with_grads = []

    rho = group['rho']
    # update internal_optim's learning rate

    for p in group['params']:
        if p.grad is not None:
            # without clone().detach(), p.grad will be zeroed by closure()
            grads.append(p.grad.clone().detach())
            params_with_grads.append(p)
    device = grads[0].device

    # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
    grad_norm = torch.stack(
        [g.detach().norm(2).to(device) for g in grads]).norm(2)
    epsilon = grads  # alias for readability
    torch._foreach_mul_(epsilon, rho / grad_norm)

    # virtual step toward \epsilon
    torch._foreach_add_(params_with_grads, epsilon)
    # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
    closure()
    # virtual step back to the original point
    torch._foreach_sub_(params_with_grads, epsilon)
