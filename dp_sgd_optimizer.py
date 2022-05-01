from typing import Optional

import torch
from torch.optim import SGD, Optimizer


class DPSGD(SGD):
    def __init__(self, params, lr, noise_scale, group_size, grad_norm_bound):
        self.lr = lr
        self.noise_scale = noise_scale
        self.group_size = group_size
        self.grad_norm_bound = grad_norm_bound

        super(DPSGD, self).__init__(params, lr)

        for group in self.param_groups:
            group['clipped_grads'] = [torch.zeros_like(param) if param.requires_grad else None for param in group['params']]

    def per_sample_gradient_clip(self):
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.requires_grad:
                    grad = param.grad.data
                    clipped_grad = grad / max(1., grad.norm(2) / self.grad_norm_bound)
                    group['clipped_grads'][i].add_(clipped_grad)

    def zero_grad(self, set_to_none: Optional[bool]=...) -> None:
        for group in self.param_groups:
            group['clipped_grads'] = [torch.zeros_like(param) if param.requires_grad else None for param in
                                      group['params']]
        super(DPSGD, self).zero_grad(set_to_none=set_to_none)

    def zero_grad_per_sample(self, set_to_none: Optional[bool]=...):
        super(DPSGD, self).zero_grad(set_to_none=set_to_none)

    def step(self, *args, **kwargs):
        for group in self.param_groups:
            for i, clipped_grad in enumerate(group['clipped_grads']):
                group['params'][i].grad = (clipped_grad / self.group_size + torch.randn_like(clipped_grad) * ((self.noise_scale * self.grad_norm_bound) ** 2)) / self.group_size
        super(DPSGD, self).step(*args, **kwargs)
