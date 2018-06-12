import numpy as np
import torch
from torch.optim import Optimizer


class VProp(Optimizer):
    """Implements Vprop algorithm.

    This algorithm is based on the paper: <https://arxiv.org/pdf/1712.01038.pdf>
    """

    def __init__(self,
                 params,
                 lr: float =1e-3,
                 beta: float =1e-1,
                 delta: float = 1.,
                 noise_scale: float = 1e-3):

        assert lr > 0.0, f"Invalid learning rate: {lr}"
        assert beta > 0.0, f"Invalid beta: {beta}"
        assert delta > 0.0, f"Invalid delta: {delta}"
        assert noise_scale > 0.0, f"Invalid noise_scale: {noise_scale}"

        self.last_noise = {}

        defaults = dict(lr=lr,
                        beta=beta,
                        delta=delta,
                        noise_scale=noise_scale)
        super(VProp, self).__init__(params, defaults)

    def add_noise_to_parameters(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'square_avg' not in state:
                        state['square_avg'] = torch.zeros_like(p)

                        denom = state['square_avg'].add(group['delta']).sqrt()

                        if torch.cuda.is_available():
                            self.last_noise[p] = torch.randn(p.data.size(), device=torch.device('cuda'))
                        else:
                            self.last_noise[p] = torch.randn(p.data.size())

                        self.last_noise[p].div_(denom).mul_(group['noise_scale'])

                        p.data.add_(self.last_noise[p])

    def remove_noise_from_parameters(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p in self.last_noise:
                            p.data.add_(-self.last_noise[p])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                # State initialization
                state = self.state[p]

                if 'square_avg' not in state:
                    state['square_avg'] = torch.zeros_like(p.data)

                if 'step' not in state:
                    state['step'] = 0

                state['step'] += 1

                if p.grad is None:
                    continue
                grad = p.grad.data
                assert not grad.is_sparse, "Vprop does not support sparse gradients"

                square_avg = state['square_avg']
                square_avg.mul_(1 - group['beta']).addcmul_(group['beta'], grad, grad)

                num = p.data.mul(group['delta']).add(grad)
                denom = square_avg.add(group['delta'])

                p.data.addcdiv_(-group['lr'], num, denom)

        return loss
