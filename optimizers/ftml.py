import math
import torch
from torch.optim import Optimizer


class FTML(Optimizer):
    """Implements Follow-the-Moving-Leader algorithm.

    This algorithm is based on the paper: <http://proceedings.mlr.press/v70/zheng17a.html>
    """

    def __init__(self,
                 params,
                 lr: float,
                 beta1: float = 0.6,
                 beta2: float = 0.999,
                 epsilon=1e-8):

        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(FTML, self).__init__(params, defaults)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def step(self, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['v'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['d'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['sigma'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['z'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['t'] = 1

                v, d, sigma, z, t = state['v'], state['d'], state['sigma'], state['z'], state['t']

                bexp1 = math.pow(self._beta1, t)
                bexp2 = math.pow(self._beta2, t)
                v = v.mul(self._beta2).add(grad.pow(2).mul(1.-self._beta2))
                d_new = v.div(1.-bexp2).sqrt().add(self._epsilon).mul(1.-bexp1).div(self._lr)
                sigma_new = d_new.sub(d.mul(self._beta1))
                z = z.mul(self._beta1).add(grad.mul(1.-self._beta1)).sub(sigma.mul(p.data))
                p.data = -z.div(d_new)

                t += 1
                state['v'], state['d'], state['sigma'], state['z'], state['t'] = v, d_new, sigma_new, z, t

        return loss
