import torch
from torch.optim import Optimizer


class COCOBBackprop(Optimizer):
    """Implements COCOBackprop algorithm.

    This algorithm is based on the paper: <https://arxiv.org/abs/1705.07795>
    """

    def __init__(self, params, alpha=100, epsilon=1e-8):

        self._alpha = alpha
        self.epsilon = epsilon
        defaults = dict(alpha=alpha, epsilon=epsilon)
        super(COCOBBackprop, self).__init__(params, defaults)

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
                    state['gradients_sum'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['grad_norm_sum'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['L'] =  torch.ones_like(p.data, dtype=torch.float32, device=self.device).mul(self.epsilon)
                    state['tilde_w'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)
                    state['reward'] = torch.zeros_like(p.data, dtype=torch.float32, device=self.device)

                gradients_sum, grad_norm_sum, tilde_w, l, reward = state['gradients_sum'], state['grad_norm_sum'], state['tilde_w'], state['L'], state['reward']

                l = torch.max(l, grad.abs())
                gradients_sum = gradients_sum.add(grad)
                grad_norm_sum = grad_norm_sum.add(grad.abs())
                reward = reward.sub(grad.mul(tilde_w)).clamp(min=0.)
                new_w = -gradients_sum.div(l.mul(torch.max(grad_norm_sum.add(l), l.mul(self._alpha)))).mul(reward.add(l))
                p.data = p.data.sub(tilde_w).add(new_w)

                state['gradients_sum'], state['grad_norm_sum'], state['tilde_w'], state['L'], state['reward'] = gradients_sum, grad_norm_sum, new_w, l, reward

        return loss
