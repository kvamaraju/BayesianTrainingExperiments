import torch
import torch.nn as nn
from layers import FFGaussDense, HSDense, DropoutDense, MAPDense, KernelDense, KernelDenseBayesian, OrthogonalDense, OrthogonalBayesianDense
from copy import deepcopy


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, layer_dims=(1024, 1024), type_net='hs', N=50000, dof=1., beta_ema=0.):
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.dof = dof
        self.beta_ema = beta_ema
        self.priors_std_z = [1., 1., 1.]

        if type_net == 'hs':
            self.fc_layer = HSDense
        elif type_net == 'dropout':
            self.fc_layer = DropoutDense
        elif type_net == 'map':
            self.fc_layer = MAPDense
        elif type_net == 'gauss':
            self.fc_layer = FFGaussDense
        elif type_net == 'kernel':
            self.fc_layer = KernelDense
        elif type_net == 'kernelbayes':
            self.fc_layer = KernelDenseBayesian
        elif type_net == 'orth':
            self.fc_layer = OrthogonalDense
        elif type_net == 'orthbayes':
            self.fc_layer = OrthogonalBayesianDense
        else:
            raise Exception()

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate = 0.2 if i == 0 else 0.5
            layers += [self.fc_layer(inp_dim, dimh, droprate=droprate, dof=self.dof, prior_std_z=self.priors_std_z[i]),
                       nn.ReLU()]

        layers.append(self.fc_layer(self.layer_dims[-1], num_classes, dof=self.dof, prior_std_z=self.priors_std_z[-1]))
        self.output = nn.Sequential(*layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, self.fc_layer):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        return self.output(x)

    def kl_div(self, annealing=1., type_anneal='kl'):
        logps, logqs, aux_kls = 0., 0., 0.
        for layer in self.layers:
            logp, logq, aux_kl = layer.eq_logpw(), layer.eq_logqw(), layer.kldiv_aux()
            logps += - (1. / self.N) * logp
            logqs += (1. / self.N) * logq
            aux_kls += - (1. / self.N) * aux_kl
        if type_anneal == 'kl':
            regularization = annealing * (aux_kls + logps + logqs)
        elif type_anneal == 'q':
            regularization = aux_kls + logps + annealing * logqs
        else:
            regularization = aux_kls + logps + logqs
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def mmd(self,
            scale_factor: float = 1.,
            num_samples: int = 2):

        def euclidean_kernel(x: torch.Tensor,
                             y: torch.Tensor):
            return torch.dist(input=x, other=y, p=2).pow(2)

        def imq_kernel(x: torch.Tensor,
                       y: torch.Tensor):
            return torch.dist(input=x, other=y, p=2).pow(2).add(1).reciprocal()

        def plummer_kernel(x: torch.Tensor,
                           y: torch.Tensor):
            return torch.dist(input=x, other=y, p=2).pow(2).add(1e-10).sqrt().reciprocal()

        def mmd_estimate(x: list,
                         y: list):

            kernel = plummer_kernel

            term1 = 0.
            term2 = 0.

            assert len(x) == len(y)
            num_entries = len(x)

            for i in range(num_entries):
                for j in range(num_entries):
                    if i != j:
                        term1 += kernel(x[i], x[j]) + kernel(y[i], y[j])
                    term2 += kernel(x[i], y[j])

            term1 /= num_entries * (num_entries - 1)
            term2 *= 2 / (num_entries * num_entries)

            return term1 - term2

        regularization = 0.

        for layer in self.layers:
            w_true = [layer.sample_W() for _ in range(num_samples)]
            w_prior = [layer.sample_pW() for _ in range(num_samples)]

            regularization += mmd_estimate(x=w_true,
                                           y=w_prior)

            if layer.use_bias:
                b_true = [layer.sample_b() for _ in range(num_samples)]
                b_prior = [layer.sample_pb() for _ in range(num_samples)]

                regularization += mmd_estimate(x=b_true,
                                               y=b_prior)
        regularization *= scale_factor

        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
