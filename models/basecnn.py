import torch
import torch.nn as nn
from layers import FFGaussConv2d, HSConv2d, DropoutConv2d, MAPConv2d, FFGaussDense, HSDense, DropoutDense, MAPDense, KernelDense, KernelConv2, KernelDenseBayesian, KernelBayesianConv2, OrthogonalDense, OrthogonalConv2d, OrthogonalBayesianDense, OrthogonalBayesianConv2d
from utils import get_flat_fts
from copy import deepcopy


class BaseCNN(nn.Module):
    def __init__(self, num_classes, input_size=(3, 32, 32), conv_dims=(96, 128, 256), fc_dims=(2048, 2048),
                 model_size=1, type_net='hs', N=50000, beta_ema=0.):
        super(BaseCNN, self).__init__()
        # assert(len(conv_dims) == 3)
        # assert(len(fc_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.model_size = model_size
        self.input_size = input_size
        self.N = N
        self.beta_ema = beta_ema
        self.kernel_sizes = [5, 5, 5]
        self.droprates = [0.1, 0.25, 0.25]
        self.prior_stds_z = [1, 1, 1, 1, 1, 1]

        if type_net == 'hs':
            self.conv_layer = HSConv2d
            self.fc_layer = HSDense
        elif type_net == 'gauss':
            self.conv_layer = FFGaussConv2d
            self.fc_layer = FFGaussDense
        elif type_net == 'dropout':
            self.conv_layer = DropoutConv2d
            self.fc_layer = DropoutDense
        elif type_net == 'map':
            self.conv_layer = MAPConv2d
            self.fc_layer = MAPDense
        elif type_net == 'kernel':
            self.conv_layer = KernelConv2
            self.fc_layer = KernelDense
        elif type_net == 'kernelbayes':
            self.conv_layer = KernelBayesianConv2
            self.fc_layer = KernelDenseBayesian
        elif type_net == 'orth':
            self.conv_layer = OrthogonalConv2d
            self.fc_layer = OrthogonalDense
        elif type_net == 'orthbayes':
            self.conv_layer = OrthogonalBayesianConv2d
            self.fc_layer = OrthogonalBayesianDense
        else:
            raise Exception()

        convs = []
        for i, dim in enumerate(self.conv_dims):
            in_channels = input_size[0] if i == 0 else self.conv_dims[i - 1] * self.model_size
            droprate = self.droprates[i]
            convs += [self.conv_layer(in_channels, dim * self.model_size, self.kernel_sizes[i], droprate=droprate,
                                      padding=2, prior_std_z=self.prior_stds_z[i]),
                      nn.ReLU(), nn.MaxPool2d(3, stride=2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [self.fc_layer(flat_fts, self.fc_dims[0] * model_size,
                             prior_std_z=self.prior_stds_z[-3]), nn.ReLU(),
               self.fc_layer(self.fc_dims[0] * model_size, self.fc_dims[1] * model_size,
                             prior_std_z=self.prior_stds_z[-2]), nn.ReLU(),
               self.fc_layer(self.fc_dims[-1] * model_size, num_classes,
                             prior_std_z=self.prior_stds_z[-1])]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, self.fc_layer) or isinstance(m, self.conv_layer):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o), x.var(), self.fcs[4](self.fcs[3](self.fcs[2](self.fcs[1](self.fcs[0](o))))).var()

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

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data = deepcopy(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

