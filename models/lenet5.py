import torch
import torch.nn as nn
from layers import FFGaussConv2d, HSConv2d, DropoutConv2d, MAPConv2d, FFGaussDense, HSDense, DropoutDense, MAPDense, KernelConv2, KernelDense, KernelBayesianConv2, KernelDenseBayesian, OrthogonalDense, OrthogonalConv2d, OrthogonalBayesianConv2d, OrthogonalBayesianDense
from utils import get_flat_fts
from copy import deepcopy


class LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(1, 28, 28), conv_dims=(32, 64), fc_dims=512, type_net='hs',
                 N=50000, beta_ema=0., dof=1., mask=None):
        super(LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.dof = dof

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
        elif type_net == 'kernelbayesian':
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

        if mask is not None:
            convs = [self.conv_layer(in_channels=input_size[0], out_channels=conv_dims[0], kernel_size=5, droprate=0.5,
                                     dof=dof, mask=mask[0]), nn.ReLU(), nn.MaxPool2d(2),
                     self.conv_layer(in_channels=conv_dims[0], out_channels=conv_dims[1], kernel_size=5, droprate=0.5,
                                     dof=dof, mask=mask[1]), nn.ReLU(), nn.MaxPool2d(2)]
            self.convs = nn.Sequential(*convs)
            if torch.cuda.is_available():
                self.convs = self.convs.cuda()

            flat_fts = get_flat_fts(input_size, self.convs)
            fcs = [self.fc_layer(flat_fts, self.fc_dims, dof=dof, mask=mask[2]), nn.ReLU(),
                   self.fc_layer(self.fc_dims, num_classes, dof=dof)]
            self.fcs = nn.Sequential(*fcs)
        else:
            convs = [self.conv_layer(in_channels=input_size[0], out_channels=conv_dims[0], kernel_size=5, droprate=0.5, dof=dof), nn.ReLU(), nn.MaxPool2d(2),
                     self.conv_layer(in_channels=conv_dims[0], out_channels=conv_dims[1], kernel_size=5, droprate=0.5, dof=dof), nn.ReLU(), nn.MaxPool2d(2)]
            self.convs = nn.Sequential(*convs)
            if torch.cuda.is_available():
                self.convs = self.convs.cuda()

            flat_fts = get_flat_fts(input_size, self.convs)
            fcs = [self.fc_layer(flat_fts, self.fc_dims, dof=dof), nn.ReLU(),
                   self.fc_layer(self.fc_dims, num_classes, dof=dof)]
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
        return self.fcs(o)

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
