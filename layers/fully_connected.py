import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import init


class HSDense(Module):

    def __init__(self, in_features, out_features, bias=True, prior_std=1., prior_std_z=1., dof=1., **kwargs):
        super(HSDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.mean_w = Parameter(torch.Tensor(in_features, out_features))
        self.logvar_w = Parameter(torch.Tensor(in_features, out_features))
        self.qz_mean = Parameter(torch.Tensor(in_features))
        self.qz_logvar = Parameter(torch.Tensor(in_features))
        self.dof = dof
        self.prior_std_z = prior_std_z
        self.use_bias = False
        if bias:
            self.mean_bias = Parameter(torch.Tensor(out_features))
            self.logvar_bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.mean_w, mode='fan_out')
        self.logvar_w.data.normal_(-9., 1e-4)

        self.qz_mean.data.normal_(math.log(math.exp(1) - 1), 1e-3)
        self.qz_logvar.data.normal_(math.log(0.1), 1e-4)
        if self.use_bias:
            self.mean_bias.data.normal_(0, 1e-2)
            self.logvar_bias.data.normal_(-9., 1e-4)

    def constrain_parameters(self, thres_std=1.):
        self.logvar_w.data.clamp_(max=2. * math.log(thres_std))
        if self.use_bias:
            self.logvar_bias.data.clamp_(max=2. * math.log(thres_std))

    def eq_logpw(self):
        logpw = - .5 * math.log(2 * math.pi * self.prior_std ** 2) - .5 * self.logvar_w.exp().div(self.prior_std ** 2)
        logpw -= .5 * self.mean_w.pow(2).div(self.prior_std ** 2)
        logpb = 0.
        if self.use_bias:
            logpb = - .5 * math.log(2 * math.pi * self.prior_std ** 2) - .5 * self.logvar_bias.exp().div \
                (self.prior_std ** 2)
            logpb -= .5 * self.mean_bias.pow(2).div(self.prior_std ** 2)
        return torch.sum(logpw) + torch.sum(logpb)

    def eq_logqw(self):
        logqw = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_w + 1))
        logqb = 0.
        if self.use_bias:
            logqb = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_bias + 1))
        return logqw + logqb

    def kldiv_aux(self):
        z = self.sample_z(1)
        z = z.view(self.in_features)

        logqm = - torch.sum(.5 * (math.log(2 * math.pi) + self.qz_logvar + 1))
        logqm = logqm.add(- torch.sum(F.sigmoid(z.exp().add(- 1).log()).log()))

        logpm = torch.sum(
            2 * math.lgamma(.5 * (self.dof + 1)) - math.lgamma(.5 * self.dof) - math.log(self.prior_std_z) -
            .5 * math.log(self.dof * math.pi) -
            .5 * (self.dof + 1) * torch.log(1. + z.pow(2) / (self.dof * self.prior_std_z ** 2)))

        return logpm - logqm

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def get_eps(self, size):
        eps = self.floatTensor(size).normal_()
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size):
        z = self.qz_mean.view(1, self.in_features)
        if self.training:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = z + eps.mul(self.qz_logvar.view(1, self.in_features).mul(0.5).exp_())
        return F.softplus(z)

    def sample_W(self):
        W = self.mean_w
        if self.training:
            eps = self.get_eps(self.mean_w.size())
            W = W.add(eps.mul(self.logvar_w.mul(0.5).exp_()))
        return W

    def sample_b(self):
        b = self.mean_bias
        if self.training:
            eps = self.get_eps(self.mean_bias.size())
            b = b.add(eps.mul(self.logvar_bias.mul(0.5).exp_()))
        return b

    def get_mean_x(self, input):
        mean_xin = input.mm(self.mean_w)
        if self.use_bias:
            mean_xin = mean_xin.add(self.mean_bias.view(1, self.out_features))

        return mean_xin

    def get_var_x(self, input):
        var_xin = input.pow(2).mm(self.logvar_w.exp())
        if self.use_bias:
            var_xin = var_xin.add(self.logvar_bias.exp().view(1, self.out_features))

        return var_xin

    def forward(self, input):
        # sampling
        batch_size = input.size(0)
        z = self.sample_z(batch_size)
        xin = input.mul(z)
        mean_xin = self.get_mean_x(xin)
        output = mean_xin
        if self.training:
            var_xin = self.get_var_x(xin)
            eps = self.get_eps(self.floatTensor(batch_size, self.out_features))
            output = output.add(var_xin.sqrt().mul(eps))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ', dof: ' \
               + str(self.dof) + ', prior_std_z: ' \
               + str(self.prior_std_z) + ')'


class FFGaussDense(Module):

    def __init__(self, in_features, out_features, bias=True, prior_std=1., **kwargs):
        super(FFGaussDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.mean_w = Parameter(torch.Tensor(in_features, out_features))
        self.logvar_w = Parameter(torch.Tensor(in_features, out_features))
        self.use_bias = False
        if bias:
            self.mean_bias = Parameter(torch.Tensor(out_features))
            self.logvar_bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.mean_w, mode='fan_out')
        self.logvar_w.data.normal_(-9., 1e-4)

        if self.use_bias:
            self.mean_bias.data.zero_()
            self.logvar_bias.data.normal_(-9., 1e-4)

    def constrain_parameters(self, thres_std=1.):
        self.logvar_w.data.clamp_(max=2. * math.log(thres_std))
        if self.use_bias:
            self.logvar_bias.data.clamp_(max=2. * math.log(thres_std))

    def eq_logpw(self):
        logpw = - .5 * math.log(2 * math.pi * self.prior_std ** 2) - .5 * self.logvar_w.exp().div(self.prior_std ** 2)
        logpw -= .5 * self.mean_w.pow(2).div(self.prior_std ** 2)
        logpb = 0.
        if self.use_bias:
            logpb = - .5 * math.log(2 * math.pi * self.prior_std ** 2) - .5 * self.logvar_bias.exp().div(self.prior_std ** 2)
            logpb -= .5 * self.mean_bias.pow(2).div(self.prior_std ** 2)
        return torch.sum(logpw) + torch.sum(logpb)

    def eq_logqw(self):
        logqw = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_w + 1))
        logqb = 0.
        if self.use_bias:
            logqb = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_bias + 1))
        return logqw + logqb

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def get_eps(self, size):
        eps = self.floatTensor(size).normal_()
        eps = Variable(eps)
        return eps

    def sample_pW(self):
        return self.floatTensor(self.in_features, self.out_features).normal_()

    def sample_pb(self):
        return self.floatTensor(self.out_features).normal_()

    def sample_W(self):
        w = self.mean_w
        if self.training:
            eps = self.get_eps(self.mean_w.size())
            w = w.add(eps.mul(self.logvar_w.mul(0.5).exp_()))
        return w

    def sample_b(self):
        b = self.mean_bias
        if self.training:
            eps = self.get_eps(self.mean_bias.size())
            b = b.add(eps.mul(self.logvar_bias.mul(0.5).exp_()))
        return b

    def get_mean_x(self, input):
        mean_xin = input.mm(self.mean_w)
        if self.use_bias:
            mean_xin = mean_xin.add(self.mean_bias.view(1, self.out_features))
        return mean_xin

    def get_var_x(self, input):
        var_xin = input.pow(2).mm(self.logvar_w.exp())
        if self.use_bias:
            var_xin = var_xin.add(self.logvar_bias.exp().view(1, self.out_features))

        return var_xin

    def forward(self, input):
        batch_size = input.size(0)
        mean_xin = self.get_mean_x(input)
        if self.training:
            var_xin = self.get_var_x(input)
            eps = self.get_eps(self.floatTensor(batch_size, self.out_features))
            output = mean_xin.add(var_xin.sqrt().mul(eps))
        else:
            output = mean_xin
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', prior_std: ' \
            + str(self.prior_std) + ')'


class DropoutDense(Module):

    def __init__(self, in_features, out_features, bias=True, droprate=0.5, weight_decay=1.,
                 **kwargs):
        super(DropoutDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.droprate = droprate
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.weight_decay = weight_decay
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out')

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, thres_std=1.):
        pass

    def eq_logpw(self, **kwargs):
        logpw = - (1 - self.droprate) * torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.eq_logpw() - self.eq_logqw() + self.kldiv_aux()

    def get_eps(self, size):
        eps = self.floatTensor(size).fill_(1. - self.droprate)
        if self.droprate > 0 and self.training:
            eps = torch.bernoulli(eps)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size):
        z = self.get_eps(self.floatTensor(batch_size, self.in_features))
        return z

    def forward(self, input_):
        # sampling
        if self.droprate > 0:
            z = self.sample_z(input_.size(0))
            input_ = input_ * z

        output = input_.mm(self.weight)
        if self.bias is not None:
            output.add_(self.bias.view(1, self.out_features).expand_as(output))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', droprate: ' \
            + str(self.droprate) + ', weight_decay: ' \
            + str(self.weight_decay) + ')'


class MAPDense(Module):

    def __init__(self, in_features, out_features, bias=True, weight_decay=1., **kwargs):
        super(MAPDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out')

        if self.bias is not None:
            self.bias.data.normal_(0, 1e-2)

    def constrain_parameters(self, thres_std=1.):
        pass

    def eq_logpw(self, **kwargs):
        logpw = - torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.eq_logpw() - self.eq_logqw() + self.kldiv_aux()

    def forward(self, input):
        output = input.mm(self.weight)
        if self.bias is not None:
            output.add_(self.bias.view(1, self.out_features).expand_as(output))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

