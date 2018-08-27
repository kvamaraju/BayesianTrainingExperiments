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


class KernelDense(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dim: int = 2,
                 use_bias=True,
                 **kwargs):
        super(KernelDense, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.columns = Parameter(self.floatTensor(in_features, dim))
        self.rows = Parameter(self.floatTensor(out_features, dim))
        self.alpha = Parameter(self.floatTensor(in_features))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Parameter(self.floatTensor(out_features))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.rows.data.normal_()
        self.columns.data.normal_()
        self.alpha.data.normal_()
        if self.use_bias:
            self.bias.data.normal_(std=1e-5)

    def eq_logpw(self, **kwargs) -> torch.Tensor:
        logpw = - torch.sum(.5 * (self._calc_rbf_weights(rows=self.rows,
                                                         columns=self.columns).pow(2)))
        logpb = 0
        if self.use_bias:
            logpb = - torch.sum(.5 * (self._calc_rbf_weights(rows=self.rows,
                                                             columns=self.columns).pow(2)))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def _calc_rbf_weights(self,
                          rows: torch.Tensor,
                          columns: torch.Tensor) -> torch.Tensor:
        x2 = rows.pow(2).sum(dim=1).view(1, self.out_features)
        y2 = columns.pow(2).sum(dim=1).view(self.in_features, 1)
        xy = columns.mm(rows.t()).mul(-2.)

        return x2.add(y2).add(xy).mul(-1).exp()

    def forward(self,
                input: torch.Tensor):

        w = self._calc_rbf_weights(rows=self.rows,
                                   columns=self.columns)
        y = input.mul(self.alpha).mm(w)

        if self.use_bias:
            return y.add(self.bias.view(1, self.out_features))
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', dim: ' \
            + str(self.dim) + ')'


class KernelDenseBayesian(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dim: int = 2,
                 use_bias: bool = True,
                 prior_std: float = 1.,
                 bias_std: float = 1e-3,
                 **kwargs):
        super(KernelDenseBayesian, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.prior_std = prior_std
        self.bias_std = bias_std

        self.columns_mean = Parameter(self.floatTensor(self.in_features, self.dim))
        self.columns_logvar = Parameter(self.floatTensor(self.in_features, self.dim))

        self.rows_mean = Parameter(self.floatTensor(self.out_features, self.dim))
        self.rows_logvar = Parameter(self.floatTensor(self.out_features, self.dim))

        self.alpha_mean = Parameter(self.floatTensor(self.in_features))
        self.alpha_logvar = Parameter(self.floatTensor(self.in_features))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_mean = Parameter(self.floatTensor(self.out_features))
            self.bias_logvar = Parameter(self.floatTensor(self.out_features))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.columns_mean.data.normal_(std=self.prior_std)
        self.columns_logvar.data.normal_(std=self.prior_std)

        self.rows_mean.data.normal_(std=self.prior_std)
        self.rows_logvar.data.normal_(std=self.prior_std)

        self.alpha_mean.data.normal_(std=self.prior_std)
        self.alpha_logvar.data.normal_(std=self.prior_std)

        if self.use_bias:
            self.bias_mean.data.normal_(std=self.bias_std)
            self.bias_logvar.data.normal_(std=self.bias_std)

    def _calc_rbf_weights(self,
                          rows: torch.Tensor,
                          columns: torch.Tensor):
        x2 = rows.pow(2).sum(dim=1).view(1, self.out_features)
        y2 = columns.pow(2).sum(dim=1).view(self.in_features, 1)
        xy = columns.mm(rows.t()).mul(-2.)

        return x2.add(y2).add(xy).mul(-1).exp()

    def _sample_eps(self,
                    shape: tuple):
        return Variable(self.floatTensor(shape).normal_())

    def _eq_logpw(self,
                  prior_std: float,
                  mean: torch.Tensor,
                  logvar: torch.Tensor) -> torch.Tensor:
        logpw = logvar.exp().add(mean ** 2).div(prior_std ** 2).add(math.log(2.*math.pi*(prior_std ** 2))).mul(-0.5)
        return torch.sum(logpw)

    def _eq_logqw(self,
                  logvar: torch.Tensor):
        logqw = logvar.add(math.log(2.*math.pi)).add(1.).mul(-0.5)
        return torch.sum(logqw)

    def eq_logpw(self) -> torch.Tensor:
        rows = self._eq_logpw(prior_std=self.prior_std, mean=self.rows_mean, logvar=self.rows_logvar)
        columns = self._eq_logpw(prior_std=self.prior_std, mean=self.columns_mean, logvar=self.columns_logvar)
        alpha = self._eq_logpw(prior_std=self.prior_std, mean=self.alpha_mean, logvar=self.alpha_logvar)
        logpw = rows.add(columns).add(alpha)

        if self.use_bias:
            bias = self._eq_logpw(prior_std=self.prior_std, mean=self.bias_mean, logvar=self.bias_logvar)
            logpw.add(bias)
        return logpw

    def eq_logqw(self):
        rows = self._eq_logqw(logvar=self.rows_logvar)
        columns = self._eq_logqw(logvar=self.columns_logvar)
        alpha = self._eq_logqw(logvar=self.alpha_logvar)
        logqw = rows.add(columns).add(alpha)

        if self.use_bias:
            bias = self._eq_logqw(logvar=self.bias_logvar)
            logqw.add(bias)
        return logqw

    def kldiv(self):
        return self.eq_logpw() - self.eq_logqw()

    def kldiv_aux(self) -> float:
        return 0.

    def forward(self,
                input: torch.Tensor):

        rows = self.rows_mean
        columns = self.columns_mean
        alpha = self.alpha_mean

        if self.training:
            rows.add(self.rows_logvar.mul(0.5).exp().mul(self._sample_eps(rows.shape)))
            columns.add(self.columns_logvar.mul(0.5).exp().mul(self._sample_eps(columns.shape)))
            alpha.add(self.alpha_logvar.mul(0.5).exp().mul(self._sample_eps(alpha.shape)))

        w = self._calc_rbf_weights(rows=rows,
                                   columns=columns)

        y = input.mul(alpha).mm(w)

        if self.use_bias:
            y.add(self.bias_mean.view(1, self.out_features))
            if self.training:
                y.add(self.bias_logvar.mul(0.5).exp().mul(self._sample_eps(self.bias_logvar.shape)).view(1, self.out_features))
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', dim: ' \
            + str(self.dim) + ')'


class OrthogonalDense(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 order: int = 8,
                 use_bias: bool = True,
                 add_diagonal: bool = True,
                 **kwargs):
        super(OrthogonalDense, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features
        self.add_diagonal = add_diagonal

        max_feature = max(self.in_features, self.out_features)

        self.order = order or max_feature
        assert 1 <= self.order <= max_feature

        self.v = Parameter(self.floatTensor(self.order, max_feature))

        if self.add_diagonal:
            self.d = Parameter(self.floatTensor(min(self.in_features, self.out_features)))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Parameter(self.floatTensor(out_features))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.v.data.normal_()
        if self.add_diagonal:
            self.d.data.normal_()

        if self.use_bias:
            self.bias.data.normal_(std=1e-2)

    def eq_logpw(self, **kwargs) -> torch.Tensor:
        logpw = 0.
        logpb = 0.
        if self.use_bias:
            logpb = - torch.sum(.5 * self.bias.pow(2))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def _chain_multiply(self,
                        t: torch.Tensor) -> torch.Tensor:
        p = t[0]
        for i in range(1, t.shape[0]):
            p = p.mm(t[i])
        return p

    def _calc_householder_tensor(self,
                                 t: torch.Tensor) -> torch.Tensor:
        norm = t.norm(p=2, dim=1)
        t = t.div(norm.unsqueeze(1))
        h = torch.einsum('ab,ac->abc', (t, t))
        return torch.eye(n=t.shape[1], device=self.device).expand_as(h) - h.mul(2.)

    def _calc_weights(self,
                      v: torch.Tensor,
                      d: torch.Tensor) -> torch.Tensor:

        u = self._chain_multiply(self._calc_householder_tensor(v))

        if self.out_features <= self.in_features:
            D = torch.eye(n=self.in_features, m=self.out_features, device=self.device).mm(torch.diag(d))
            W = u.mm(D)
        else:
            D = torch.diag(d).mm(torch.eye(n=self.in_features, m=self.out_features, device=self.device))
            W = D.mm(u)
        return W

    def forward(self,
                input: torch.Tensor):

        if self.add_diagonal:
            w = self._calc_weights(v=self.v,
                                   d=self.d)
        else:
            d = torch.ones(min(self.in_features, self.out_features), device=self.device)
            w = self._calc_weights(v=self.v,
                                   d=d)

        y = input.mm(w)

        if self.use_bias:
            return y.add(self.bias.view(1, self.out_features))
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', order: ' \
            + str(self.order) + ', add_diagonal: ' \
            + str(self.add_diagonal) + ')'


class OrthogonalBayesianDense(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 order: int = 8,
                 use_bias: bool = True,
                 add_diagonal: bool = True,
                 prior_std: float = 1.,
                 bias_std: float = 1e-2,
                 **kwargs):
        super(OrthogonalBayesianDense, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features
        self.add_diagonal = add_diagonal
        self.prior_std = prior_std
        self.bias_std = bias_std

        max_feature = max(self.in_features, self.out_features)

        self.order = order or max_feature
        assert 1 <= self.order <= max_feature

        self.v_mean = Parameter(self.floatTensor(self.order, max_feature))
        self.v_logvar = Parameter(self.floatTensor(self.order, max_feature))

        if self.add_diagonal:
            self.d_mean = Parameter(self.floatTensor(min(self.in_features, self.out_features)))
            self.d_logvar = Parameter(self.floatTensor(min(self.in_features, self.out_features)))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_mean = Parameter(self.floatTensor(out_features))
            self.bias_logvar = Parameter(self.floatTensor(out_features))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.v_mean.data.normal_(std=self.prior_std)
        self.v_logvar.data.normal_(mean=-5., std=self.prior_std)

        if self.add_diagonal:
            self.d_mean.data.normal_(std=self.prior_std)
            self.d_logvar.data.normal_(mean=-5., std=self.prior_std)

        if self.use_bias:
            self.bias_mean.data.normal_(std=self.bias_std)
            self.bias_logvar.data.normal_(mean=-5., std=self.bias_std)

    def _sample_eps(self,
                    shape: tuple):
        return Variable(self.floatTensor(shape).normal_())

    def _eq_logpw(self,
                  prior_mean: float,
                  prior_std: float,
                  mean: torch.Tensor,
                  logvar: torch.Tensor) -> torch.Tensor:
        logpw = logvar.exp().add((prior_mean - mean) ** 2).div(prior_std ** 2).add(math.log(2.*math.pi*(prior_std ** 2))).mul(-0.5)
        return torch.sum(logpw)

    def _eq_logqw(self,
                  logvar: torch.Tensor):
        logqw = logvar.add(math.log(2.*math.pi)).add(1.).mul(-0.5)
        return torch.sum(logqw)

    def eq_logpw(self) -> torch.Tensor:
        v = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.v_mean, logvar=self.v_logvar)

        if self.add_diagonal:
            d = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.d_mean, logvar=self.d_logvar)
            logpw = v.add(d)
        else:
            logpw = v

        if self.use_bias:
            bias = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.bias_mean, logvar=self.bias_logvar)
            logpw.add(bias)
        return logpw

    def eq_logqw(self):
        v = self._eq_logqw(logvar=self.v_logvar)

        if self.add_diagonal:
            d = self._eq_logqw(logvar=self.d_logvar)
            logqw = v.add(d)
        else:
            logqw = v

        if self.use_bias:
            bias = self._eq_logqw(logvar=self.bias_logvar)
            logqw.add(bias)
        return logqw

    def kldiv(self):
        return self.eq_logpw() - self.eq_logqw()

    def kldiv_aux(self) -> float:
        return 0.

    def _chain_multiply(self,
                        t: torch.Tensor) -> torch.Tensor:
        p = t[0]
        for i in range(1, t.shape[0]):
            p = p.mm(t[i])
        return p

    def _calc_householder_tensor(self,
                                 t: torch.Tensor) -> torch.Tensor:
        norm = t.norm(p=2, dim=1)
        t = t.div(norm.unsqueeze(1))
        h = torch.einsum('ab,ac->abc', (t, t))
        return torch.eye(n=t.shape[1], device=self.device).expand_as(h) - h.mul(2.)

    def _calc_weights(self,
                      v: torch.Tensor,
                      d: torch.Tensor) -> torch.Tensor:

        u = self._chain_multiply(self._calc_householder_tensor(v))

        if self.out_features <= self.in_features:
            D = torch.eye(n=self.in_features, m=self.out_features, device=self.device).mm(torch.diag(d))
            W = u.mm(D)
        else:
            D = torch.diag(d).mm(torch.eye(n=self.in_features, m=self.out_features, device=self.device))
            W = D.mm(u)
        return W

    def forward(self,
                input: torch.Tensor):

        v = self.v_mean.add(self.v_logvar.mul(0.5).exp().mul(self._sample_eps(self.v_logvar.shape)))

        if self.add_diagonal:
            d = self.d_mean.add(self.d_logvar.mul(0.5).exp().mul(self._sample_eps(self.d_logvar.shape)))
        else:
            d = torch.ones(min(self.in_features, self.out_features), device=self.device)

        w = self._calc_weights(v=v,
                               d=d)
        y = input.mm(w)

        if self.use_bias:
            bias = self.bias_mean.add(self.bias_logvar.mul(0.5).exp().mul(self._sample_eps(self.bias_logvar.shape)))
            return y.add(bias.view(1, self.out_features))
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ', order: ' \
            + str(self.order) + ', add_diagonal: ' \
            + str(self.add_diagonal) + ')'
