import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd as ConvNd
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init


class HSConv2d(Module):
    '''Input channel noise'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 prior_std=1., prior_std_z=1., dof=1., **kwargs):
        super(HSConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_std = prior_std
        self.prior_std_z = prior_std_z
        self.use_bias = False
        self.dof = dof
        self.mean_w = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.logvar_w = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.qz_mean = Parameter(torch.Tensor(in_channels // groups))
        self.qz_logvar = Parameter(torch.Tensor(in_channels // groups))
        self.dim_z = in_channels // groups

        if bias:
            self.mean_bias = Parameter(torch.Tensor(out_channels))
            self.logvar_bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.mean_w, mode='fan_in')
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
        logpw = - .5 * math.log(2 * math.pi * self.prior_std**2) - .5 * self.logvar_w.exp().div(self.prior_std**2)
        logpw -= .5 * self.mean_w.pow(2).div(self.prior_std**2)
        logpb = 0.
        if self.use_bias:
            logpb = - .5 * math.log(2 * math.pi * self.prior_std**2) - .5 * self.logvar_bias.exp().div(self.prior_std**2)
            logpb -= .5 * self.mean_bias.pow(2).div(self.prior_std**2)
            logpb = torch.sum(logpb)
        return torch.sum(logpw) + logpb

    def eq_logqw(self):
        logqw = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_w + 1))
        logqb = 0.
        if self.use_bias:
            logqb = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_bias + 1))
        return logqw + logqb

    def kldiv_aux(self):
        z = self.sample_z(1)
        z = z.view(self.dim_z)

        logqm = - torch.sum(.5 * (math.log(2 * math.pi) + self.qz_logvar + 1))
        logqm = logqm.add(- torch.sum(F.sigmoid(z.exp().add(- 1).log()).log()))

        logpm = torch.sum(2 * math.lgamma(.5 * (self.dof + 1)) - math.lgamma(.5 * self.dof) - math.log(self.prior_std_z) -
                          .5 * math.log(self.dof * math.pi) -
                          .5 * (self.dof + 1) * torch.log(1. + z.pow(2)/(self.dof * self.prior_std_z**2)))

        return logpm - logqm

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def get_eps(self, size):
        eps = self.floatTensor(size).normal_()
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size):
        z = self.qz_mean.view(1, self.dim_z).expand(batch_size, self.dim_z)
        if self.training:
            eps = self.get_eps(self.floatTensor(batch_size, self.dim_z))
            z = z.add(eps.mul(self.qz_logvar.view(1, self.dim_z).expand(batch_size, self.dim_z).mul(0.5).exp_()))
        z = z.contiguous().view(batch_size, self.dim_z, 1, 1)
        return F.softplus(z)

    def sample_W(self):
        W = self.mean_w
        if self.training:
            eps = self.get_eps(self.mean_w.size())
            W = W.add(eps.mul(self.logvar_w.mul(0.5).exp_()))
        return W

    def sample_b(self):
        if not self.use_bias:
            return None
        b = self.mean_bias
        if self.training:
            eps = self.get_eps(self.mean_bias.size())
            b = b.add(eps.mul(self.logvar_bias.mul(0.5).exp_()))
        return b

    def forward(self, input_):
        z = self.sample_z(input_.size(0))
        W = self.sample_W()
        b = self.sample_b()
        return F.conv2d(input_.mul(z.expand_as(input_)), W, b, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, prior_std_z={prior_std_z}, dof={dof}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class FFGaussConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 prior_std=1, mask=None, **kwargs):
        super(FFGaussConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_std = prior_std
        self.use_bias = False
        self.mean_w = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.logvar_w = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.mean_bias = Parameter(torch.Tensor(out_channels))
            self.logvar_bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

        if mask is not None:
            self.weight_mask = self.floatTensor(torch.from_numpy(mask[0]))
            if bias:
                self.bias_mask = self.floatTensor(torch.from_numpy(mask[1]))
        else:
            self.weight_mask = None
            self.bias_mask = None

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.mean_w, mode='fan_in')
        self.logvar_w.data.normal_(-9., 1e-4)
        if self.use_bias:
            self.mean_bias.data.zero_()
            self.logvar_bias.data.normal_(-9., 1e-4)

    def constrain_parameters(self, thres_std=1.):
        self.logvar_w.data.clamp_(max=2. * math.log(thres_std))
        if self.use_bias:
            self.logvar_bias.data.clamp_(max=2. * math.log(thres_std))

    def eq_logpw(self):
        if self.weight_mask is not None:
            logpw = - .5 * math.log(2 * math.pi * self.prior_std**2) - .5 * self.logvar_w.exp().mul(self.weight_mask).div(self.prior_std**2)
            logpw -= .5 * self.mean_w.mul(self.weight_mask).pow(2).div(self.prior_std**2)
        else:
            logpw = - .5 * math.log(2 * math.pi * self.prior_std ** 2) - .5 * self.logvar_w.exp().mul(self.weight_mask).div(self.prior_std ** 2)
            logpw -= .5 * self.mean_w.mul(self.weight_mask).pow(2).div(self.prior_std ** 2)
        logpb = 0.
        if self.use_bias:
            if self.bias_mask is not None:
                logpb = - .5 * math.log(2 * math.pi * self.prior_std**2) - .5 * self.logvar_bias.exp().mul(self.bias_mask).div(self.prior_std**2)
                logpb -= .5 * self.mean_bias.mul(self.bias_mask).pow(2).div(self.prior_std**2)
            else:
                logpb = - .5 * math.log(2 * math.pi * self.prior_std**2) - .5 * self.logvar_bias.exp().div(self.prior_std**2)
                logpb -= .5 * self.mean_bias.pow(2).div(self.prior_std**2)
        return torch.sum(logpw) + torch.sum(logpb)

    def eq_logqw(self):
        if self.weight_mask is not None:
            logqw = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_w.mul(self.weight_mask) + 1))
        else:
            logqw = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_w + 1))
        logqb = 0.
        if self.use_bias:
            if self.bias_mask is not None:
                logqb = - torch.sum(.5 * (math.log(2 * math.pi) + self.logvar_bias.mul(self.bias_mask) + 1))
            else:
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

    def sample_W(self):
        W = self.mean_w
        if self.training:
            eps = self.get_eps(self.mean_w.size())
            W = W.add(eps.mul(self.logvar_w.mul(0.5).exp_()))
        if self.weight_mask is not None:
            W = W.mul(self.weight_mask)
        return W

    def sample_b(self):
        if not self.use_bias:
            return None
        b = self.mean_bias
        if self.training:
            eps = self.get_eps(self.mean_bias.size())
            b = b.add(eps.mul(self.logvar_bias.mul(0.5).exp_()))
        if self.bias_mask is not None:
            b = b.mul(self.bias_mask)
        return b

    def forward(self, input_):
        W = self.sample_W()
        b = self.sample_b()

        return F.conv2d(input_, W, b, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, prior_std={prior_std}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class DropoutConv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 droprate=0.5, weight_decay=1., share_mask=False, **kwargs):
        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        super(DropoutConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                            pair(0), groups, bias)
        self.droprate = droprate
        self.dim_z = self.weight.size(0)
        self.weight_decay = weight_decay
        self.share_mask = share_mask
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_in')
        if self.bias is not None:
            self.bias.data.zero_()

    def constrain_parameters(self, thres_std=1.):
        pass

    def eq_logpw(self, **kwargs):
        logpw = - (1 - self.droprate) * self.weight_decay * torch.sum(.5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            logpb = - (1 - self.droprate) * self.weight_decay * torch.sum(.5 * (self.bias.pow(2)))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def get_eps(self, size):
        eps = self.floatTensor(size).fill_(1. - self.droprate)
        if self.droprate > 0 and self.training:
            eps = torch.bernoulli(eps)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size):
        if self.share_mask:
            z = self.get_eps(self.floatTensor(batch_size, self.dim_z))
            z = z.contiguous().view(batch_size, self.dim_z, 1, 1)
        else:
            z = self.get_eps(batch_size)
        return z

    def forward(self, input_):
        output = F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.droprate > 0:
            if self.share_mask:
                z = self.sample_z(output.size(0))
                return output.mul(z.expand_as(output))
            z = self.sample_z(output.size())
            return output.mul(z)

        return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, share_mask={share_mask}'
             ', stride={stride}, droprate={droprate}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MAPConv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_decay=1., mask=None, **kwargs):
        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)
        self.weight_decay = weight_decay
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        super(MAPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                        pair(0), groups, bias)

        if mask is not None:
            if torch.cuda.is_available():
                self.weight_mask = self.floatTensor(torch.from_numpy(mask[0]).cuda())
                if bias:
                    self.bias_mask = self.floatTensor(torch.from_numpy(mask[1]).cuda())
            else:
                self.weight_mask = self.floatTensor(torch.from_numpy(mask[0]))
                if bias:
                    self.bias_mask = self.floatTensor(torch.from_numpy(mask[1]))
        else:
            self.weight_mask = None
            self.bias_mask = None

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_in')
        if self.bias is not None:
            self.bias.data.zero_()

    def constrain_parameters(self, thres_std=1.):
        pass

    def eq_logpw(self, **kwargs):
        if self.weight_mask is not None:
            logpw = - torch.sum(self.weight_decay * .5 * (self.weight.mul(self.weight_mask).pow(2)))
        else:
            logpw = - torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.bias is not None:
            if self.bias_mask is not None:
                logpb = - torch.sum(self.weight_decay * .5 * (self.bias.mul(self.bias_mask).pow(2)))
            else:
                logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def forward(self, input_):
        if self.weight_mask is not None:
            output = F.conv2d(input_, self.weight.mul(self.weight_mask), self.bias.mul(self.bias_mask), self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input_, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class KernelConv2(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dim: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 use_bias: bool = True,
                 weight_decay: float = 1.,
                 **kwargs):
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dim = dim
        self.kernel_size = pair(kernel_size)
        self.use_bias = use_bias
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

        super(KernelConv2, self).__init__(in_channels, out_channels, self.kernel_size, stride, padding, dilation, False,
                                          pair(0), groups, use_bias)

        self.columns = Parameter(self.floatTensor(self.in_channels * int(np.prod(self.kernel_size)), self.dim))
        self.rows = Parameter(self.floatTensor(self.out_channels * int(np.prod(self.kernel_size)) // groups, self.dim))
        self.alpha = Parameter(self.floatTensor(self.out_channels // self.groups, self.in_channels))

        self.weight_decay = weight_decay

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_in')

        if hasattr(self, 'rows'):
            self.rows.data.normal_()
        if hasattr(self, 'columns'):
            self.columns.data.normal_()
        if hasattr(self, 'alpha'):
            self.alpha.data.normal_()
        if self.use_bias:
            self.bias.data.normal_(std=1e-5)

    def _calc_rbf_weights(self,
                          rows: torch.Tensor,
                          columns: torch.Tensor,
                          alpha: torch.Tensor) -> Parameter:
        w = self.floatTensor(self.out_channels // self.groups, self.in_channels, np.prod(self.kernel_size))

        for i in range(int(np.prod(self.kernel_size))):
            row_start = i*(self.out_channels // self.groups)
            row_stop = (i+1)*(self.out_channels // self.groups)
            col_start = i*self.in_channels
            col_stop = (i+1)*self.in_channels

            x2 = rows[row_start:row_stop, :].pow(2).sum(dim=1).view(self.out_channels // self.groups, 1)
            y2 = columns[col_start:col_stop, :].pow(2).sum(dim=1).view(1, self.in_channels)
            xy = rows[row_start:row_stop, :].mm(columns[col_start:col_stop, :].t()).mul(-2.)

            w[:, :, i] = x2.add(y2).add(xy).mul(-1).exp().mul(alpha)

        return w.view(self.out_channels // self.groups, self.in_channels, *self.kernel_size)

    def eq_logpw(self, **kwargs):
        logpw = - torch.sum(self.weight_decay * .5 * (self.weight.pow(2)))
        logpb = 0
        if self.use_bias:
            logpb = - torch.sum(self.weight_decay * .5 * (self.bias.pow(2)))
        return logpw + logpb

    def eq_logqw(self):
        return 0.

    def kldiv_aux(self):
        return 0.

    def kldiv(self):
        return self.kldiv_aux() + self.eq_logpw() - self.eq_logqw()

    def forward(self,
                input: torch.Tensor):
        weight = self._calc_rbf_weights(rows=self.rows,
                                        columns=self.columns,
                                        alpha=self.alpha)
        y = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}, dim={dim}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class KernelBayesianConv2(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dim: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 use_bias: bool = True,
                 prior_std: float = 1.,
                 bias_std: float = 1e-3,
                 **kwargs):

        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)

        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dim = dim
        self.kernel_size = pair(kernel_size)
        self.use_bias = use_bias
        self.prior_std = prior_std
        self.bias_std = bias_std

        super(KernelBayesianConv2, self).__init__(in_channels, out_channels, self.kernel_size, stride, padding, dilation, False,
                                                  pair(0), groups, use_bias)

        self.columns_mean = Parameter(self.floatTensor(self.in_channels * int(np.prod(self.kernel_size)), self.dim))
        self.columns_logvar = Parameter(self.floatTensor(self.in_channels * int(np.prod(self.kernel_size)), self.dim))

        self.rows_mean = Parameter(self.floatTensor(self.out_channels * int(np.prod(self.kernel_size)) // groups, self.dim))
        self.rows_logvar = Parameter(self.floatTensor(self.out_channels * int(np.prod(self.kernel_size)) // groups, self.dim))

        self.alpha_mean = Parameter(self.floatTensor(self.out_channels // groups, self.in_channels))
        self.alpha_logvar = Parameter(self.floatTensor(self.out_channels // groups, self.in_channels))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_mean = Parameter(self.floatTensor(self.out_channels // self.groups))
            self.bias_logvar = Parameter(self.floatTensor(self.out_channels // self.groups))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_in')

        if hasattr(self, 'columns_mean'):
            self.columns_mean.data.normal_(std=self.prior_std)
            self.columns_logvar.data.normal_(std=self.prior_std)

        if hasattr(self, 'rows_mean'):
            self.rows_mean.data.normal_(std=self.prior_std)
            self.rows_logvar.data.normal_(std=self.prior_std)

        if hasattr(self, 'alpha_mean'):
            self.alpha_mean.data.normal_(std=self.prior_std)
            self.alpha_logvar.data.normal_(std=self.prior_std)

        if hasattr(self, 'bias_mean'):
            self.bias_mean.data.normal_(std=self.bias_std)
            self.bias_logvar.data.normal_(std=self.bias_std)

    def _calc_rbf_weights(self,
                          rows: torch.Tensor,
                          columns: torch.Tensor,
                          alpha: torch.Tensor) -> Parameter:
        w = self.floatTensor(self.out_channels // self.groups, self.in_channels, np.prod(self.kernel_size))

        for i in range(int(np.prod(self.kernel_size))):
            row_start = i*(self.out_channels // self.groups)
            row_stop = (i+1)*(self.out_channels // self.groups)
            col_start = i*self.in_channels
            col_stop = (i+1)*self.in_channels

            x2 = rows[row_start:row_stop, :].pow(2).sum(dim=1).view(self.out_channels // self.groups, 1)
            y2 = columns[col_start:col_stop, :].pow(2).sum(dim=1).view(1, self.in_channels)
            xy = rows[row_start:row_stop, :].mm(columns[col_start:col_stop, :].t()).mul(-2.)

            w[:, :, i] = x2.add(y2).add(xy).mul(-1).exp().mul(alpha)

        return w.view(self.out_channels // self.groups, self.in_channels, *self.kernel_size)

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
            rows = self.rows_mean.add(self.rows_logvar.mul(0.5).exp().mul(self._sample_eps(rows.shape)))
            columns = self.columns_mean.add(self.columns_logvar.mul(0.5).exp().mul(self._sample_eps(columns.shape)))
            alpha = self.alpha_mean.add(self.alpha_logvar.mul(0.5).exp().mul(self._sample_eps(alpha.shape)))

        weight = self._calc_rbf_weights(rows=rows,
                                        columns=columns,
                                        alpha=alpha)

        bias = self.bias_mean
        if self.training:
            bias = self.bias_mean.add(self.bias_logvar.mul(0.5).exp().mul(self._sample_eps(self.bias_logvar.shape)))

        if not self.use_bias:
            bias = None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, dim={dim}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class OrthogonalConv2d(ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 simple=True,
                 add_diagonal=True,
                 weight_decay=1., **kwargs):
        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)
        self.weight_decay = weight_decay
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.simple = simple
        self.add_diagonal = add_diagonal
        super(OrthogonalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                               pair(0), groups, use_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0]

        if simple:
            self.r = Parameter(self.floatTensor(self.kernel_size*self.kernel_size, self.out_channels))
        else:
            self.r = Parameter(self.floatTensor(2, self.out_channels))
            self.t = Parameter(self.floatTensor(2 * (self.kernel_size - 1), self.out_channels))

        if self.add_diagonal:
            self.d = Parameter(self.floatTensor(self.kernel_size, self.kernel_size, min(self.in_channels, self.out_channels)))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.normal_(std=1e-2)

        if hasattr(self, 'r'):
            torch.nn.init.orthogonal_(self.r)
        if hasattr(self, 't'):
            torch.nn.init.orthogonal_(self.t)

        if hasattr(self, 'd'):
            self.d.data.normal_()

    def constrain_parameters(self, thres_std=1.):
        pass

    def eq_logpw(self, **kwargs):
        logpw = 0.
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

    def _calc_block_orthogonal_tensor(self,
                                      size: int,
                                      t: torch.Tensor):
        h = torch.zeros(size, 2, 2, t.shape[1], t.shape[2], device=self.device)

        for i in range(size):
            p = t[2 * i]
            q = t[2 * i + 1]
            pq = p.mm(q)
            h[i, 0, 0] = pq
            h[i, 0, 1] = p.sub(pq)
            h[i, 1, 0] = q.sub(pq)
            h[i, 1, 1] = torch.eye(p.shape[0], device=self.device).add(pq).sub(p).sub(q)
        return h

    def _matrix_conv(self,
                     s: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        assert s[0, 0].shape[0] == t[0, 0].shape[0]

        n = s[0, 0].shape[0]
        k = int(np.sqrt(s.shape[0] * s.shape[1]))
        l = int(np.sqrt(t.shape[0] * t.shape[1]))
        size = k + l - 1

        result = torch.zeros(size, size, n, n, device=self.device)

        for i in range(size):
            for j in range(size):
                for index1 in range(min(k, i + 1)):
                    for index2 in range(min(k, j + 1)):
                        if (i - index1) < l and (j - index2) < l:
                            result[i, j] += torch.mm(s[index1, index2],
                                                     t[i - index1, j - index2])
        return result

    def _orthogonal_kernel(self,
                           r: torch.Tensor,
                           t: torch.Tensor,
                           d: torch.Tensor = None,
                           transpose: bool = False) -> torch.Tensor:
        assert self.in_channels <= self.out_channels

        if self.in_channels <= self.out_channels:
            d = torch.einsum('abc,cd->abcd',
                             (d, torch.eye(n=self.in_channels, m=self.out_channels, device=self.device)))
        else:
            d = torch.einsum('ab, cda->cdab',
                             (torch.eye(n=self.in_channels, m=self.out_channels, device=self.device), d))

        if self.simple:
            q = self._calc_householder_tensor(r).view(self.kernel_size,
                                                      self.kernel_size,
                                                      self.out_channels,
                                                      self.out_channels)
        else:
            r = self._chain_multiply(self._calc_householder_tensor(r))

            if self.kernel_size == 1:
                return torch.unsqueeze(torch.unsqueeze(r, 0), 0)

            t = self._calc_block_orthogonal_tensor(size=self.kernel_size - 1,
                                                   t=self._calc_householder_tensor(t))

            s = t[0]
            for i in range(1, self.kernel_size - 1):
                s = self._matrix_conv(s=s,
                                      t=t[i])

            q = torch.einsum('ab,debc->deac', (r, s))

        q = torch.einsum('abcd,abde->abce', (d, q))

        if transpose:
            q = q.permute(2, 3, 1, 0)
        else:
            q = q.permute(3, 2, 1, 0)

        return q

    def forward(self, input_):
        if self.add_diagonal:
            d = self.d
        else:
            d = torch.ones(self.kernel_size, self.kernel_size, min(self.in_channels, self.out_channels),
                           device=self.device)

        if self.simple:
            weight = self._orthogonal_kernel(r=self.r,
                                             t=None,
                                             d=d)
        else:
            weight = self._orthogonal_kernel(r=self.r,
                                             t=self.t,
                                             d=d)
        return F.conv2d(input_, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', simple={simple}'
        s += ', add_diagonal={add_diagonal}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class OrthogonalBayesianConv2d(ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 simple=True,
                 add_diagonal=True,
                 weight_decay=1.,
                 prior_std = 1.,
                 bias_std = 1e-2,
                 **kwargs):
        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)
        self.weight_decay = weight_decay
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.simple = simple
        self.add_diagonal = add_diagonal
        super(OrthogonalBayesianConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                       False, pair(0), groups, use_bias)

        self.in_channels = in_channels
        self.out_channels = out_channels // groups
        self.kernel_size = kernel_size[0]

        self.prior_std = prior_std
        self.bias_std = bias_std

        if simple:
            self.r_mean = Parameter(self.floatTensor(self.kernel_size * self.kernel_size, self.out_channels))
            self.r_logvar = Parameter(self.floatTensor(self.kernel_size * self.kernel_size, self.out_channels))
        else:
            self.r_mean = Parameter(self.floatTensor(2, self.out_channels))
            self.r_logvar = Parameter(self.floatTensor(2, self.out_channels))
            self.t_mean = Parameter(self.floatTensor(2 * (self.kernel_size - 1), self.out_channels))
            self.t_logvar = Parameter(self.floatTensor(2 * (self.kernel_size - 1), self.out_channels))

        if self.add_diagonal:
            self.d_mean = Parameter(self.floatTensor(self.kernel_size, self.kernel_size, min(self.in_channels, self.out_channels)))
            self.d_logvar = Parameter(self.floatTensor(self.kernel_size, self.kernel_size, min(self.in_channels, self.out_channels)))

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_mean = Parameter(self.floatTensor(self.out_channels // self.groups))
            self.bias_logvar = Parameter(self.floatTensor(self.out_channels // self.groups))

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        if hasattr(self, 'r_mean'):
            torch.nn.init.orthogonal_(self.r_mean)
            self.r_logvar.data.normal_(mean=-5., std=self.prior_std)

        if hasattr(self, 't_mean'):
            torch.nn.init.orthogonal_(self.t_mean)
            self.t_logvar.data.normal_(mean=-5., std=self.prior_std)

        if hasattr(self, 'd_mean'):
            self.d_mean.data.normal_(std=self.prior_std)
            self.d_logvar.data.normal_(mean=-5, std=self.prior_std)

        if hasattr(self, 'bias_mean'):
            self.bias_mean.data.normal_(std=self.bias_std)
            self.bias_logvar.data.normal_(mean=-5., std=self.bias_std)

    def constrain_parameters(self, thres_std=1.):
        pass

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
        r = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.r_mean, logvar=self.r_logvar)

        if self.add_diagonal:
            d = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.d_mean, logvar=self.d_logvar)
            logpw = r.add(d)
        else:
            logpw = r

        if not self.simple:
            t = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.t_mean, logvar=self.t_logvar)
            logpw = logpw.add(t)

        if self.use_bias:
            bias = self._eq_logpw(prior_mean=0., prior_std=self.prior_std, mean=self.bias_mean, logvar=self.bias_logvar)
            logpw.add(bias)

        return logpw

    def eq_logqw(self):
        r = self._eq_logqw(logvar=self.r_logvar)

        if self.add_diagonal:
            d = self._eq_logqw(logvar=self.d_logvar)
            logqw = r.add(d)
        else:
            logqw = r

        if not self.simple:
            t = self._eq_logqw(logvar=self.t_logvar)
            logqw.add(t)

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

    def _calc_block_orthogonal_tensor(self,
                                      size: int,
                                      t: torch.Tensor):
        h = torch.zeros(size, 2, 2, t.shape[1], t.shape[2], device=self.device)

        for i in range(size):
            p = t[2 * i]
            q = t[2 * i + 1]
            pq = p.mm(q)
            h[i, 0, 0] = pq
            h[i, 0, 1] = p.sub(pq)
            h[i, 1, 0] = q.sub(pq)
            h[i, 1, 1] = torch.eye(p.shape[0], device=self.device).add(pq).sub(p).sub(q)
        return h

    def _matrix_conv(self,
                     s: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        assert s[0, 0].shape[0] == t[0, 0].shape[0]

        n = s[0, 0].shape[0]
        k = int(np.sqrt(s.shape[0] * s.shape[1]))
        l = int(np.sqrt(t.shape[0] * t.shape[1]))
        size = k + l - 1

        result = torch.zeros(size, size, n, n, device=self.device)

        for i in range(size):
            for j in range(size):
                for index1 in range(min(k, i + 1)):
                    for index2 in range(min(k, j + 1)):
                        if (i - index1) < l and (j - index2) < l:
                            result[i, j] += torch.mm(s[index1, index2],
                                                     t[i - index1, j - index2])
        return result

    def _orthogonal_kernel(self,
                           r: torch.Tensor,
                           t: torch.Tensor,
                           d: torch.Tensor = None,
                           transpose: bool = False) -> torch.Tensor:
        assert self.in_channels <= self.out_channels

        if self.in_channels <= self.out_channels:
            d = torch.einsum('abc,cd->abcd',
                             (d, torch.eye(n=self.in_channels, m=self.out_channels, device=self.device)))
        else:
            d = torch.einsum('ab, cda->cdab',
                             (torch.eye(n=self.in_channels, m=self.out_channels, device=self.device), d))

        if self.simple:
            q = self._calc_householder_tensor(r).view(self.kernel_size,
                                                      self.kernel_size,
                                                      self.out_channels,
                                                      self.out_channels)
        else:
            r = self._chain_multiply(self._calc_householder_tensor(r))

            if self.kernel_size == 1:
                return torch.unsqueeze(torch.unsqueeze(r, 0), 0)

            t = self._calc_block_orthogonal_tensor(size=self.kernel_size - 1,
                                                   t=self._calc_householder_tensor(t))

            s = t[0]
            for i in range(1, self.kernel_size - 1):
                s = self._matrix_conv(s=s,
                                      t=t[i])

            q = torch.einsum('ab,debc->deac', (r, s))

        q = torch.einsum('abcd,abde->abce', (d, q))

        if transpose:
            q = q.permute(2, 3, 1, 0)
        else:
            q = q.permute(3, 2, 1, 0)

        return q

    def forward(self, input_):
        r = self.r_mean.add(self.r_logvar.mul(0.5).exp().mul(self._sample_eps(self.r_logvar.shape)))

        if self.add_diagonal:
            d = self.d_mean.add(self.d_logvar.mul(0.5).exp().mul(self._sample_eps(self.d_logvar.shape)))
        else:
            d = torch.ones(self.kernel_size, self.kernel_size, min(self.in_channels, self.out_channels),
                           device=self.device)

        if self.simple:
            weight = self._orthogonal_kernel(r=r, t=None, d=d)
        else:
            t = self.t_mean.add(self.t_logvar.mul(0.5).exp().mul(self._sample_eps(self.t_logvar.shape)))
            weight = self._orthogonal_kernel(r=r, t=t, d=d)

        bias = self.bias_mean.add(self.bias_logvar.mul(0.5).exp().mul(self._sample_eps(self.bias_logvar.shape)))
        if not self.use_bias:
            bias = None

        return F.conv2d(input_, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', simple={simple}'
        s += ', add_diagonal={add_diagonal}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
