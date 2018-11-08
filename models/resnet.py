import torch
import torch.nn as nn
from layers import FFGaussConv2d, HSConv2d, DropoutConv2d, MAPConv2d, FFGaussDense, HSDense, DropoutDense, MAPDense, KernelDense, KernelConv2, KernelDenseBayesian, KernelBayesianConv2, OrthogonalDense, OrthogonalConv2d, OrthogonalBayesianDense, OrthogonalBayesianConv2d
from copy import deepcopy


def conv3x3(conv_layer, in_planes, out_planes, stride=1, mask=None):
    return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False, droprate=0.5, prior_std_z=1., mask=mask)


def conv1x1(conv_layer, in_planes, out_planes, stride=1, mask=None):
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, droprate=0.5, prior_std_z=1., mask=mask)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_layer, inplanes, planes, stride=1, downsample=None, mask=None):
        super(BasicBlock, self).__init__()
        if mask is not None:
            self.conv1 = conv3x3(conv_layer, inplanes, planes, stride, mask[0])
        else:
            self.conv1 = conv3x3(conv_layer, inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if mask is not None:
            self.conv2 = conv3x3(conv_layer, planes, planes, mask[1])
        else:
            self.conv2 = conv3x3(conv_layer, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, conv_layer, inplanes, planes, stride=1, downsample=None, mask=None):
        super(Bottleneck, self).__init__()
        if mask is not None:
            self.conv1 = conv1x1(conv_layer, inplanes, planes, mask[0])
        else:
            self.conv1 = conv1x1(conv_layer, inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if mask is not None:
            self.conv2 = conv3x3(conv_layer, planes, planes, stride, mask[1])
        else:
            self.conv2 = conv3x3(conv_layer, planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        if mask is not None:
            self.conv3 = conv1x1(conv_layer, planes, planes * self.expansion, mask[2])
        else:
            self.conv3 = conv1x1(conv_layer, planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, type_net='map', mask=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

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

        self.N = 50000
        self.beta_ema = 0

        self.layers = []

        if mask is not None:
            self.conv1 = self.conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False, droprate=0.5,
                                         prior_std_z=1., mask=mask[0])
        else:
            self.conv1 = self.conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False, droprate=0.5, prior_std_z=1., mask=None)

        self.layers.append(self.conv1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if mask is not None:
            self.layer1, l1 = self._make_layer(block, 64, layers[0], mask=mask[1])
            self.layer2, l2 = self._make_layer(block, 128, layers[1], stride=2, mask=mask[2])
            self.layer3, l3 = self._make_layer(block, 256, layers[2], stride=2, mask=mask[3])
            self.layer4, l4 = self._make_layer(block, 512, layers[3], stride=2, mask=mask[4])
        else:
            self.layer1, l1 = self._make_layer(block, 64, layers[0])
            self.layer2, l2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3, l3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4, l4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layers += l1
        self.layers += l2
        self.layers += l3
        self.layers += l4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self.fc_layer(512 * block.expansion, num_classes)

        self.layers.append(self.fc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, mask=None):
        layers_ = []

        downsample = None
        mask0 = 0
        if stride != 1 or self.inplanes != planes * block.expansion:
            if mask is not None:
                downsample = nn.Sequential(
                    conv1x1(self.conv_layer, self.inplanes, planes * block.expansion, stride, mask=mask[0]),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                mask0 = 1
            else:
                downsample = nn.Sequential(
                    conv1x1(self.conv_layer, self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers_.append(downsample[0])

        layers = []
        if mask is not None:
            layers.append(block(self.conv_layer, self.inplanes, planes, stride, downsample, mask=mask[mask0]))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, mask=mask[mask0+i]))
        else:
            layers.append(block(self.conv_layer, self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.conv_layer, self.inplanes, planes))

        for layer in layers:
            if isinstance(layer, BasicBlock):
                layers_.append(layer.conv1)
                layers_.append(layer.conv2)
            elif isinstance(layer, Bottleneck):
                layers_.append(layer.conv1)
                layers_.append(layer.conv2)
                layers_.append(layer.conv3)

        return nn.Sequential(*layers), layers_

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
