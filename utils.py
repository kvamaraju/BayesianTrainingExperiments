import torch
import numpy as np
import os

_LAYER_UIDS = {}

prng = np.random.RandomState(1)
torch.manual_seed(1)

try:
    DATA_DIR = os.environ['DATA_DIR']
except:
    DATA_DIR = 'data/'

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def change_random_seed(seed):
    global prng
    prng = np.random.RandomState(seed)
    torch.manual_seed(seed)


def to_one_hot(x, n_cats=10):
    y = np.zeros((x.shape[0], n_cats))
    y[np.arange(x.shape[0]), x] = 1
    return y.astype(np.float32)


def ecdf(x, max_entropy=-10*0.1*np.log(0.1)):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    xs, ys = np.append(xs, np.array([max_entropy])), np.append(ys, np.array([1]))
    return xs, ys


def entropy(p, epsilon=1e-8, per_x=False):
    p = np.clip(p, epsilon, 1. - epsilon)
    entr = -np.sum(p * np.log(p), axis=1)
    if per_x:
        return entr
    return np.mean(entr)


def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AnnealScheduler(object):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.reset()

    def reset(self):
        self.lamba = 0
        self.steps = 0

    def update(self, val):
        self.steps += 1
        self.lamba = self.lamba * self.beta + (1 - self.beta) * val

    def get_lamba(self):
        return self.lamba / (1 - self.beta**self.steps)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
