import torch
import torch.nn as nn
from schedules import kl_linear


class CrossEntropyLossWithAnnealing:
    def __init__(self,
                 iter_per_epoch: int,
                 total_steps: int,
                 anneal_type: str,
                 anneal_kl: bool = False,
                 epzero: int = 0,
                 epmax: int = 100,
                 anneal_maxval: float = 1.,
                 writer: object = None):
        self._loglikelihood = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._loglikelihood = self._loglikelihood.cuda()

        self._iter_per_epoch = iter_per_epoch
        self._total_steps = total_steps
        self._epzero = epzero
        self._epmax = epmax
        self._anneal_kl = anneal_kl
        self._anneal_maxval = anneal_maxval
        self._anneal_type = anneal_type
        self._writer = writer

    def __call__(self, output, target_var, model):
        loss = self._loglikelihood(output, target_var)
        annealing = 1.
        if self._anneal_kl:
            annealing = kl_linear(self._epzero,
                                  self._epmax,
                                  self._iter_per_epoch,
                                  self._total_steps,
                                  maxval=self._anneal_maxval)
            if annealing > self._anneal_maxval or annealing < 0.:
                raise Exception()
            if self._writer is not None:
                self._writer.add_scalar('annealing', annealing, self._total_steps)

        total_loss = loss + model.kl_div(annealing=annealing,
                                         type_anneal=self._anneal_type)
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss


class CrossEntropyLossWithMMD:
    def __init__(self,
                 scale_factor: float = 1.,
                 num_samples: int = 2):
        self._loglikelihood = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self._loglikelihood = self._loglikelihood.cuda()

        self._scale_factor = scale_factor
        self._num_samples = num_samples

    def __call__(self, output, target_var, model):
        loss = self._loglikelihood(output, target_var)

        total_loss = loss + model.mmd(scale_factor=self._scale_factor,
                                      num_samples=self._num_samples)
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss
