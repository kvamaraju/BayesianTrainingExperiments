import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# used for logging to TensorBoard
from tensorboardX import SummaryWriter


from models import MLP
from schedules import kl_linear
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch MLP Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='MLP', type=str,
                    help='name of experiment')
# parser.add_argument('--tensorboard',
#                     help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--thres_std', nargs='*', type=float, default=[0.2, 0.5, 1.0])
parser.add_argument('--clip_var', action='store_true')
parser.add_argument('--type_net', default='hs',
                    help='Layer types for the network')
parser.add_argument('--anneal_kl', action='store_true')
parser.add_argument('--epzero', type=int, default=0)
parser.add_argument('--epmax', type=int, default=100)
parser.add_argument('--anneal_maxval', type=float, default=1.)
parser.add_argument('--anneal_type', choices=['kl', 'entr', 'weight', 'bits'], default='bits')
parser.add_argument('--anneal_schedule', choices=['linear'], default='linear')
parser.add_argument('--dof', type=float, default=1.)
parser.add_argument('--beta_ema', type=float, default=0.)
parser.add_argument('--ldims', type=int, nargs='*', default=[1024, 1024])
parser.add_argument('--ep_anneal', type=int, default=10)
parser.add_argument('--lr_decay_ratio', type=float, default=0.2)
parser.set_defaults(tensorboard=True)

best_prec1, total_steps, writer = 0, 0, None


def main():
    global args, best_prec1, total_steps, writer
    args = parser.parse_args()
    args.name += '_{}'.format(args.type_net)
    if args.type_net == 'hs':
        args.name += '_dof{}'.format(args.dof)
    if args.tensorboard:
        directory = 'logs/{}'.format(args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        writer = SummaryWriter(directory)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    iter_per_epoch = len(train_loader)

    # create model
    model = MLP(784, 10, layer_dims=args.ldims, type_net=args.type_net, N=60000, dof=args.dof, beta_ema=args.beta_ema)
    thres_stds = args.thres_std

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            total_steps = checkpoint['total_steps']
            if checkpoint['beta_ema'] > 0:
                model.avg_param = checkpoint['avg_params']
                model.steps_ema = checkpoint['steps_ema']
                model.beta_ema = checkpoint['beta_ema']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    loglike = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loglike = loglike.cuda()

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        annealing = 1.
        if args.anneal_kl:
            annealing = kl_linear(args.epzero, args.epmax, iter_per_epoch, total_steps, maxval=args.anneal_maxval)
            if annealing > args.anneal_maxval or annealing < 0.:
                raise Exception()
            if args.tensorboard:
                writer.add_scalar('annealing', annealing, total_steps)

        total_loss = loss + model.kl_div(annealing=annealing, type_anneal=args.anneal_type)
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    criterion = loss_function

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, thres_stds=thres_stds)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'beta_ema': model.beta_ema,
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps
        }
        if model.beta_ema > 0:
            state['avg_params'] = model.avg_param
            state['steps_ema'] = model.steps_ema

        save_checkpoint(state, is_best)
    print('Best accuracy: ', best_prec1)
    if args.tensorboard:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, thres_stds=()):
    """Train for one epoch on the training set"""
    global total_steps
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    ema_loss, steps = 0, 0
    for i, (input_, target) in enumerate(train_loader):
        steps += 1
        total_steps += 1
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_.view(-1, 784))
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1[0], input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp the variances
        if args.clip_var:
            for k, layer in enumerate(model.layers):
                layer.constrain_parameters(thres_std=thres_stds[k])

        if model.beta_ema > 0.:
            model.update_ema()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # input()
        if i % args.print_freq == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()
    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_.view(-1, 784))
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1.item(), input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    if model.beta_ema > 0:
        model.load_params(old_params)
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')


if __name__ == '__main__':
    main()
