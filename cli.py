import click
import os
import shutil
import time

import numpy as np

from utils import AverageMeter, accuracy, construct_optimizer
from models import MLP, LeNet5, BaseCNN
from losses import CrossEntropyLossWithAnnealing
from optimizers import VProp

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


def set_directory(name: str,
                  type_net: str,
                  dof: str) -> (str, str):
    name += f'_{type_net}'
    if type_net == 'hs':
        name += f'_dof{dof}'
    directory = f'logs/{name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return name, directory


def load_cifar10(batch_size: int,
                 num_workers: int = 4,
                 pin_memory: bool = torch.cuda.is_available(),
                 augment: bool = True) -> (object, object, int):
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data',
                       train=True,
                       download=True,
                       transform=transform_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data',
                       train=False,
                       transform=transform_test),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

    iter_per_epoch = len(train_loader)

    return train_loader, val_loader, iter_per_epoch


def load_mnist(batch_size: int,
               num_workers: int = 4,
               pin_memory: bool = torch.cuda.is_available()) -> (object, object, int):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
                       train=True,
                       download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
                       train=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

    iter_per_epoch = len(train_loader)

    return train_loader, val_loader, iter_per_epoch


def resume_from_checkpoint(resume_path: str,
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer) -> (int, float, int, torch.nn.Module, torch.optim.Optimizer):
    if os.path.isfile(resume_path):
        print(f"=> loading checkpoint '{resume_path}'")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['beta_ema'] > 0:
            model.avg_param = checkpoint['avg_params']
            model.steps_ema = checkpoint['steps_ema']
            model.beta_ema = checkpoint['beta_ema']
        print(f"=> loaded checkpoint '{resume_path}' (epoch {checkpoint['epoch']})")

        return checkpoint['epoch'], checkpoint['best_prec1'], checkpoint['total_steps'], model, optimizer

    return 0, 0., 0, model, optimizer


def save_checkpoint(state: object,
                    is_best: bool,
                    name: str,
                    filename: str ='checkpoint.pth.tar'):
    directory = f'runs/{name}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, filename)
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'runs/{name}/model_best.pth.tar')


def train_single_epoch(train_loader: object,
                       model: torch.nn.Module,
                       criterion: object,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       clip_var: float,
                       total_steps: int,
                       print_freq: int,
                       writer: object,
                       thres_stds: tuple =(),
                       shape: list = None) -> int:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    ema_loss, steps = 0, 0
    for i, (input_, target) in enumerate(train_loader):
        steps += 1
        total_steps += 1

        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()

        if shape is None:
            input_var = torch.autograd.Variable(input_)
        else:
            input_var = torch.autograd.Variable(input_.view(shape))
        target_var = torch.autograd.Variable(target)

        if isinstance(optimizer, VProp):
            # Calculate noisy loss
            optimizer.add_noise_to_parameters()
            output = model(input_var)
            loss = criterion(output, target_var, model)

            # Do an update
            optimizer.zero_grad()
            loss.backward()
            optimizer.remove_noise_from_parameters()
            optimizer.step()

            # Calculate clean loss to update metrics
            output = model(input_var)
            loss = criterion(output, target_var, model)

        else:
            optimizer.zero_grad()
            output = model(input_var)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(output, target_var)
            else:
                loss = criterion(output, target_var, model)
            loss.backward()
            if isinstance(optimizer, torch.optim.LBFGS):
                def closure():
                    return loss
                optimizer.step(closure)
            else:
                optimizer.step()

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1, input_.size(0))

        if clip_var:
            for k, layer in enumerate(model.layers):
                layer.constrain_parameters(thres_std=thres_stds[k])

        if isinstance(model, torch.nn.DataParallel):
            if model.module.beta_ema > 0.:
                model.module.update_ema()
        else:
            if model.beta_ema > 0.:
                model.update_ema()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f' Epoch: [{epoch}][{i}/{len(train_loader)}]\t' +
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' +
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', top1.avg, epoch)

    return total_steps


def validate(val_loader: object,
             model: torch.nn.Module,
             criterion: object,
             epoch: int,
             print_freq: int,
             writer: object,
             shape: list = None) -> float:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    if isinstance(model, torch.nn.DataParallel):
        if model.module.beta_ema > 0:
            old_params = model.module.get_params()
            model.module.load_ema_params()
    else:
        if model.beta_ema > 0:
            old_params = model.get_params()
            model.load_ema_params()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            input_ = input_.cuda()

        if shape is None:
            input_var = torch.autograd.Variable(input_)
        else:
            input_var = torch.autograd.Variable(input_.view(shape))
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(output, target_var)
        else:
            loss = criterion(output, target_var, model)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1, input_.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t' +
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' +
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f}')
    if isinstance(model, torch.nn.DataParallel):
        if model.module.beta_ema > 0:
            model.module.load_params(old_params)
    else:
        if model.beta_ema > 0:
            model.load_params(old_params)

    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/acc', top1.avg, epoch)

    return top1.avg


@click.group()
def cli():
    pass


@cli.command()
@click.option('--epochs', default=200, type=int)
@click.option('--start_epoch', default=0, type=int)
@click.option('--batch_size', default=100, type=int)
@click.option('--lr', default=0.001, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--print_freq', default=100, type=int)
@click.option('--resume', default='', type=str)
@click.option('--name', default='MLP', type=str)
@click.option('--tensorboard', type=bool, default=False)
@click.option('--multi_gpu', default=False)
@click.option('--thres_std', type=list, default=[0.2, 0.5, 1.0])
@click.option('--clip_var', default=False)
@click.option('--type_net', type=click.Choice(['hs', 'dropout', 'map', 'gauss', 'kernel', 'kernelbayes', 'orth', 'orthbayes']), default='gauss')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd', 'cocob', 'ftml']), default='adadelta')
@click.option('--restart_optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd', 'cocob', 'ftml']), default='adadelta')
@click.option('--restart_lr', default=0.001, type=float)
@click.option('--restart', type=bool, default=False)
@click.option('--restart_interval', type=int, default=100)
@click.option('--anneal_kl', default=False)
@click.option('--epzero', type=int, default=0)
@click.option('--epmax', type=int, default=100)
@click.option('--anneal_maxval', type=float, default=1.)
@click.option('--anneal_type', type=click.Choice(['kl', 'q', 'none']), default='none')
@click.option('--anneal_schedule', type=click.Choice(['linear']), default='linear')
@click.option('--dof', type=float, default=1.)
@click.option('--beta_ema', type=float, default=0.)
@click.option('--ldims', type=list, default=[1024, 1024])
@click.option('--ep_anneal', type=int, default=10)
@click.option('--use_mask', type=bool, default=True)
@click.option('--mask_prob', type=float, default=0.5)
@click.option('--save_at', type=list, default=[1, 10, 50, 100])
@click.option('--device', type=int, default=0)
def train_mlp(**kwargs):
    name, directory = set_directory(name=kwargs['name'],
                                    type_net=kwargs['type_net'],
                                    dof=kwargs['dof'])
    if kwargs['tensorboard']:
        writer = SummaryWriter(directory)
    else:
        writer = None

    train_loader, val_loader, iter_per_epoch = load_mnist(batch_size=kwargs['batch_size'])

    if kwargs['use_mask']:
        if kwargs['type_net'] == 'map' or kwargs['type_net'] == 'gauss':
            mask = [[np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(784, 1024), (1024,)]]]
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(1024, 1024), (1024,)]])
        else:
            mask = None
    else:
        mask = None

    model = MLP(input_dim=784,
                num_classes=10,
                layer_dims=kwargs['ldims'],
                type_net=kwargs['type_net'],
                N=60000,
                dof=kwargs['dof'],
                beta_ema=kwargs['beta_ema'],
                mask=mask)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if kwargs['multi_gpu']:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr'])

    if kwargs['resume'] != '':
        kwargs['start_epoch'], best_prec1, total_steps, model, optimizer = resume_from_checkpoint(resume_path=kwargs['resume'],
                                                                                                  model=model,
                                                                                                  optimizer=optimizer)
    else:
        total_steps = 0
        best_prec1 = 0.

    cudnn.benchmark = True

    if kwargs['type_net'] == 'kerneldense':
        loss_function = torch.nn.CrossEntropyLoss().cuda()
    else:
        loss_function = CrossEntropyLossWithAnnealing(iter_per_epoch=iter_per_epoch,
                                                      total_steps=total_steps,
                                                      anneal_type=kwargs['anneal_type'],
                                                      anneal_kl=kwargs['anneal_kl'],
                                                      epzero=kwargs['epzero'],
                                                      epmax=kwargs['epmax'],
                                                      anneal_maxval=kwargs['anneal_maxval'],
                                                      writer=writer)

    for epoch in range(kwargs['start_epoch'], kwargs['epochs']):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         clip_var=kwargs['clip_var'],
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         writer=writer,
                                         thres_stds=kwargs['thres_std'],
                                         shape=[-1, 784])

        prec1 = validate(val_loader=val_loader,
                         model=model,
                         criterion=loss_function,
                         epoch=epoch,
                         print_freq=kwargs['print_freq'],
                         shape=[-1, 784],
                         writer=writer)

        if kwargs['restart'] and epoch % kwargs['restart_interval'] == 0:
            print('Restarting optimizer...')
            optimizer = construct_optimizer(optimizer=kwargs['restart_optim'],
                                            model=model,
                                            lr=kwargs['restart_lr'])

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        if isinstance(model, torch.nn.DataParallel):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max(prec1, best_prec1),
                'beta_ema': model.module.beta_ema,
                'optimizer': optimizer.state_dict(),
                'total_steps': total_steps
            }
            if model.module.beta_ema > 0:
                state['avg_params'] = model.module.avg_param
                state['steps_ema'] = model.module.steps_ema
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max(prec1, best_prec1),
                'beta_ema': model.beta_ema,
                'optimizer': optimizer.state_dict(),
                'total_steps': total_steps
            }
            if model.beta_ema > 0:
                state['avg_params'] = model.avg_param
                state['steps_ema'] = model.steps_ema

        if epoch in kwargs['save_at']:
            name = f'checkpoint_{epoch}.pth.tar'
        else:
            name = 'checkpoint.pth.tar'

        save_checkpoint(state=state,
                        is_best=is_best,
                        name=name)

    print('Best accuracy: ', best_prec1)

    if writer is not None:
        writer.close()


@cli.command()
@click.option('--epochs', default=300, type=int)
@click.option('--start_epoch', default=0, type=int)
@click.option('--batch_size', default=100, type=int)
@click.option('--lr', default=0.001, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--print_freq', default=100, type=int)
@click.option('--resume', default='', type=str)
@click.option('--name', default='Lenet5', type=str)
@click.option('--tensorboard', type=bool, default=False)
@click.option('--multi_gpu', default=False)
@click.option('--thres_std', type=list, default=[0.2, 0.5, 1.0])
@click.option('--clip_var', default=False)
@click.option('--type_net', type=click.Choice(['hs', 'gauss', 'dropout', 'map', 'kernel', 'kernelbayes', 'orth', 'orthbayes']), default='map')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd', 'cocob', 'ftml']), default='adadelta')
@click.option('--restart', type=bool, default=False)
@click.option('--restart_interval', type=int, default=100)
@click.option('--restart_optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd', 'cocob', 'ftml']), default='adadelta')
@click.option('--restart_lr', default=0.001, type=float)
@click.option('--anneal_kl', default=False)
@click.option('--epzero', type=int, default=0)
@click.option('--epmax', type=int, default=100)
@click.option('--anneal_maxval', type=float, default=1.)
@click.option('--anneal_type', type=click.Choice(['kl', 'q', 'none']), default='none')
@click.option('--anneal_schedule', type=click.Choice(['linear']), default='linear')
@click.option('--dof', type=float, default=1.)
@click.option('--beta_ema', type=float, default=0.)
@click.option('--use_mask', type=bool, default=True)
@click.option('--mask_prob', type=float, default=0.5)
@click.option('--save_at', type=list, default=[1, 10, 50, 100])
@click.option('--device', type=int, default=0)
def train_lenet(**kwargs):
    if kwargs['tensorboard']:
        name, directory = set_directory(name=kwargs['name'],
                                        type_net=kwargs['type_net'],
                                        dof=kwargs['dof'])
        writer = SummaryWriter(directory)
    else:
        writer = None

    train_loader, val_loader, iter_per_epoch = load_mnist(batch_size=kwargs['batch_size'])

    if kwargs['use_mask']:
        if kwargs['type_net'] == 'map' or kwargs['type_net'] == 'gauss':
            mask = [[np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(32, 1, 5, 5), (32,)]]]
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(64, 32, 5, 5), (64,)]])
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(1024, 512), (512,)]])
        else:
            mask = None
    else:
        mask = None

    model = LeNet5(num_classes=10,
                   type_net=kwargs['type_net'],
                   N=60000,
                   beta_ema=kwargs['beta_ema'],
                   dof=kwargs['dof'],
                   mask=mask)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if kwargs['multi_gpu']:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr'])

    if kwargs['resume'] != '':
        kwargs['start_epoch'], best_prec1, total_steps, model, optimizer = resume_from_checkpoint(resume_path=kwargs['resume'],
                                                                                                  model=model,
                                                                                                  optimizer=optimizer)
    else:
        total_steps = 0
        best_prec1 = 0.

    cudnn.benchmark = True

    loss_function = CrossEntropyLossWithAnnealing(iter_per_epoch=iter_per_epoch,
                                                  total_steps=total_steps,
                                                  anneal_type=kwargs['anneal_type'],
                                                  anneal_kl=kwargs['anneal_kl'],
                                                  epzero=kwargs['epzero'],
                                                  epmax=kwargs['epmax'],
                                                  anneal_maxval=kwargs['anneal_maxval'],
                                                  writer=writer)

    for epoch in range(kwargs['start_epoch'], kwargs['epochs']):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         clip_var=kwargs['clip_var'],
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         writer=writer,
                                         thres_stds=kwargs['thres_std'])

        prec1 = validate(val_loader=val_loader,
                         model=model,
                         criterion=loss_function,
                         epoch=epoch,
                         print_freq=kwargs['print_freq'],
                         writer=writer)

        if kwargs['restart'] and epoch % kwargs['restart_interval'] == 0:
            print('Restarting optimizer...')
            optimizer = construct_optimizer(optimizer=kwargs['restart_optim'],
                                            model=model,
                                            lr=kwargs['restart_lr'])

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        if isinstance(model, torch.nn.DataParallel):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max(prec1, best_prec1),
                'beta_ema': model.module.beta_ema,
                'optimizer': optimizer.state_dict(),
                'total_steps': total_steps
            }
            if model.module.beta_ema > 0:
                state['avg_params'] = model.module.avg_param
                state['steps_ema'] = model.module.steps_ema
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max(prec1, best_prec1),
                'beta_ema': model.beta_ema,
                'optimizer': optimizer.state_dict(),
                'total_steps': total_steps
            }
            if model.beta_ema > 0:
                state['avg_params'] = model.avg_param
                state['steps_ema'] = model.steps_ema

        if epoch in kwargs['save_at']:
            name = f'checkpoint_{epoch}.pth.tar'
        else:
            name = 'checkpoint.pth.tar'

        save_checkpoint(state=state,
                        is_best=is_best,
                        name=name)
    print('Best accuracy: ', best_prec1)

    if writer is not None:
        writer.close()


@cli.command()
@click.option('--epochs', default=300, type=int)
@click.option('--start_epoch', default=0, type=int)
@click.option('--batch_size', default=100, type=int)
@click.option('--lr', default=0.001, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--print_freq', default=100, type=int)
@click.option('--resume', default='', type=str)
@click.option('--name', default='Basecnn', type=str)
@click.option('--tensorboard', type=bool, default=False)
@click.option('--multi_gpu', default=False)
@click.option('--thres_std', type=list, default=[0.2, 0.5, 1.0])
@click.option('--clip_var', default=False)
@click.option('--type_net', type=click.Choice(['hs', 'gauss', 'dropout', 'map', 'kernel', 'kernelbayes', 'orth', 'orthbayes']), default='map')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd', 'cocob', 'ftml']), default='adadelta')
@click.option('--restart', type=bool, default=False)
@click.option('--restart_interval', type=int, default=100)
@click.option('--restart_optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd', 'cocob', 'ftml']), default='adadelta')
@click.option('--restart_lr', default=0.001, type=float)
@click.option('--anneal_kl', default=False)
@click.option('--epzero', type=int, default=0)
@click.option('--epmax', type=int, default=100)
@click.option('--anneal_maxval', type=float, default=1.)
@click.option('--anneal_type', type=click.Choice(['kl', 'q', 'none']), default='kl')
@click.option('--anneal_schedule', type=click.Choice(['linear']), default='linear')
@click.option('--dof', type=float, default=1.)
@click.option('--beta_ema', type=float, default=0.)
@click.option('--use_mask', type=bool, default=True)
@click.option('--mask_prob', type=float, default=0.5)
@click.option('--save_at', type=list, default=[1, 10, 50, 100])
@click.option('--device', type=int, default=0)
def train_basecnn(**kwargs):
    if kwargs['tensorboard']:
        name, directory = set_directory(name=kwargs['name'],
                                        type_net=kwargs['type_net'],
                                        dof=kwargs['dof'])
        writer = SummaryWriter(directory)
    else:
        writer = None

    train_loader, val_loader, iter_per_epoch = load_cifar10(batch_size=kwargs['batch_size'])

    if kwargs['use_mask']:
        if kwargs['type_net'] == 'map' or kwargs['type_net'] == 'gauss':
            mask = [[np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(96, 3, 5, 5), (96,)]]]
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(128, 96, 5, 5), (128,)]])
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(256, 128, 5, 5), (256,)]])
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(2304, 2048), (2048,)]])
            mask.append([np.random.binomial(n=1, p=kwargs['mask_prob'], size=p).astype('float32') for p in [(2048, 2048), (2048,)]])
        else:
            mask = None
    else:
        mask = None

    model = BaseCNN(num_classes=10,
                    model_size=1,
                    type_net=kwargs['type_net'],
                    N=50000,
                    beta_ema=kwargs['beta_ema'],
                    mask=mask)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if kwargs['multi_gpu']:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr'])

    if kwargs['resume'] != '':
        kwargs['start_epoch'], best_prec1, total_steps, model, optimizer = resume_from_checkpoint(resume_path=kwargs['resume'],
                                                                                                  model=model,
                                                                                                  optimizer=optimizer)
    else:
        total_steps = 0
        best_prec1 = 0.

    cudnn.benchmark = True

    loss_function = CrossEntropyLossWithAnnealing(iter_per_epoch=iter_per_epoch,
                                                  total_steps=total_steps,
                                                  anneal_type=kwargs['anneal_type'],
                                                  anneal_kl=kwargs['anneal_kl'],
                                                  epzero=kwargs['epzero'],
                                                  epmax=kwargs['epmax'],
                                                  anneal_maxval=kwargs['anneal_maxval'],
                                                  writer=writer)

    for epoch in range(kwargs['start_epoch'], kwargs['epochs']):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         clip_var=kwargs['clip_var'],
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         writer=writer,
                                         thres_stds=kwargs['thres_std'])

        prec1 = validate(val_loader=val_loader,
                         model=model,
                         criterion=loss_function,
                         epoch=epoch,
                         print_freq=kwargs['print_freq'],
                         writer=writer)

        if kwargs['restart'] and epoch % kwargs['restart_interval'] == 0:
            print('Restarting optimizer...')
            optimizer = construct_optimizer(optimizer=kwargs['restart_optim'],
                                            model=model,
                                            lr=kwargs['restart_lr'])

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        if isinstance(model, torch.nn.DataParallel):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max(prec1, best_prec1),
                'beta_ema': model.module.beta_ema,
                'optimizer': optimizer.state_dict(),
                'total_steps': total_steps
            }
            if model.module.beta_ema > 0:
                state['avg_params'] = model.module.avg_param
                state['steps_ema'] = model.module.steps_ema
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max(prec1, best_prec1),
                'beta_ema': model.beta_ema,
                'optimizer': optimizer.state_dict(),
                'total_steps': total_steps
            }
            if model.beta_ema > 0:
                state['avg_params'] = model.avg_param
                state['steps_ema'] = model.steps_ema

        if epoch in kwargs['save_at']:
            name = f'checkpoint_{epoch}.pth.tar'
        else:
            name = 'checkpoint.pth.tar'

        save_checkpoint(state=state,
                        is_best=is_best,
                        name=name)
    print('Best accuracy: ', best_prec1)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    cli()
