import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data

from learning.models import *
from utils import *
from learning.dataloader import EyeDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, default='/mnt/md0/eye_move_data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lambda_', default=0.5, type=float, metavar='M',
                    help='lambda for uncertainty loss')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_prior', default='/home/mcz/eye-movements/model_best_prior.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--print_freq', type=int, default=10000, metavar='N',
                        help='print')
best_prec1 = 0


def main():
    global args
    args = parser.parse_args()
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    train_dataset = EyeDataset(
        path=os.path.join(args.data, 'train'),
        normalization=normalize
    )

    test_dataset = EyeDataset(
        path=os.path.join(args.data, 'test'),
        normalization=normalize
    )

    train_dataloader = data.DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    mask = Mask().cuda()
    dict_saved = torch.load(args.resume_prior)
    mask.load_state_dict(dict_saved['state_dict'])

    args.start_epoch = dict_saved['epoch']
    vgg_features1 = models.vgg16(pretrained=True).features.cuda()

    bayesian_pred = BayesPred().cuda()

    optimizer = torch.optim.SGD(bayesian_pred.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model_list = [vgg_features1, bayesian_pred, mask]
    # criterion = #MSE

    mse_best = 99999

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(train_dataloader, model_list, optimizer, epoch)

        # evaluate on validation set
        mse = validate(test_dataloader, model_list)

        # remember best mse error and save checkpoint
        is_best = mse < mse_best
        mse_best = min(mse, mse_best)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': bayesian_pred.state_dict(),
            'best_prec1': mse_best,
            'optimizer': optimizer.state_dict(),
        }, is_best, filetype='bayesian')


def train(train_loader, model_list, optimizer, epoch):
    vgg_features1, bayesian_pred, mask = model_list

    bayesian_pred.train()
    mask.eval()
    vgg_features1.eval()

    RMSE_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        with torch.no_grad():
            fea = vgg_features1(input_var)

            mask_prior = mask(fea)

        f_mean, f_var = bayesian_pred(fea.detach(), mask_prior)

        loss = torch.sum((1 - args.lambda_) * (f_mean - target_var) ** 2 / (f_var + 1e-8)
                         + args.lambda_ * torch.log(f_var + 1e-8))
        loss = loss / f_mean.size(0) / f_mean.size(2) / f_mean.size(3)

        mse_loss = torch.sum((f_mean - target_var) ** 2) / f_mean.size(0) / f_mean.size(2) / f_mean.size(3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        RMSE_losses.update(mse_loss.data[0], input.size(0))
        losses.update(loss.data[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'RMSE Loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'normalized Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, rloss=RMSE_losses, loss=losses
            ))


def validate(val_loader, model_list):
    RMSE_losses = AverageMeter()
    class_losses = AverageMeter()
    batch_time = AverageMeter()

    vgg_features1, bayesian_pred, mask = model_list

    vgg_features1.eval()
    bayesian_pred.eval()
    mask.eval()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        fea = vgg_features1(input_var)
        mask_prior = mask(fea)
        f_mean, f_var = bayesian_pred(fea.detach(), mask_prior)

        mse_loss = torch.sum((f_mean - target_var) ** 2) / f_mean.size(0) / f_mean.size(2) / f_mean.size(3)
        loss = torch.sum((1 - args.lambda_) * (f_mean - target_var) ** 2 / (f_var + 1e-8)
                         + args.lambda_ * torch.log(f_var))
        loss = loss / f_mean.size(0) / f_mean.size(2) / f_mean.size(3)

        RMSE_losses.update(mse_loss.data[0], input.size(0))
        class_losses.update(loss.data[0], input.size(0))

    print('Test')
    print(' * Prec@1 RMSE {RMSE_losses.avg:.3f}\t '
          '* Prec@1 normalized loss {class_losses.avg:.3f}'.format(RMSE_losses=RMSE_losses, class_losses=class_losses))

    return RMSE_losses.avg


if __name__ == '__main__':
    main()

