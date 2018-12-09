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

from learning.models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, default='/mnt/md0/ImageNet', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.000001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
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
parser.add_argument('--top_k_n', type=int, default=8, metavar='N',
                        help='print')
best_prec1 = 0


def main():
    global args
    args = parser.parse_args()
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ]))
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    dense_net = models.densenet121(pretrained=True)
    vgg_features = models.vgg16(pretrained=True).features

    dense_net = dense_net.cuda()
    vgg_features = vgg_features.cuda()

    mask = Mask(top_k_n=args.top_k_n).cuda()
    mse_best = 99999

    if args.resume:
        dict_saved = torch.load(args.resume)
        mask.load_state_dict(dict_saved['state_dict'])

        args.start_epoch = dict_saved['epoch']
        mse_best = dict_saved['best_prec1']

    # optimizer = torch.optim.SGD(mask.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                                 weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(params = mask.parameters(), lr = args.lr,
                                weight_decay=args.weight_decay)

    model_list = [dense_net, vgg_features, mask]
    # criterion = #MSE


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(train_loader, model_list, optimizer, epoch)

        # evaluate on validation set
        mse = validate(val_loader, model_list)

        # remember best mse error and save checkpoint
        is_best = mse < mse_best
        mse_best = min(mse, mse_best)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': mask.state_dict(),
            'best_prec1': mse_best,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filetype = 'prior2n')


def train(train_loader, model_list, optimizer, epoch):
    vgg, vgg_features, mask_model = model_list

    mask_model.train()

    vgg.train()
    vgg_features.eval()

    RMSE_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()

    criterion = nn.CrossEntropyLoss().cuda()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        with torch.no_grad():
            fea = vgg_features(input_var)
        attention_mask = mask_model(fea.detach())

        masked_image = attention_mask * input_var   # apply the attention mask on the origin picture

        masked_image = masked_image / (args.top_k_n/49)
        class_pred = vgg.forward(masked_image)
        class_loss = criterion(class_pred, target_var)

        loss = class_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.data[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.avg:.3f}\t'                  
                  'classification Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses
            ))
            if i>5000:
                break


def validate(val_loader, model_list):
    RMSE_losses = AverageMeter()
    class_losses = AverageMeter()
    batch_time = AverageMeter()

    vgg, vgg_features, mask_model = model_list

    mask_model.eval()

    vgg.eval()
    vgg_features.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        target = target.cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        with torch.no_grad():
            fea = vgg_features(input_var)
            attention_mask = mask_model(fea.detach())

            masked_image = attention_mask * input_var  # apply the attention mask on the origin picture

            masked_image = masked_image / args.top_k_n * 49
            class_pred = vgg.forward(masked_image)
            class_loss = criterion(class_pred, target_var)

        class_losses.update(class_loss.data[0], input.size(0))

    print(' * TEST:'
          '* Prec@1 class {c_loss.avg:.3f}'.format(c_loss=class_losses))

    return class_losses.avg


if __name__ == '__main__':
    main()
