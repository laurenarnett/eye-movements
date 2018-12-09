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
parser.add_argument('--balance_num', default=200, type=int, metavar='N',
                    help='balance the prediction')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lambda_', default=0.5, type=float, metavar='M',
                    help='lambda for uncertainty loss')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
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
parser.add_argument('--model_type', default='distill', type=str, metavar='type',
                    help='type of the model')
parser.add_argument('--top_k_n', type=int, default=8, metavar='N',
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

    # vgg_features = models.vgg16(pretrained=True).features.cuda()
    vgg_fea_hi = Vgg16Hi().cuda()
    bayesian_pred = BayesPred_N(args).cuda()

    mask = Mask(top_k_n=args.top_k_n).cuda()
    checkpoint = torch.load("./prior2ncheckpoint.pth.tar")
    mask.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.SGD(bayesian_pred.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model_list = [vgg_fea_hi, bayesian_pred, mask]
    # criterion = #MSE

    mse_best = 99999

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(train_dataloader, model_list, optimizer, epoch, args)

        # evaluate on validation set
        mse = validate(test_dataloader, model_list, args)

        # remember best mse error and save checkpoint
        is_best = mse < mse_best
        mse_best = min(mse, mse_best)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': bayesian_pred.state_dict(),
            'best_prec1': mse_best,
            'optimizer': optimizer.state_dict(),
        }, is_best, filetype='bayes_{}_'.format(args.balance_num))


def train(train_loader, model_list, optimizer, epoch, args):
    vgg_fea_hi, fix_pred, mask = model_list

    fix_pred.train()
    vgg_fea_hi.eval()
    mask.eval()

    RMSE_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.asarray([1, args.balance_num], dtype='float32')).cuda()).cuda()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        with torch.no_grad():
            fea = vgg_fea_hi(input_var)
            attention_mask = mask(fea[-1])

        output = fix_pred(fea, attention_mask)

        loss = criterion.forward(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # RMSE_losses.update(mse_loss.data[0], input.size(0))
        losses.update(loss.data[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.avg:.3f}\t'                  
                  'normalized Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses
            ))


def validate(val_loader, model_list, args):
    RMSE_losses = AverageMeter()
    class_losses = AverageMeter()
    batch_time = AverageMeter()

    vgg_fea_hi, fix_pred, mask = model_list

    fix_pred.eval()
    vgg_fea_hi.eval()
    mask.eval()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.asarray([1, args.balance_num], dtype='float32')).cuda()).cuda()

    correct = 0
    cnt_all = 0
    pred_tobe_fix = 0
    gt_tobe_fix = 0
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        with torch.no_grad():
            fea = vgg_fea_hi(input_var)
            attention_mask = mask(fea[-1])

            output = F.softmax(fix_pred(fea, attention_mask), dim=1)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            TP = pred.eq(target.view_as(pred)).float() * pred.float()
            TP = TP.sum().item()
            cnt_all += pred.size(0) * pred.size(2) * pred.size(3)
            pred_tobe_fix += pred.sum().item()
            gt_tobe_fix += target.sum().item()

            loss = criterion.forward(output, target_var)

        # mse_loss = torch.sum((f_mean - target_var) ** 2) / f_mean.size(0) / f_mean.size(2) / f_mean.size(3)
        # loss = torch.sum((1 - args.lambda_) * (f_mean - target_var) ** 2 / (f_var + 1e-8)
        #                  + args.lambda_ * torch.log(f_var))
        # loss = loss / f_mean.size(0) / f_mean.size(2) / f_mean.size(3)

        # RMSE_losses.update(mse_loss.data[0], input.size(0))
        class_losses.update(loss.data[0], input.size(0))

    print('Test')
    print(' * Prec@1 acc {acc:.8f}\t  precision {prec: .8f} \t recall {recal: .8f}\t'
          '* Prec@1 normalized loss {class_losses.avg:.8f}'.format(acc= correct / cnt_all,
                                                                   prec=TP / pred_tobe_fix,
                                                                   recal=TP / gt_tobe_fix,
                                                                   class_losses=class_losses))

    return correct / cnt_all


if __name__ == '__main__':
    main()

