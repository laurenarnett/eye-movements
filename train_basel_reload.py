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
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='baseline_50model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model_type', default='distill', type=str, metavar='type',
                    help='type of the model')
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
    fix_pred = FixPred(args).cuda()
    print('reload', args.resume)

    checkpoint = torch.load(args.resume)
    fix_pred.load_state_dict(checkpoint["state_dict"])

    model_list = [vgg_fea_hi, fix_pred]

    mse = validate(test_dataloader, model_list)


def validate(val_loader, model_list):
    RMSE_losses = AverageMeter()
    class_losses = AverageMeter()
    batch_time = AverageMeter()

    vgg_features, fix_pred = model_list

    vgg_features.eval()
    fix_pred.eval()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.asarray([1, 10], dtype='float32')).cuda()).cuda()

    correct = 0
    cnt_all = 0
    pred_tobe_fix = 0
    gt_tobe_fix = 0

    probability_save_list = []
    target_list = []
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        with torch.no_grad():
            fea = vgg_features(input_var)
            output = F.softmax(fix_pred(fea), dim=1)

            probability_save_list.append(output[:, 1].data.cpu().numpy())
            target_list.append(target.data.cpu().numpy())

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


    prob = np.concatenate(probability_save_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    print('shape', prob.shape, target_list.shape)
    np.save(args.resume[:12] + '_prob.npy', (prob, target_list))
    print('Test')
    print(' * Prec@1 acc {acc:.3f}\t  precision {prec: .8f} \t recall {recal: .8f}\t'
          '* Prec@1 normalized loss {class_losses.avg:.8f}'.format(acc= correct / cnt_all,
                                                                   prec=TP / pred_tobe_fix,
                                                                   recal=TP / gt_tobe_fix,
                                                                   class_losses=class_losses))

    return correct / cnt_all


if __name__ == '__main__':
    main()

