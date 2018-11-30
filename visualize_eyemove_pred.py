import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import json
import sys
import os
from learning.models import FixPred, Vgg16Hi
import matplotlib.pyplot as plt
import cv2
from torch.utils import data
from utils import *
from learning.dataloader import EyeDataset
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

def main():
    parser.add_argument('--model_type', default='distill', type=str, metavar='type',
                        help='type of the model')
    args = parser.parse_args()

    vgg16hi = Vgg16Hi().cuda()
    fix_pred = FixPred(args).cuda()
    checkpoint = torch.load("baselinecheckpoint.pth.tar")
    fix_pred.load_state_dict(checkpoint["state_dict"])

    normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    batchsize=1

    dataPath = '/mnt/md0/eye_move_data'

    train_dataset = EyeDataset(
        path=os.path.join(dataPath, 'train'),
        normalization=normalize
    )

    test_dataset = EyeDataset(
        path=os.path.join(dataPath, 'test'),
        normalization=normalize
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=False
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )
    visualize_loop(test_dataloader, [vgg16hi, fix_pred])


def visualize_loop(loader, models):
    vgg_features, fix_pred = models
    for i, (input, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input.cuda())

        with torch.no_grad():
            fea = vgg_features(input_var)
            output = fix_pred(fea)
            output = F.softmax(output, dim=1)

        output = output.data.cpu().numpy()
        gt = target.numpy()
        img = input.data.cpu().numpy()[0]

        fig, axis = plt.subplots(1, 4)
        axis[0].imshow(output[0,0])
        axis[1].imshow(output[0,1])
        axis[2].imshow(gt[0])
        axis[3].imshow(img.transpose(1,2,0))
        # axis[2].imshow(gt[0].transpose(1,2,0))

        plt.show()



if __name__ == "__main__":
    main()