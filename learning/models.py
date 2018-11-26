from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sru import SRU
from learning.convLSTM import ConvLSTM


class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        k = 7
        x_resize = x.resize(x.size()[0], k*k)
        topk, indices = torch.topk(x_resize, 5)

        # print('indices shape', indices.size())   ## batch  topk_num

        mask_ind = torch.zeros(x.size()[0], 1, k, k).cuda()
        for b in range(x.size()[0]):
            for i in range(indices.shape[1]):
                row_i = indices[b, i] // k
                column_i = indices[b, i] % k
                mask_ind[b, 0, row_i, column_i] = 1

        x = x * mask_ind

        x = x > 0
        x = self.upsample(x.float())   # get the attention mask matrix here

        return x


class Reconst(nn.Module):
    def __init__(self):
        super(Reconst, self).__init__()

        l = [32, 64, 128, 256, 256, 256]

        self.conv1 = nn.Conv2d(3, l[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(l[0], l[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(l[1], l[2], kernel_size=5, stride=2, padding=2)

        self.conv4 = nn.Conv2d(l[2], l[3], kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(l[3], l[4], kernel_size=3, stride=2, padding=1)
        # self.conv6 = nn.Conv2d(l[4], l[5], kernel_size=3, stride=2, padding=1)

        # self.deconv6 = nn.ConvTranspose2d(l[5], l[4], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(l[4], l[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(l[3], l[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deconv3 = nn.ConvTranspose2d(l[2] * 2, l[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(l[1], l[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(l[0], 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.pr1 = nn.PReLU()
        self.pr2 = nn.PReLU()
        self.pr3 = nn.PReLU()
        self.pr4 = nn.PReLU()
        self.pr5 = nn.PReLU()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x1))
        x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))

        # x = self.pr1(self.deconv6(x))
        x = self.pr2(self.deconv5(x))
        x = self.pr3(self.deconv4(x))

        x = torch.cat((x, x1), 1)
        x = self.pr4(self.deconv3(x))
        x = self.pr5(self.deconv2(x))
        x = self.deconv1(x)

        return x


class FixPred(nn.Module):
    def __init__(self):
        super(FixPred, self).__init__()

        l=[512, 256, 128, 64, 32]

        self.deconv5 = nn.ConvTranspose2d(l[4], l[3], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(l[3], l[2], kernel_size=5, stride=2, padding=2, output_padding=1)

        self.deconv3 = nn.ConvTranspose2d(l[2], l[1], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(l[1], l[0], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv11 = nn.ConvTranspose2d(l[0], 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv12 = nn.ConvTranspose2d(l[0], 1, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.pr2 = nn.PReLU()
        self.pr3 = nn.PReLU()
        self.pr4 = nn.PReLU()
        self.pr5 = nn.PReLU()

    def forward(self, x):

        x = self.pr2(self.deconv5(x))
        x = self.pr3(self.deconv4(x))

        x = self.pr4(self.deconv3(x))
        x = self.pr5(self.deconv2(x))
        mean = self.deconv11(x)
        var = self.deconv12(x)

        return mean, var




