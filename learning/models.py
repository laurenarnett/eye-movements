from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sru import SRU
from learning.convLSTM import ConvLSTM
from torchvision.models import vgg16


class Vgg16Hi(nn.Module):
    def __init__(self):
        super(Vgg16Hi, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                # print('shape', x.size())
                results.append(x)
        return results


class Mask(nn.Module):
    def __init__(self, top_k_n=5):
        super(Mask, self).__init__()

        self.top_k_n = top_k_n
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        size_x = x.size()

        x_resize = x.resize(size_x[0], size_x[2]*size_x[3])
        topk, indices = torch.topk(x_resize, self.top_k_n)

        # print('indices shape', indices.size())   ## batch  topk_num

        mask_ind = torch.zeros(x.size()[0], 1, size_x[2], size_x[3]).cuda()
        for b in range(x.size()[0]):
            for i in range(indices.shape[1]):
                row_i = indices[b, i] // size_x[3]
                column_i = indices[b, i] % size_x[3]
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
    def __init__(self, args):
        super(FixPred, self).__init__()

        # l=[512, 256, 128, 64, 32]
        l = [32, 64, 128, 256, 512]
        self.model_type = args.model_type

        if args.model_type == 'raw':
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

        elif args.model_type == 'distill':

            self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
            self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)


            self.conv1 = nn.Conv2d(l[4], l[3], kernel_size=5, padding=2)
            self.conv12 = nn.Conv2d(l[4], l[3], kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(l[3], l[2], kernel_size=5, padding=2)
            self.conv22 = nn.Conv2d(l[3], l[2], kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(l[2], l[1], kernel_size=5, padding=2)
            self.conv32 = nn.Conv2d(l[1], l[1], kernel_size=5, padding=2)
            self.conv41 = nn.Conv2d(l[1], 2, kernel_size=5, padding=2)
            # self.conv42 = nn.Conv2d(l[1], 1, kernel_size=5, padding=2)

    def forward(self, x):
        if self.model_type == 'raw':
            x = self.pr2(self.deconv5(x))
            x = self.pr3(self.deconv4(x))

            x = self.pr4(self.deconv3(x))
            x = self.pr5(self.deconv2(x))
            mean = self.deconv11(x)
            var = self.deconv12(x)

            return mean, torch.exp(var)

        elif self.model_type == 'distill':

            x1, x2, x3, x4 = x
            x = F.relu(self.conv1(x4))
            x = self.upsample1(x)
            x = torch.cat((x, x3), dim=1)
            x = F.relu(self.conv12(x))
            x = F.relu(self.conv2(x))
            x = self.upsample2(x)
            x = torch.cat((x, x2), dim=1)
            x = F.relu(self.conv22(x))
            x = F.relu(self.conv3(x))
            x = self.upsample3(x)
            # x = torch.cat((x, x1), dim=1)
            x = F.relu(self.conv32(x))
            mean = self.conv41(x)
            # var = self.conv42(x)

            return mean


class BayesPred(nn.Module):
    def __init__(self, args):
        super(BayesPred, self).__init__()

        l = [32, 64, 128, 256, 512]

        self.deconv5 = nn.ConvTranspose2d(l[4], l[3], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(l[3], l[2], kernel_size=5, stride=2, padding=2, output_padding=1)

        self.deconv3 = nn.ConvTranspose2d(l[2], l[1], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(l[1] + 1, l[0], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(l[0], l[0], kernel_size=5, stride=1, padding=2, output_padding=0)
        self.deconv11 = nn.ConvTranspose2d(l[0], 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv12 = nn.ConvTranspose2d(l[0], 1, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.pr2 = nn.PReLU()
        self.pr3 = nn.PReLU()
        self.pr4 = nn.PReLU()
        self.pr5 = nn.PReLU()
        self.pr6 = nn.PReLU()

    def forward(self, x, mask_prior):
        x = self.pr2(self.deconv5(x))
        x = self.pr3(self.deconv4(x))

        x = self.pr4(self.deconv3(x))

        mask_prior = mask_prior[:, :, ::4, ::4]
        x = torch.cat((mask_prior, x), dim=1)

        x = self.pr5(self.deconv2(x))
        x = self.pr6(self.deconv1(x))
        mean = self.deconv11(x)
        var = self.deconv12(x)

        return mean, torch.exp(var)




