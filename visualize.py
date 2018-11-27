import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import json
import sys
from learning.models import Mask, Reconst
import matplotlib.pyplot as plt
import cv2

def main(valdir):
    mask = Mask().cuda()
    reconst = Reconst().cuda()
    checkpoint = torch.load("../../cm3797/checkpoint.pth.tar")
    mask.load_state_dict(checkpoint["state_dict"])
    reconst.load_state_dict(checkpoint["reconstruct"])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])),
            batch_size=32, shuffle=False,
            num_workers=2, pin_memory=True)

    vgg = models.vgg16(pretrained=True)
    vgg_features = models.vgg16(pretrained=True).features

    vgg = vgg.cuda()
    vgg_features = vgg_features.cuda()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        fea = vgg_features(input_var)
        attention_mask = mask(fea.detach())

        masked_image = attention_mask * input_var  # apply the attention mask on the origin picture
        mask_image_np = masked_image.data.cpu().numpy()[0]
        #mask_image_np = mask_image_np.reshape(224,224,3)
        cv2.imwrite("./viz/mask/checkpoint_1/" + str(i) + "_mask", mask_image_np)
        


if __name__ == "__main__":
    main(sys.argv[1])
