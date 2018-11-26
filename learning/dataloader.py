import numpy as np
from torch.utils import data
import os
import scipy


class EyeDataset(data.Dataset):
    def __init__(self, path, normalization):
        self.path = path
        self.file_list = os.listdir(os.path.join(self.path, 'image'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):

        filename = self.file_list[item]

        image_path = os.path.join(os.path.join(self.path, 'image'), filename)
        label_path = os.path.join(os.path.join(self.path, 'filters'), filename)

        image = scipy.ndimage.imread(image_path)
        label = scipy.ndimage.imread(label_path)

        print(image.shape, label.shape)

        #TODO: normalize

        return image, label



        