import numpy as np
from torch.utils import data
import os
from PIL import Image
from scipy import signal
from utils import gaussian2D


class EyeDataset(data.Dataset):
    def __init__(self, path, normalization):
        self.path = path
        file_list = os.listdir(os.path.join(self.path, 'image'))
        self.file_list = []
        for each in file_list:
            if '._' not in each and '.jpg' in each:
                self.file_list.append(each)

        self.kernel = gaussian2D(sigma=2)
        self.mean = normalization[0]
        self.std = normalization[1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):

        filename = self.file_list[item]

        image_path = os.path.join(self.path, 'image') + '/' + filename
        label_path = os.path.join(self.path, 'filters') + '/' + filename

        try:
            img = Image.open(image_path)
            img.load()
            image = np.asarray(img, dtype='float32')

            label = Image.open(label_path)
            label.load()
            label = np.asarray(label, dtype='float32')
            label = label > 0
            label = np.asarray(label, dtype='int64')

            image = np.transpose(image, (2, 0, 1))

            temp = image.copy()
            # for i in range(3):
            #     temp[i] = signal.convolve2d(image[i], self.kernel, mode='same')

            image = temp[:, ::5, ::5]
            label = label[::5, ::5]

            image = image / np.max(image)

            for i in range(3):
                image[i] = (image[i] - self.mean[i]) / self.std[i] # normalize

            # image = image - np.mean(image)
            # label = label - np.mean(label)
            return image, label

        except:
            print('error')
            return np.zeros((3,192, 256), dtype='float32'), np.zeros((192, 256), dtype='int64')



