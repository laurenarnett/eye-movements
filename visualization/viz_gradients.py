import numpy as np
from PIL import Image
import sys
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
import matplotlib
matplotlib.use('agg')
import pylab as plt
from skimage.filters import try_all_threshold


def main(path_to_img,path_to_output):

    im = np.asarray(Image.open(path_to_img).convert("L"), dtype=float)
    grad_x = np.zeros((im.shape[0],im.shape[1]))
    grad_y = np.zeros((im.shape[1],im.shape[0]))
    amp = np.zeros((im.shape[0],im.shape[1]))

    # convolve to get gradients
    for i,row in enumerate(im):
        grad_x[i] = convolve1d(row,weights=[1,0, -1])

    for i,row in enumerate(im.T):
        grad_y[i] = convolve1d(row,weights=[1,0, -1])

    grad_y = grad_y.T


    # construct the amplitude
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            amp[i,j] = np.sqrt(grad_x[i,j]**2 + grad_y[i,j]**2)

    box_filter = np.ones((5,5))/25

    amp = convolve2d(amp, box_filter, mode='same')
    
    # make areas above a certain threshold brighter
    xs,ys = np.nonzero(amp > 30)
    for i in range(len(xs)):
        amp[xs[i],ys[i]] += 100
    
    # normalize
    amp *= 255.0/amp.max()

    plt.imsave(path_to_output+".png", amp,cmap='gray')


if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
