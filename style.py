import numpy as np
import scipy.io
import scipy.misc

VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def load_image(path):
    img = scipy.misc.imread(path).astype(np.float32)
    return img[np.newaxis] - VGG_MEAN


def save_image(path, img):
    scipy.misc.imsave(path, (img + VGG_MEAN)[0])
