import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

LAYERS = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
]


def load_image(path):
    img = scipy.misc.imread(path).astype(np.float32)
    return img[np.newaxis] - VGG_MEAN


def save_image(path, img):
    scipy.misc.imsave(path, (img + VGG_MEAN)[0])


def build_vgg(shape, weights_path='imagenet-vgg-verydeep-19.mat'):
    data = scipy.io.loadmat(weights_path)
    layers = data['layers'][0]
    net = {'input': tf.Variable(np.zeros(shape, dtype=np.float32))}
    current = net['input']
    for i, name in enumerate(LAYERS):
        kind = name[:4]
        if kind == 'conv':
            w, b = layers[i][0][0][2][0]
            # matconvnet stores kernels as [width, height, in, out]
            w = np.transpose(w, (1, 0, 2, 3))
            current = tf.nn.conv2d(current, tf.constant(w), strides=[1, 1, 1, 1], padding='SAME')
            current = tf.nn.bias_add(current, tf.constant(b.reshape(-1)))
        elif kind == 'relu':
            current = tf.nn.relu(current)
        else:
            current = tf.nn.avg_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net[name] = current
    return net
