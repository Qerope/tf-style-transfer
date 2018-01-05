import numpy as np
import scipy.io
import scipy.misc
import argparse
import tensorflow as tf

VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

LAYERS = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
]
CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


def load_image(path):
    img = scipy.misc.imread(path).astype(np.float32)
    return img[np.newaxis] - VGG_MEAN


def load_style(path, shape):
    img = scipy.misc.imread(path, mode='RGB')
    img = scipy.misc.imresize(img, shape[1:3]).astype(np.float32)
    return img[np.newaxis] - VGG_MEAN


def save_image(path, img):
    out = np.clip((img + VGG_MEAN)[0], 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, out)


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


def gram(x):
    _, h, w, c = x.get_shape().as_list()
    f = tf.reshape(x, (h * w, c))
    return tf.matmul(f, f, transpose_a=True) / (h * w * c)


def total_variation(x):
    dh = x[:, 1:, :, :] - x[:, :-1, :, :]
    dw = x[:, :, 1:, :] - x[:, :, :-1, :]
    return tf.reduce_sum(tf.square(dh)) + tf.reduce_sum(tf.square(dw))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--content', required=True)
    p.add_argument('--style', required=True)
    p.add_argument('--out', default='out/result.png')
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--content-weight', type=float, default=5.0)
    p.add_argument('--style-weight', type=float, default=100.0)
    p.add_argument('--tv-weight', type=float, default=1e-3)
    p.add_argument('--noise-init', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    content = load_image(args.content)
    style = load_style(args.style, content.shape)
    shape = content.shape

    with tf.Session() as sess:
        net = build_vgg(shape)
        sess.run(net['input'].assign(content))
        content_target = sess.run(net[CONTENT_LAYER])
        sess.run(net['input'].assign(style))
        style_targets = {l: sess.run(gram(net[l])) for l in STYLE_LAYERS}

        content_loss = tf.reduce_mean(tf.square(net[CONTENT_LAYER] - content_target))
        style_loss = sum(tf.reduce_mean(tf.square(gram(net[l]) - style_targets[l])) for l in STYLE_LAYERS)
        style_loss /= len(STYLE_LAYERS)
        loss = (args.content_weight * content_loss + args.style_weight * style_loss
                + args.tv_weight * total_variation(net['input']))
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B', options={'maxiter': args.iterations})
        if args.noise_init:
            init = np.random.uniform(-20, 20, shape).astype(np.float32)
        else:
            init = content
        sess.run(net['input'].assign(init))
        optimizer.minimize(sess, loss_callback=lambda l: print(l), fetches=[loss])
        save_image(args.out, sess.run(net['input']))


if __name__ == '__main__':
    main()
