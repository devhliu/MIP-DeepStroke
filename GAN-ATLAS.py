import tensorflow as tf
import numpy as np
import cv2
import random
import os
import scipy.misc

slim = tf.contrib.slim

HEIGHT, WIDTH, DEPTH, CHANNELS = 128, 128, 128, 3
BATCH_SIZE = 8
EPOCH = 5000

os.environ['CUDA_VISIBLE_DEVICES'] = 15
version = 'ATLAS'
path = version


def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)


def process_data():
    current_dir = os.getcwd()
    atlas_dir = os.path.join(current_dir)
    image_paths = []
    for path in os.listdir(atlas_dir):
        image_paths.append(os.path.join(atlas_dir, path))

    all_images = tf.convert_to_tensor(image_paths, dtype=tf.string)  # TODO WHY string?
    images_queue = tf.train.slice_input_producer([all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels=CHANNELS)

    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=0.1)
    #image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    size = (HEIGHT, WIDTH, DEPTH)
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT, WIDTH, DEPTH, CHANNELS])

    image = tf.cast(image, tf.float32)
    image = image/255.0

    # TODO Verify this function
    images_batch = tf.train.shuffle_batch([image], batch_size=BATCH_SIZE,
                                          num_threads=4, capacity=200 + 3 *BATCH_SIZE, min_after_dequeue=200)

    num_images = len(image_paths)

    return images_batch, num_images


def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32
    s4 = 4
    output_dim = CHANNELS
    with tf.VariableScope('gen') as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w1', shape=[random_dim, s4*s4*c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape[c4*s4*s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        # Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')

        # Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv3d_transpose(act1, c8, kernel_size=[5, 5, 5], strides=[2, 2, 2], padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')

        # TODO Continue development





