#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:38:26 2018

@author: ekele
"""

import tensorflow as tf

sess.close()

sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

width = 28
height = 28
flat = width * height
class_output = 10

x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

x_image = tf.reshape(x, [-1,28,28,1])
x_image

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1

h_conv1 = tf.nn.relu(convolve1)

conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv1

W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2

h_conv2 = tf.nn.relu(convolve2)

conv2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv2

layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1

h_fc1 = tf.nn.relu(fcl)
h_fc1

keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

fc = tf.matmul(layer_drop, W_fc2) + b_fc2

y_CNN = tf.nn.softmax(fc)
y_CNN


import numpy as np
layer4_test = [ [0.9, 0.1, 0.1], [0.9, 0.1, 0.1] ]
y_test = [ [1.0, 0.0, 0.0], [1.0, 0.0, 0.0] ]
np.mean(-np.sum(y_test * np.log(layer4_test), 1))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), axis=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_CNN,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, float(train_accuracy)))
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    
print("test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2,3,0,1]),[32,-1]))

!wget --output-document utils1.py http://deeplearning.net/tutorial/code/utils.py
import utils1
from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image

image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5), tile_shape=(4, 8), tile_spacing=(1, 1)))
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage, [28,28]), cmap='gray')

ActivatedUnits = sess.run(convolve1, feed_dict={x:np.reshape(sampleimage, [1, 784], order='F'), keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation='nearest', cmap='gray')
    
sess.close()