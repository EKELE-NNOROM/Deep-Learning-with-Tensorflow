#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 22:14:55 2018

@author: ekele
"""

import numpy as np

x = [3,4,5]
h = [2,1,0]

y = np.convolve(x,h)
y

# convolution with padding
x = [6,2]
h = [1,2,5,4]

y = np.convolve(x,h,"full")
y

# convolution without padding
x = [6,2]
h = [1,2,5,4]

y = np.convolve(x,h,"valid")
y

from scipy import signal as sg

I = [ [255, 7, 3],
      [212, 240, 4],
      [218, 216, 230],]

g = [[-1,1]]

print('Without zero padding \n')
print( '{0} \n'.format(sg.convolve(I,g,'valid')))

print('With zero padding \n')
print(sg.convolve(I,g))

I = [ [255, 7, 3],
      [212, 240, 4],
      [218, 216, 230],]

g = [[-1,1],
     [2,3],]

print('Without zero padding \n')
print( '{0} \n'.format(sg.convolve(I,g,'full')))

print('With zero padding \n')
print(sg.convolve(I,g,'valid'))


import tensorflow as tf
input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print('Filter/Kernel \n')
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print('Result/Feature Map with padding \n')
    result2 = sess.run(op2)
    print(result2)