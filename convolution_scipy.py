#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:26:14 2018

@author: ekele
"""

import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open('Canada_day.jpg')

image_gr = im.convert("L")
print('\n Original type: %r \n\n' % image_gr)

arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)


imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

kernel = np.array([ [0,1,0],
                    [1,-4,1],
                    [0,1,0] ])

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature Map')

fig, aux = plt.subplots(figsize=(10,10))
aux.imshow(np.absolute(grad), cmap='gray')

type(grad)

grad_biases = np.abs(grad) + 100

grad_biases[grad_biases > 255] = 255

print('GRADIENT MAGNITUDE - Feature Map')

fig, aux = plt.subplots(figsize=(10,10))
aux.imshow(np.absolute(grad_biases), cmap='gray')