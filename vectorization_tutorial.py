#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:32:42 2018

@author: ekele
"""

import numpy as np
from math import sin as sn
import matplotlib.pyplot as plt
import time

N_point = 1000

def myfunc(x,y):
    if (x>0.5*y and y<0.3):
        return (sn(x-y))
    elif (x<0.5*y):
        return 0
    elif (x>0.2*y):
        return (2*sn(x+2*y))
    else:
        return (sn(y+x))
    
lst_x = np.random.randn(N_point)
lst_y = np.random.randn(N_point)
lst_result = []

tic=time.time()
for i in range(len(lst_x)):
    x = lst_x[i]
    y = lst_y[i]
    if (x>0.5*y and y<0.3):
        lst_result.append(sn(x-y))
    elif (x<0.5*y):
        lst_result.append(0)
    elif (x>0.2*y):
        lst_result.append(2*sn(x+2*y))
    else:
        lst_result.append(sn(y+x))
toc=time.time()

print("\nTime taken by the plain vanilla for-loop\n----------------------------------------------\n{} us".format(1000000*(toc-tic)))

# List comprehension
print("\nTime taken by list comprehension and zip\n"+'-'*40)
%timeit lst_result = [myfunc(x,y) for x,y in zip(lst_x,lst_y)]

# Map() function
print("\nTime taken by map function\n"+'-'*40)
%timeit list(map(myfunc,lst_x,lst_y))

# Numpy.vectorize method
print("\nTime taken by numpy.vectorize method\n"+'-'*40)
vectfunc = np.vectorize(myfunc, otypes=[np.float], cache=False)
%timeit list(vectfunc(lst_x,lst_y))