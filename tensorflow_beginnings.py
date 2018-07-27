#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:17:41 2018

@author: ekele
"""

import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)

session = tf.Session()

result = session.run(c)
print(result)

with tf.Session() as session:
    result = session.run(c)
    print(result)
    
Scalar = tf.constant([3])
Vector = tf.constant([4,3,5])
Matrix = tf.constant([ [1,2,3],[2,3,4],[3,4,5] ])
Tensor = tf.constant([ [[1,2,3],[2,3,4],[3,4,5]], [[4,5,6],[5,6,7],[6,7,8]], [[7,8,9],[8,9,10],[9,10,11]] ])

with tf.Session() as session:
    result = session.run(Scalar)
    print("Scalar (1 entry):\n %s \n" % result)
    result = session.run(Vector)
    print("Vector (3 entries):\n %s \n" % result)
    result = session.run(Matrix)
    print("Matrix (3x3 entries):\n %s \n" % result)
    result = session.run(Tensor)
    print("Tensor (3x3x3 entries): \n %s \n" % result)
    
Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

first_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session() as sess:
    result = sess.run(first_operation)
    print("Defined using tensorflow function :")
    print(result)
    
state = tf.Variable(0)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        
import numpy as np        
x = tf.placeholder(tf.float32, shape=(1024,1024))
y = tf.matmul(x,x)

with tf.Session() as sess:
    rand_array = np.random.rand(1024,1024)
    print(sess.run(y, feed_dict = {x: rand_array}))
    
    
a = tf.placeholder(tf.float32)
b = a*2

with tf.Session() as sess:
    result = sess.run(b, feed_dict={a:3.5})
    print(result)
    
a = tf.constant([5])    
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)

with tf.Session() as sess:
    result = sess.run(c)
    print('c =: %s' % result)
    result = sess.run(d)
    print('d =: %s' % result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    