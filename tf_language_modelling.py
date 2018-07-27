#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:21:40 2018

@author: ekele
"""

import tensorflow as tf
import numpy as np
import time

!mkdir data
!wget -q -O data/ptb.zip https://ibm.box.com/shared/static/z2yvmhbskc45xd2a9a4kkn6hg4g4kj5r.zip
!unzip -o data/ptb.zip -d data
!cp data/ptb/reader.py .
import reader

!wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz 
!tar xzf simple-examples.tgz -C data/

