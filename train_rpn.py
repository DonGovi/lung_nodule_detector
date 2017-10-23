#-*-coding:utf-8-*-

from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
import config, data_generators
import losses as losses
import label_transform as lt
from keras.utils import generic_utils
import resnet as nn


C = config.Config()

C.use_z_flips = bool(options.use_z_flips)
C.use_y_flips = bool(options.use_y_flips)
C.use_x_flips = bool(options.use_x_flips)

all_imgs = get_data(C)

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen