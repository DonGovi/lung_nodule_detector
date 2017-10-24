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

data_gen_train = data_generators.get_anchor_gt(train_imgs, C, nn.get_img_output_length, 
                K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, C, nn.get_img_output_length,
                K.image_dim_ordering(), mode='val')


if K.image_dim_ordering() == 'th':
    input_shape_img = (1, None, None, None)
else:
    input_shape_img = (None, None, None, 1)

img_input = Input(shape=input_shape_img)

shared_layers = nn.nn_base(img_input, trainable=True)

num_anchors = len(C.anchor_scales)
rpn = nn.rpn(shared_layers, num_anchors)

model_rpn = Model(img_input, rpn[:2])
