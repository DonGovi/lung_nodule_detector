#-*-coding:utf-8-*-

from keras import backend as K

class Config:
    def __init__(self):

        self.data_dir = 'E:\\LUNA16\\data\\'
        self.label_dir = 'E:\\LUNA16\\csvfiles\\'
        self.use_z_flips = False
        self.use_y_flips = False
        self.use_x_flips = False

        self.anchor_scales = [8, 15, 26, 40]
        self.rpn_stride = 16

        self.rpn_max_overlap = 0.7
        self.rpn_min_overlap = 0.3

        self.min_bound = -1000.0
        self.max_bound = 600.0