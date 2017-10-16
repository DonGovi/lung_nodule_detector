#-*-coding:utf-8-*-

from keras import backend as K

class Config:
    def __init__(self):

        self.data_dir = 'E:\\LUNA16\\data\\'
        self.label_dir = 'E:\\LUNA16\\csvfiles\\'
        self.use_z_flips = False
        self.use_y_flips = False
        self.use_x_flips = False
        self.rot_90 = False