#-*-coding:utf-8-*-

import lung_seg as ls
import numpy as np
import copy
import config as cfg

def flip(scan, axis, dim=3):
    aug_scan = np.zeros(scan.shape)

    if axis == 0:
        for i in range(scan.shape[0]):
            aug_scan[i,:,:] = scan[scan.shape[0]-i-1,:,:]

    elif axis == 1:
        for i in range(scan.shape[1]):
            aug_scan[:,i,:] = scan[:,scan.shape[i]-i-1,:]

    elif axis == 2:
        for i in range(scan.shape[2]):
            aug_scan[:,:,i] = scan[:, :, scan.shape-i-1]

    return aug_scan



def augment(img_data, cfg, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data
    assert 'depth' in img_data

    img_data_aug = copy.deepcopy(img_data)

    scan, origin, new_spacing = ls.lung_seg(img_data_aug['filepath'])

    if augment:
        depth, height, width = scan.shape[:3]

        if cfg.use_z_flips and np.random.randint(0, 3) == 0:
