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
            scan = flip(scan, 0)
            for bbox in img_data_aug['bboxes']:
                z = bbox['z']
                bbox['z'] = depth - z - 1

        if cfg.use_y_flips and np.random.randint(0, 3) == 1:
            scan = flip(scan, 1)
            for bbox in img_data_aug['bboxes']:
                y = bbox['y']
                bbox['y'] = height - y - 1

        if cfg.use_x_flips and np.random.randint(0, 3) == 2:
            scan = flip(scan, 2)
            for bbox in img_data_aug['bboxes']:
                x = bbox['x']
                bbox['x'] = width - x - 1

    return img_data_aug, scan


