#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import random
import lung_seg as ls
import label_transform as lf
import data_augment as da

def union(au, bu, area_intersection):
    '''
    0: diam
    1: z
    2: y
    3: x
    '''
    area_a = (au[1] + au[0]/2) * (au[2] + au[0]/2) * (au[3] + au[0]/2)
    area_b = (bu[1] + bu[0]/2) * (bu[2] + bu[0]/2) * (bu[3] + bu[0]/2)
    area_union = area_a + area_b - area_intersection

    return area_union


def intersection(ai, bi):
    z = max(ai[1]-ai[0]/2, bi[1]-bi[0]/2)
    y = max(ai[2]-ai[0]/2, bi[2]-bi[0]/2)
    x = max(ai[3]-ai[0]/2, bi[3]-bi[0]/2)

    d = min(ai[1]+ai[0]/2, bi[1]+bi[0]/2) - z
    h = min(ai[2]+ai[0]/2, bi[2]+bi[0]/2) - y
    w = min(ai[3]+ai[0]/2, bi[3]+bi[0]/2) - x
    if d < 0 or y < 0 or x < 0:
        return 0

    return d*h*w


def iou(a, b):

    area_i = intersection(a, b)
    area_union = union(a, b, area_i)

    return float(area_i) / float(area_union)


def calc_rpn(cfg, img_data, depth, height, width, img_size_calc_function):

    downscale = float(cfg.rpn_stride)
    anchor_size = cfg.anchor_scales
    num_anchors = len(anchor_scales)

    (output_depth, output_height, output_width) = img_size_calc_function(depth, height, width)

    y_rpn_overlap = np.zeros((output_depth, output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_depth, output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_depth, output_height, output_width, num_anchors*4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and the coordnates were already transfromed to voxel space
        gta[bbox_num, 0] = bbox['diam']
        gta[bbox_num, 1] = bbox['z']
        gta[bbox_num, 2] = bbox['y']
        gta[bbox_num, 3] = bbox['x']

    for anchor_size_index in range(len(anchor_size)):
        anchor_edge = anchor_size[anchor_size_index]

        for iz in range(output_depth):
            z_anc = downscale * (iz + 0.5)
            z1_anc = z_anc - anchor_edge / 2
            z2_anc = z_anc + anchor_edge / 2

            if z1_anc < 0 or z2_anc > depth:
                continue

            for jy in range(output_height):
                y_anc = downscale * (jy + 0.5) 
                y1_anc = y_anc - anchor_edge / 2
                y2_anc = y_anc + anchor_edge / 2

                if y1_anc < 0 or y2_anc > height:
                    continue

                for kx in range(output_width):
                    x_anc = downscale * (kx + 0.5)
                    x1_anc = x_anc - anchor_edge / 2
                    x2_anc = x_anc + anchor_edge / 2

                    if x1_anc < 0 or x2_anc > width:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the coord (z,y,x) and the current anchor
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], 
                                        [anchor_edge, z_anc, y_anc, x_anc])

                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > cfg.rpn_max_overlap:
                            td = np.log(gta[bbox_num, 0] / anchor_edge)
                            tz = (gta[bbox_num, 1] - z_anc) / anchor_edge
                            ty = (gta[bbox_num, 2] - y_anc) / anchor_edge
                            tx = (gta[bbox_num, 3] - x_anc) / anchor_edge

                        if curr_iou > best_iou_for_bbox[bbox_num]:
                            best_anchor_for_bbox[bbox_num] = [iz, jy, kx, anchor_size_index]
                            best_iou_for_bbox[bbox_num] = curr_iou
                            best_x_for_bbox[bbox_num,:] = [z_anc, y_anc, x_anc, anchor_edge]
                            best_dx_for_bbox[bbox_num,:] = [td, tz, ty, tx]

                        if curr_iou > cfg.rpn_max_overlap:
                            bbox_type = 'pos'
                            num_anchors_for_bbox[bbox_num] += 1

                            if curr_iou > best_iou_for_loc:
                                best_iou_for_loc = curr_iou
                                best_regr = (td, tz, ty, tx)

                        if cfg.rpn_min_overlap < curr_iou < cfg.rpn_max_overlap:
                            if bbox_type != 'pos':
                                bbox_type = 'neutral'

                    if bbox_type == 'neg':
                        y_is_box_valid[iz, jy, kx, anchor_size_index] = 1
                        y_rpn_overlap[iz, jy, kx, anchor_size_index] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[iz, jy, kx, anchor_size_index] = 0
                        y_rpn_overlap[iz, jy, kx, anchor_size_index] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[iz, jy, kx, anchor_size_index] = 1
                        y_rpn_overlap[iz, jy, kx, anchor_size_index] = 1
                        start = 4 * anchor_size_index
                        y_rpn_regr[iz, jy, kx, start:start+4] = best_regr

    # ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                best_anchor_for_bbox[idx, 2], best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                best_anchor_for_bbox[idx, 2], best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * best_anchor_for_bbox[idx, 3]
            y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                best_anchor_for_bbox[idx, 2], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (3, 0, 1, 2))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (3, 0, 1, 2))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (3, 0, 1, 2))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:,:]==1, y_is_box_valid[0,:,:,:,:]==1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:,:]==0, y_is_box_valid[0,:,:,:,:]==1))

    num_pos = len(pos_locs[0])

    num_regions = 50

    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs],
                    pos_locs[3][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs],
                    neg_locs[3][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


def get_anchor_gt(all_img_data, cfg, img_size_calc_function, backend, mode='train'):

    if mode == 'train':
        np.random.shuffle(all_img_data)

    for img_data in all_img_data:
        if mode == 'train':
            img_data_aug, x_img = da.augment(img_data, cfg, augment=True)
        else:
            img_data_aug, x_img = da.augment(img_data, cfg, augment=False)

        (depth, height, width) = (img_data_aug['depth'], img_data_aug['height'], img_data_aug['width'])
        (deps, rows, cols) = x_img.shape

        assert deps == depth
        assert rows == height
        assert cols == width

        try:
            y_rpn_cls, y_rpn_regr = calc_rpn(cfg, img_data_aug, depth, height, width, img_size_calc_function)
        except:
            continue

        # zero-center by mean voxel, and preprocess scan
        x_img = x_img.astype('float32')
        x_img = (x_img - cfg.min_bound) / (cfg.max_bound - cfg.max_bound)
        x_img = np.clip(x_img, 0, 1)

        x_img = np.expand_dims(x_img, axis=0)
        # expand channel dim
        x_img = np.expand_dims(x_img, axis=0)
        # expand batch dims

        if backend == 'tf':
            x_img = np.transpose(x_img, (0,2,3,4,1))
            y_rpn_cls = np.transpose(y_rpn_cls, (0,2,3,4,1))
            y_rpn_regr = np.transpose(y_rpn_regr, (0,2,3,4,1))

        yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug























