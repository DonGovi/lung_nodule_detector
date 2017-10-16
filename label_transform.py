#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import lung_seg as ls
import config
from random import sample
import os

'''
def trans_label():

    
    Transform label files' world coordinates into voxel coordinates,
    and save all nodule information in a dataframe
    
    cfg = config.Config()

    label_df = pd.read_csv(cfg.label_dir+"annotations.csv")
    #read annotations file as a pandas dataframe
    trans_label_df = label_df.copy()
    #make a copy of original label, save dataframe transformed

    for i in label_df.index:
        world_x = label_df.ix[i, 'coordX']
        world_y = label_df.ix[i, 'coordY']
        world_z = label_df.ix[i, 'coordZ']

        if i == 0 or label_df.ix[i, 'seriesuid'] != label_df.ix[i-1, 'seriesuid']:
            img_array, origin, old_spacing = ls.load_scan(cfg.data_dir+label_df.ix[i, 'seriesuid']+'.mhd')
            resampled_img, new_spacing = ls.resample(img_array, old_spacing)
            # z, y, x

        w_coord = np.array([world_z, world_y, world_x])
        v_coord = np.rint((w_coord - origin) / new_spacing)
        v_coord = np.array(v_coord, dtype=int)

        trans_label_df.ix[i, 'coordZ'] = v_coord[0]
        trans_label_df.ix[i, 'coordY'] = v_coord[1]
        trans_label_df.ix[i, 'coordX'] = v_coord[2]

    return trans_label_df
'''

def trans_coord(i, origin, new_spacing, this_df):
    world_z = this_df.ix[i, 'coordZ']
    world_y = this_df.ix[i, 'coordY']
    world_x = this_df.ix[i, 'coordX']
    world_coord = np.array([world_z, world_y, world_x])
    voxel_coord = np.rint((world_coord - origin) / new_spacing)
    voxel_coord = np.array(voxel_coord, dtype=int)
    return voxel_coord



def select_set(label_df):
    '''
    random choose some case as test set
    dataset=1    test
    dataset=0    trainval
    '''
    label_df['dataset'] = np.zeros(label_df.shape[0])
    file_list = label_df['seriesuid'].drop_duplicates().tolist()
    test_list = sample(file_list, np.int(len(file_list)/8))
    label_df.loc[label_df['seriesuid'].isin(test_list), 'dataset'] = 1

    return label_df, file_list, test_list



def get_data():

    # save all nodule annotations in a dictionary
    cfg = config.Config()

    all_imgs = []

    label_df = pd.read_csv(cfg.label_dir+"annotations.csv")
    annot_df, seriesuid_list, test_list = select_set(label_df)


    for seriesuid in seriesuid_list:
        filename = seriesuid + '.mhd'
        file_path = cfg.data_dir + filename
        print("load %s" % filename)
        scan, origin, new_spacing = ls.lung_seg(file_path)           # z,y,x
        scan_depth = scan.shape[0]
        scan_height = scan.shape[1]
        scan_width = scan.shape[2]
        annotations_data = {'filepath':file_path, 'depth':scan_depth, 'height':scan_height, 
                            'width':scan_width, 'bboxes':[]}
        if seriesuid in test_list:
            annotations_data['imageset'] = 'test'
        else:
            annotations_data['imageset'] = 'trainval'

        this_df = annot_df[annot_df['seriesuid']==seriesuid]

        for i in this_df.index:
            voxel_coord = trans_coord(i, origin, new_spacing, this_df)
            z = voxel_coord[0]
            y = voxel_coord[1]
            x = voxel_coord[2]
            diam = this_df.loc[i, 'diameter_mm']
            annotations_data['bboxes'].append({'diam':diam, 
                            'z':z, 'y':y, 'x':x})
        print(annotations_data)
        all_imgs.append(annotations_data)

    return all_imgs


if __name__ == '__main__':

     all_imgs = get_data()
     





