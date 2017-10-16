#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = "/home/donjuan/Luna16/"
csvfiles_path = file_path + "csvfiles/"
annotations_file = csvfiles_path + "annotations.csv"

annotations = pd.read_csv(annotations_file)

d_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# <5, 5-7, 7-10. 10-13, 13-15, 15-17, 17-20, 20-25, 25-30, >30
n_groups = 10
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4 


for i in range(annotations.shape[0]):
    if annotations.ix[i, 'diameter_mm'] < 5:
        d_count[0] += 1
    elif annotations.ix[i, 'diameter_mm'] >=5 and annotations.ix[i, 'diameter_mm'] < 7:
        d_count[1] += 1
    elif annotations.ix[i, 'diameter_mm'] >=7 and annotations.ix[i, 'diameter_mm'] < 10:
        d_count[2] += 1
    elif annotations.ix[i, 'diameter_mm'] >=10 and annotations.ix[i, 'diameter_mm'] < 13:
        d_count[3] += 1
    elif annotations.ix[i, 'diameter_mm'] >=13 and annotations.ix[i, 'diameter_mm'] < 15:
        d_count[4] += 1
    elif annotations.ix[i, 'diameter_mm'] >=15 and annotations.ix[i, 'diameter_mm'] < 17:
        d_count[5] += 1
    elif annotations.ix[i, 'diameter_mm'] >=17 and annotations.ix[i, 'diameter_mm'] < 20:
        d_count[6] += 1
    elif annotations.ix[i, 'diameter_mm'] >=20 and annotations.ix[i, 'diameter_mm'] < 25:
        d_count[7] += 1
    elif annotations.ix[i, 'diameter_mm'] >=25 and annotations.ix[i, 'diameter_mm'] < 30:
        d_count[8] += 1
    elif annotations.ix[i, 'diameter_mm'] >=30:
        d_count[9] += 1

count_bar = plt.bar(index, d_count, bar_width, alpha=opacity, color='b', label='diameter_count')

plt.xlabel('Range')
plt.ylabel('Count')
plt.title('Count of Nodule diameter')
plt.xticks(index+bar_width/2, ('<5', '5-7', '7-10', '10-13', '13-15', '15-17', '17-20', '20-25', '25-30', '>30'))
plt.legend()

plt.show()



