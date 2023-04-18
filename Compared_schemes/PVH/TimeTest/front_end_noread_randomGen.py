import hashlib
import numpy as np
from moviepy.editor import VideoFileClip
from skimage.transform import resize
import imagehash
from PIL import Image
import cv2
import distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import skimage
import math
import timeit
from time import process_time
import random
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r
def average_hash_self(img):

    preprocess_resize_constants=(8,8)
    img = skimage.transform.resize(np.array(img), preprocess_resize_constants)
    img = skimage.color.rgb2gray(img)
    mean_value=np.mean(img)
    ahash_value=''
    for i in range(preprocess_resize_constants[0]):
        for j in range(preprocess_resize_constants[1]):
            if img[i][j]>mean_value:
                ahash_value=ahash_value+'1'
            else:
                ahash_value = ahash_value + '0'
    return ahash_value


def read_time(path1):
    clip = cv2.VideoCapture(path1)

    imgs = []
    while True:
        success, frame = clip.read()
        if not success:
            break
        imgs.append(frame)
    return []
def baseline1(imgs):


    block_size = 16

    frame_hash_values = []
    sample_num=20
    sampling_lists=[]
    img_i=0
    for f in imgs:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)
        random.seed(img_i)
        sampling_list = []
        for i in range(sample_num):
            sampling_list.append(int(random.random() * row_block_range*col_block_range))
        #sampling_list = np.random.randint(low=0, high=row_block_range*col_block_range, size=(sample_num,))
        sampling_lists.append(sampling_list)

        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]
        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]
            blocks.extend(block)
        cur_frame_hash = imagehash.average_hash(Image.fromarray(np.array(blocks)))
        frame_hash_values.append((cur_frame_hash))
        img_i+=1
def baseline2(imgs):


    block_size = 16


    frame_hash_values = []
    frame_hash_values_forged = []
    sample_num = 20
    sample_insertion = 2
    sampling_lists = []
    img_i = 0

    for f in imgs:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)

        cor_blocks = []
        sampling_list = []
        sampling_list_new = []
        sampling_list = []
        if img_i != 0:
            f2 = imgs[img_i - 1]
            random.seed(img_i+1)
            for i in range(sample_num):
                sampling_list_new.append(int(random.random() * row_block_range * col_block_range))
            sampling_list_row_new = [int(sample_index / col_block_range) for sample_index in sampling_list_new]
            sampling_list_col_new = [int(sample_index % col_block_range) for sample_index in sampling_list_new]
            for i in range(sample_num):
                block_temp = f[sampling_list_row_new[i] * block_size:(sampling_list_row_new[i] + 1) * block_size,
                             sampling_list_col_new[i] * block_size:(sampling_list_col_new[i] + 1) * block_size]

                block_temp2 = f2[sampling_list_row_new[i] * block_size:(sampling_list_row_new[i] + 1) * block_size,
                              sampling_list_col_new[i] * block_size:(sampling_list_col_new[i] + 1) * block_size]
                cor_blocks.append(corr2(block_temp, block_temp2))
            sorted_blocks = sorted(cor_blocks, key=abs)
            random.seed(img_i)
            for i in range(sample_num - sample_insertion):
                sampling_list.append(int(random.random() * row_block_range * col_block_range))
            i = 0
          #  sampling_list.extend(np.random.randint(low=0, high=row_block_range * col_block_range,
                                                 #  size=(sample_num - sample_insertion,)))
            while (len(sampling_list) < sample_num):
                index_temp = sampling_list_new[cor_blocks.index(sorted_blocks[i])]
                cor_blocks[cor_blocks.index(sorted_blocks[i])] = 2

                if index_temp in sampling_list:  # 此方法不好，对于block的corr2都差不多的部分frame而言 会sample0-19的block
                    i += 1
                    continue
                i += 1
                sampling_list.append(index_temp)

        else:
            random.seed(img_i)

            for i in range(sample_num):
                sampling_list.append(int(random.random() * row_block_range * col_block_range))

        img_i += 1
        sampling_lists.append(sampling_list)

        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]

        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]

            blocks.extend(block)

        cur_frame_hash = imagehash.average_hash(Image.fromarray(np.array(blocks)))
        frame_hash_values.append((cur_frame_hash))
def CCSH(imgs):



    block_size = 16


    frame_hash_values = []

    sample_num = 20
    sample_insertion = 2
    sampling_lists = []
    img_i = 0
    last_sampling_list_col = []
    last_sampling_list_row = []
    for f in imgs:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)

        cor_blocks = []
        sampling_list = []
        if img_i != 0:
            f2=imgs[img_i-1]
            for i in range(sample_num):
                block_temp = f[last_sampling_list_row[i] * block_size:(last_sampling_list_row[i] + 1) * block_size,
                             last_sampling_list_col[i] * block_size:(last_sampling_list_col[i] + 1) * block_size]
                block_temp2 = f2[last_sampling_list_row[i] * block_size:(last_sampling_list_row[i] + 1) * block_size,
                              last_sampling_list_col[i] * block_size:(last_sampling_list_col[i] + 1) * block_size]
                cor_blocks.append(corr2(block_temp, block_temp2))
            sorted_blocks = sorted(cor_blocks, key=abs)
            i = 0
            random.seed(img_i)
            for i in range(sample_num - sample_insertion):
                sampling_list.append(int(random.random() * row_block_range * col_block_range))
            #sampling_list.extend(np.random.randint(low=0, high=row_block_range * col_block_range,
                                                #   size=(sample_num - sample_insertion,)))
            while (len(sampling_list) < sample_num):
                index_temp = last_sampling_list[cor_blocks.index(sorted_blocks[i])]
                cor_blocks[cor_blocks.index(sorted_blocks[i])] = 2

                if index_temp in sampling_list:  # 此方法不好，对于block的corr2都差不多的部分frame而言 会sample0-19的block
                    i += 1
                    continue
                i += 1
                sampling_list.append(index_temp)

        else:
            random.seed(img_i)
            for i in range(sample_num):
                sampling_list.append(int(random.random() * row_block_range * col_block_range))


        # print("col",col_block_range,row_block_range)
        img_i += 1
        sampling_lists.append(sampling_list)
        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]
        last_sampling_list_col = sampling_list_col
        last_sampling_list_row = sampling_list_row
        last_sampling_list = sampling_list
        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]

            blocks.extend(block)

        cur_frame_hash = imagehash.average_hash(Image.fromarray(np.array(blocks)))
        frame_hash_values.append((cur_frame_hash))
time_start1 = process_time()
path_1920 = 'C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
clip = cv2.VideoCapture(path_1920)

imgs = []
while True:
    success, frame = clip.read()
    if not success:
        break
    imgs.append(frame)
time_end1=process_time()
print(time_end1-time_start1)
print(len(imgs))
repeatTime=40
time_costs=[]
methods=[baseline1,baseline2,CCSH]
for method in methods:
    time_start = process_time()

    for i in range(repeatTime):
        method(imgs)
    time_end=process_time()

    time_costs.append((time_end-time_start)/(repeatTime*len(imgs)))
print("time costs:",time_costs)
#320*240: [0.0004166666666666667, 0.00140625, 0.0013095238095238095]
#704*480: [0.0004352367688022284, 0.001392757660167131, 0.0013709958217270194]
#1280*720: [0.00046875, 0.0014750744047619048, 0.001333705357142857]
#1920*1080: [0.0004388078193832599, 0.0013525605726872246, 0.0013043777533039648]
#3840*2160: [0.0004388020833333333, 0.0014622395833333334, 0.0013502604166666667]