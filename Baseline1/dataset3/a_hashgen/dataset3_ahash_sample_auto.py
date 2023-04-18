import hashlib

import numpy as np
import pyximport
from moviepy.editor import VideoFileClip

import distance
import cv2
import skimage
from skimage.morphology import disk
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import imagehash
from PIL import Image
import timeit
from time import process_time





def hash_compare(sample_num,video_path1,video_path2):
    clip = cv2.VideoCapture(video_path1)
    clip_forged = cv2.VideoCapture(video_path2)
    imgs = []
    imgs_forged = []
    while True:
        success, frame = clip.read()
        if not success:
            break
        imgs.append(frame)
    while True:
        success, frame = clip_forged.read()
        if not success:
            break
        imgs_forged.append(frame)


    frame_hash_values = []
    frame_hash_values_forged=[]
    changed_index=[]
    block_size=16

    sampling_lists=[]
    for f in imgs:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)
        # print(f.shape[0],f.shape[1])#240,320
        # print(row_block_range,col_block_range)
        sampling_list = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num,))
        sampling_lists.append(sampling_list)
        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]
        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]
            # median_value=np.median(block)
            blocks.extend(block)

        cur_frame_hash = imagehash.average_hash(Image.fromarray(np.array(blocks)))
        frame_hash_values.append((cur_frame_hash))
    f_index = 0
    for f in imgs_forged:
        if f_index>=len(imgs):
            break
        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_lists[f_index]]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_lists[f_index]]
        f_index += 1
        blocks_forged = []
        for i in range(len(sampling_list_row)):
            block_forged = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                           sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]
            # median_value=np.median(block)
            blocks_forged.extend(block_forged)
        cur_frame_hash = imagehash.average_hash(Image.fromarray(np.array(blocks_forged)))
        frame_hash_values_forged.append((cur_frame_hash))




    forged_frame_indices=[]
    for i in range(min(len(frame_hash_values),len(frame_hash_values_forged))):
        #print(frame_hash_values[i])
        hd = frame_hash_values[i]-frame_hash_values_forged[i]
        forged_frame_indices.append(hd)

    return forged_frame_indices



def compare_hd(sample_num):
    data_path = 'C:/Users/tangli/Desktop/dataset/dataset3/used'
    original_videos=[[],[]]
    forged_videos=[[],[]]

    for file in os.listdir(data_path):
        video_path = os.path.join(data_path, file)
        video_path = video_path.replace('\\', '/')
        if 'compression' in file:
            if 'Real' in file:
                original_videos[1].append(video_path)
            elif 'Forgery' in file:
                forged_videos[1].append(video_path)
        else:
            if 'Real' in file:
                original_videos[0].append(video_path)
            elif 'Forgery' in file:
                forged_videos[0].append(video_path)

    hds_forgery=[]
    for i in range(len(original_videos)):
        for j in range(len(forged_videos)):
            sub_hds_forgery = []
            for k in range(len(original_videos[i])):

                sub_hds_forgery.append(hash_compare(sample_num, original_videos[i][k], forged_videos[j][k]))
            hds_forgery.append(sub_hds_forgery)

    hds_compression=[]
    for i in range(len(original_videos)-1):
        for j in range(i+1,len(original_videos)):
            sub_hds_compression = []
            for k in range(len(original_videos[i])):

                sub_hds_compression.append(hash_compare(sample_num, original_videos[i][k], original_videos[j][k]))
            hds_compression.append(sub_hds_compression)
    for i in range(len(forged_videos)-1):
        for j in range(i+1,len(forged_videos)):
            sub_hds_compression = []
            for k in range(len(forged_videos[i])):

                sub_hds_compression.append(hash_compare(sample_num, forged_videos[i][k], forged_videos[j][k]))
            hds_compression.append(sub_hds_compression)
    print("The hd info for forgery detection:",hds_forgery)
    print("The hd info for compression",hds_compression)


    return hds_forgery,hds_compression

#sample_nums=[140,160,180,]
sample_nums=[25,30,80,85,90,95,100]# time:[3126.046875] #[120,130,170,190,200] #time [2613.640625, 2834.34375, 3660.453125, 4109.71875, 4359.25]
print(list(sample_nums))
time_costs=[]
repeatTime=2
for sample_num in sample_nums:
    time_start = process_time()
    hds_forgery_mul = []
    hds_compression_mul = []
    for i in range(repeatTime):
        temp_forgery, temp_compression = compare_hd(sample_num)
        hds_forgery_mul.extend(temp_forgery)
        hds_compression_mul.extend(temp_compression)
        hds_forgery_arr = np.array(hds_forgery_mul, dtype=object)
        np.save('../data/dataset3/hds_ahash_sample_num_0'+str(sample_num)+'.npy', hds_forgery_arr)
        hds_compression_arr = np.array(hds_compression_mul, dtype=object)
        np.save('../data/dataset3/hds_ahash_sample_num_0'+str(sample_num)+'_compression.npy', hds_compression_arr)
    time_end=process_time()
    time_costs.append(time_end-time_start)
print("time costs:",time_costs)
print("finish")

#[15,17,19,21,23,25,27,29] [88.296875, 91.421875, 94.171875, 96.609375, 99.4375, 103.203125, 107.21875, 109.921875]
#[5,10,15,20,25,30,35,40,45,50] [349.890625, 381.03125, 423.890625, 468.96875, 522.640625, 578.765625, 625.734375, 668.78125, 708.5625, 760.015625]
#[55,60,65,70,75,80] [771.671875, 804.8125, 845.296875, 888.609375, 929.921875, 977.453125]