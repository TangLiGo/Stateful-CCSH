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

def hash_compare(insertion_num,video_path1,video_path2,flag,seq1,seq2):

   # print(path1,path2)
   # print(video_path1,video_path2)
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
    block_size = 16
    height=imgs[0].shape[0]
    width=imgs[0].shape[1]

    row_block_range=int(height/block_size)
    col_block_range=int(width/block_size)
    frame_hash_values = []
    frame_hash_values_forged=[]
    sample_num=20
    sample_insertion=insertion_num
    sampling_lists=[]
    img_i=0
    last_img_blocks=(np.zeros((block_size*sample_num,block_size*sample_num,3)) )
    last_sampling_list_col = []
    last_sampling_list_row = []
    for f in imgs:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)

        cor_blocks=[]
        sampling_list = []
        if img_i != 0:

           # print("last",last_sampling_list_row)
            for i in range(sample_num):
                block_temp=f[last_sampling_list_row[i] * block_size:(last_sampling_list_row[i] + 1) * block_size,
                    last_sampling_list_col[i] * block_size:(last_sampling_list_col[i] + 1) * block_size]
                #print("error",img_i,block_temp)

                #print(last_img_blocks[i*block_size:(i+1)*block_size])
                cor_blocks.append(corr2(block_temp, last_img_blocks[i*block_size:(i+1)*block_size]))
            sorted_blocks = sorted(cor_blocks,key=abs)
            i=0
            sampling_list.extend(np.random.randint(low=0, high=row_block_range * col_block_range,
                                               size=(sample_num - sample_insertion,)))
            while (len(sampling_list) <sample_num):
                index_temp = last_sampling_list[cor_blocks.index(sorted_blocks[i])]
                cor_blocks[cor_blocks.index(sorted_blocks[i])] = 2
                #print(index_temp,sampling_list)
                if index_temp in sampling_list:  # 此方法不好，对于block的corr2都差不多的部分frame而言 会sample0-19的block
                    i += 1
                    continue
                i += 1
                sampling_list.append(index_temp)
           # for i in range(sample_insertion):
               # index_temp = last_sampling_list[cor_blocks.index(sorted_blocks[i])]
               # cor_blocks[cor_blocks.index(sorted_blocks[i])]=2

              #  sampling_list.append(index_temp)
           # print("sampling_list", sampling_list)
           # sampling_list.extend(np.random.randint(low=0, high=row_block_range*col_block_range ,
                         #                   size=(sample_num-sample_insertion,)))
           # print("sampling_list", sampling_list)
        else:
            sampling_list = np.random.randint(low=0, high=row_block_range*col_block_range, size=(sample_num,))

       # print("col",col_block_range,row_block_range)
        img_i+=1
        sampling_lists.append(sampling_list)

        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]
       # print("cur",sampling_list_row)
        last_sampling_list_col = sampling_list_col
        last_sampling_list_row=sampling_list_row
        last_sampling_list=sampling_list
      #  print("sampling_list_row",sampling_list_row)
       # print("sampling_list_col",sampling_list_col)
        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]

            blocks.extend(block)
        last_img_blocks = blocks

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
    plt.figure()
    plt.plot(forged_frame_indices)
    img_name="../result_sample/"+str(flag)+"_"+str(seq1)+"_"+str(seq2)+".png"
    plt.savefig(img_name)
    return forged_frame_indices
def compare_hd(insertion_num,seq):
    data_path0 = 'C:/Users/tangli/Desktop/dataset/dataset4_revised/Lossless'
    data_path1 = 'C:/Users/tangli/Desktop/dataset/dataset4_revised/Lossy1'
    data_path2 = 'C:/Users/tangli/Desktop/dataset/dataset4_revised/Lossy2'
    data_path3 = 'C:/Users/tangli/Desktop/dataset/dataset4_revised/Lossy3'
    data_path=[data_path0,data_path1,data_path2,data_path3]
    original_videos=[[],[],[],[]]
    forged_videos=[[],[],[],[]]
    for i in range(len(data_path)):
        path=data_path[i]
        for file in os.listdir(path):
            video_path = os.path.join(path, file)
            video_path = video_path.replace('\\', '/')
            if 'Real' in file:
                original_videos[i].append(video_path)
            elif 'Forged' in file:
                forged_videos[i].append(video_path)
    print("info", original_videos, forged_videos)
    hds_forgery=[]
    for i in range(len(original_videos)):
        for j in range(len(forged_videos)):
            sub_hds_forgery = []
            for k in range(len(original_videos[i])):

                sub_hds_forgery.append(hash_compare(insertion_num, original_videos[i][k], forged_videos[j][k],0,i+j,k))
            hds_forgery.append(sub_hds_forgery)

    hds_compression=[]
    for i in range(len(original_videos)-1):
        for j in range(i+1,len(original_videos)):
            sub_hds_compression = []
            for k in range(len(original_videos[i])):

                sub_hds_compression.append(hash_compare(insertion_num, original_videos[i][k], original_videos[j][k],1,i+j,k))
            hds_compression.append(sub_hds_compression)
    for i in range(len(forged_videos)-1):
        for j in range(i+1,len(forged_videos)):
            sub_hds_compression = []
            for k in range(len(forged_videos[i])):

                sub_hds_compression.append(hash_compare(insertion_num,forged_videos[i][k], forged_videos[j][k],2,i+j,k))
            hds_compression.append(sub_hds_compression)
   # print("hds_forgery=",hds_forgery)
   # print("hds_compression=",hds_compression)
    return hds_forgery, hds_compression


insertion_nums=[0]#[5078.234375, 5073.171875, 5081.953125, 5100.796875, 5290.890625]
repeatTime=1
time_costs=[]
for insertion_num in insertion_nums:
    time_start = process_time()
    hds_forgery_mul = []
    hds_compression_mul = []
    for i in range(repeatTime):
        temp_forgery, temp_compression = compare_hd(insertion_num,1)
        hds_forgery_mul.extend(temp_forgery)
        hds_compression_mul.extend(temp_compression)


