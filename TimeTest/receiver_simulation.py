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


def read_time(path1):
    clip = cv2.VideoCapture(path1)

    imgs = []
    while True:
        success, frame = clip.read()
        if not success:
            break
        imgs.append(frame)
    return []
def baseline1(imgs,hashes_sender,sampling_lists_sender):


    block_size = 16

    frame_hash_values = []
    sample_num=20
    sampling_lists=[]
    for f in imgs:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)
        sampling_list = np.random.randint(low=0, high=row_block_range*col_block_range, size=(sample_num,))
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

def baseline2(imgs,hashes_sender,sampling_lists_sender):


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
        if img_i != 0:
            sampling_list_new = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num-sample_insertion,))
            sampling_list=sampling_lists_sender[img_i]

        else:
            sampling_list = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num,))

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

def CCSH(imgs,hashes_sender,sampling_lists_sender):



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

            sampling_list_new = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num-sample_insertion,))
            sampling_list=sampling_lists_sender[img_i]

        else:
            sampling_list = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num,))

        # print("col",col_block_range,row_block_range)
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

path,hashes_CCSH,sampling_lists_CCSH=np.load('data/CCSH.npy', allow_pickle=True)
path,hashes_baseline2,sampling_lists_baseline2=np.load('data/baseline2.npy', allow_pickle=True)
path,hashes_baseline1,sampling_lists_baseline1=np.load('data/baseline1.npy', allow_pickle=True)
#path_1920 = 'C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
clip = cv2.VideoCapture(path)
imgs = []
while True:
    success, frame = clip.read()
    if not success:
        break
    imgs.append(frame)
print(len(imgs))
repeatTime=100
time_costs=[]
methods=[baseline1,baseline2,CCSH]
hashes=[hashes_baseline1,hashes_baseline2,hashes_CCSH]
sampling_lists=[sampling_lists_baseline1,sampling_lists_baseline2,sampling_lists_CCSH]
for j in range(len(methods)):
    method=methods[j]
    hashes_sender=hashes[j]
    sampling_lists_sender=sampling_lists[j]
    time_start = process_time()
    for i in range(repeatTime):
        method(imgs,hashes_sender,sampling_lists_sender)
    time_end=process_time()
    time_costs.append((time_end-time_start)/(repeatTime*len(imgs)))
print(time_costs)
#320*240: [0.0004211309523809524, 0.00041220238095238094, 0.0004166666666666667]
#704*480: [0.0004299321503131524, 0.0004224295407098121, 0.0004221033402922756]
#1280*720: [0.0004380580357142857, 0.00042410714285714285, 0.00043675595238095236]
#1920*1080:[0.0004408727973568282, 0.00043054790748898677, 0.00043467786343612334]
#3840*2160:[0.0004328125, 0.0004234375, 0.00043802083333333334]