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
import sys
import random
from PIL import Image, ImageFilter
from skimage import io, color,filters
import numpy as np
import tensorly as tl
import binascii
import PIL
from imagehash import ImageHash
from tensorly.decomposition import tucker
from PIL import Image
import pywt
from itertools import combinations
from sympy import *
def preprocessing2(images, size=(256, 256),r_len=256):
    img_array=[]
    starting = True
    prev_frame = np.uint8([256])
    v_len=len(images)
    r_rate=v_len/r_len
    a=int(v_len/r_len)
    b=math.ceil(v_len/r_len)
    y=
    for i in range(v_len):
        frame=images[i]
        lu = Image.fromarray(frame).convert('L')
        img_array.append(resize(np.array(lu), size, anti_aliasing=True))
        if starting == True:
            prev_frame = frame
            starting = False
        else:
            for j in range(1, 10):
                weight = i / 10
                # get the blended frames in between
                mid_frame = cv2.addWeighted(prev_frame, weight, frame, 1 - weight, 0)
            prev_frame = frame
    return img_array



def low_rank_calculation(imgs,N,r):
    #print("he",len(imgs),imgs)
    L=len(imgs)//N
   # print("L",L)
    Rs=[]
    for i in range(L):
        group=np.array(imgs[i*N:(i+1)*N])
     #   print("group",group)
        #SVD
        u, s, vh = np.linalg.svd(group, full_matrices=False)
        u_,s_,vh_ = u[:, :, :r],s[:, None, :r],vh[:, :r, :]
        # print("before",(np.matmul(u * s[:, None, :], vh)).shape, np.matmul(u * s[:, None, :], vh))
        # print("after",(np.matmul(u_ * s_, vh_)).shape, np.matmul(u_ * s_, vh_))
        # print(s.shape, u.shape, vh.shape)
        # print(s_.shape,u_.shape,vh_.shape)
        # print("before",(np.matmul(u * s[:, None, :], vh)).shape,)
        # print("after",(np.matmul(u_ * s_, vh_)).shape)
        Gr=np.matmul(u_ * s_, vh_)#ur.dot(sr).dot(vhr)

        # group2 = np.array(imgs[i * N:(i + 1) * N]).T
        # u2, s2, vh2 = np.linalg.svd(group2, full_matrices=False)
        # u2_, s2_, vh2_ = u2[:, :, :r], s2[:, None, :r], vh2[:, :r, :]
        # Gr2 = np.matmul(u2_ * s2_, vh2_)  # ur.dot(sr).dot(vhr)
        # R2 = np.average(Gr2, axis=2)
        # print("S",Gr2.shape,R2.shape,R2)
        R=np.average(Gr,axis=0)
        # print(Gr)
      #  print("R",R.shape,)
        Rs.append(R)
    return Rs
def low_rank_compression(Rs):
    hash=[]
    for R in Rs:
        D,highlevel=pywt.dwt2(R,'haar')
      #  print(len(D),D.shape,D)
       # print(np.max(D),np.min(D))
        u=np.average(D)
        b=round(u-1)
        hash.append(b)
   # print("hahs",hash)
    return hash


def tensor_decomposition(
        l_matrix,
        u_block_size=2,
        q_block_size=32,
        i_core=1,
        j_core=1,
        k_core=1,
        random_state=1234,
):

    l_matrix_shape = l_matrix.shape[0]
    u_matrix = (
        l_matrix.reshape((l_matrix_shape, l_matrix_shape))
            .reshape(
            l_matrix_shape // u_block_size,
            u_block_size,
            l_matrix_shape // u_block_size,
            u_block_size,
        )
            .mean(axis=(1, 3))
    )
    n = (l_matrix_shape // u_block_size // q_block_size) ** 2
    block_matrix = (
        u_matrix.reshape(
            u_matrix.shape[0] // q_block_size, q_block_size, -1, q_block_size
        )
            .swapaxes(1, 2)
            .reshape(-1, q_block_size, q_block_size)
    )
    x = tl.tensor(block_matrix, dtype="float")
    return tucker(x, rank=[i_core, j_core, k_core], random_state=random_state)


def make_hash(a_factor, b_factor, c_factor):
    a_p = a_factor.mean(axis=1)
    a_h = (a_p >= a_p.mean()).astype(int)
    b_p = b_factor.mean(axis=1)
    b_h = (b_p >= b_p.mean()).astype(int)
    c_p = c_factor.mean(axis=1)
    c_h = (c_p >= c_p.mean()).astype(int)
    return np.hstack((a_h, b_h, c_h))


def gen_hash(video_path):
    N,r=16,2
    frame_samples = []
    clip = cv2.VideoCapture(video_path)
    frame_id = 0
    hashes=[]
    frames=[]
    while True:
        success, frame = clip.read()
        if not success:
            break

        frames.append(frame)
        frame_id += 1

    imgs = preprocessing(frames)
    Rs=low_rank_calculation(imgs,N,r)
    low_rank_compression(Rs)
    return hashes
def traverseVideo(dataset=4):
    data_d1_path0 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless'
    data_d1_path1 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossy_q10'
    data_d1_path2 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossy_q20'
    data_d1_path3 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossy_q30'

    data_d4_path0 = 'C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless'
    data_d4_path1 = 'C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossy1'
    data_d4_path2 = 'C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossy2'
    data_d4_path3 = 'C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossy3'
    if dataset==1:
        data_path = [data_d1_path0, data_d1_path1, data_d1_path2, data_d1_path3]
    elif dataset==4:
        data_path = [data_d4_path0, data_d4_path1, data_d4_path2, data_d4_path3]

    original_videos=[[],[],[],[]]
    forged_videos=[[],[],[],[]]
    for i in range(len(data_path)):
        path=data_path[i]
        for file in os.listdir(path):
            video_path = os.path.join(path, file)
            video_path = video_path.replace('\\', '/')
            if '.mp4' in file or '.avi' in file:
                p1 = os.path.basename(video_path)
                file_name = os.path.splitext(p1)[0]
                db_path = path + '/' + file_name + '_compared_LRH_relength.npy'

                # if os.path.exists(db_path):
                #     print("exist for video", video_path)
                #     continue
                print("gen for ", video_path)
                TD=gen_hash(video_path)
                np.save(db_path, TD)
    return

traverseVideo(dataset=4)

