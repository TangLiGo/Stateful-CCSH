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
from thash.tucker_hash import tucker_hash
from itertools import combinations
def preprocessing_image_array(img_array, size=(256, 256)):
    img_array = resize(img_array, size, anti_aliasing=True)
    L = rgb2lab(img_array)[:, :, 0]
    GaussKernel = (
        np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ]
        )
        / 256
    )
    blured = convolve2d(
        L, GaussKernel, boundary="fill", mode="same", fillvalue=np.mean(L)
    )
    return blured
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
    frame_samples = []
    clip = cv2.VideoCapture(video_path)
    frame_id = 0
    hashes=[]
    while True:
        success, frame = clip.read()
        if not success:
            break
        hash=tucker_hash(frame)
        print(hash.shape)
        frame_id += 1
        hashes.append(hash)

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
                db_path = path + '/' + file_name + '_compared_TD.npy'

                # if os.path.exists(db_path):
                #     print("exist for video", video_path)
                #     continue
                print("gen for ", video_path)
                TD=gen_hash(video_path)
                np.save(db_path, TD)
    return

traverseVideo(dataset=4)

