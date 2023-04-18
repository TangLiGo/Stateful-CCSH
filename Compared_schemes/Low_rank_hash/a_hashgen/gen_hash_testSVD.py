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
import scipy


def preprocessingLRH(images, size=(256, 256), r_len=256):
    img_array = []
    v_len = len(images)

    for i in range(v_len):
        frame = images[i]
        lu = Image.fromarray(frame).convert('L')
       # b, g, r = cv2.split(frame)
       # lu=(65.481 * r) + (128.553 * g) + (24.966 * b)+16
        img_array.append(np.array(lu.resize(size)))
       # print("lu",np.array(lu))
       # print("h",np.array(lu.resize(size)))
       # print("res",resize(np.array(lu), size, anti_aliasing=True))
    return img_array
def low_rank_calculation(imgs,N,r):
    #print("he",len(imgs),imgs)
    L=len(imgs)//N
   # print("L",L)
    Rs=[]
    for i in range(L):
        group=np.asarray(imgs[i*N:(i+1)*N])
     #   print("group",group)
        #SVD
        u, s, vh = np.linalg.svd(group)
        u_,s_,vh_ = u[:, :, :r],s[:, None, :r],vh[:, :r, :]
      #  print("s",s)
       # print("shape",u.shape,s.shape,vh.shape)
        u1, s1, vh1 = np.linalg.svd(group[0])
        u1_, s1_, vh1_ = u1[:, :r], s1[ :r], vh1[:r, :]
        u2, s2, vh2 = scipy.linalg.svd(np.asarray(group[0]))
        u2_, s2_, vh2_ = u2[:, :r], s2[ :r], vh2[:r, :]
       # u3, s3, vh3 = scipy.linalg.svd(np.matrix(group))
      #  print("s2",s2)
      #  print("s1",s1)
        #print((u1_ @ np.diag(s1_) @ vh1_).shape,u1_ @ np.diag(s1_) @ vh1_)
      #  print((u2_ @ np.diag(s2_) @ vh2_).shape, u2_ @ np.diag(s2_) @ vh2_)
        #print((u2_ @ np.diag(s2_) @ vh2_).shape, u2_ @ np.diag(s2_) @ vh2_)
       # print("s_",s_)
       # print("shape s_", u_.shape,s_.shape,vh_.shape)
        # print("before",(np.matmul(u * s[:, None, :], vh)).shape, np.matmul(u * s[:, None, :], vh))
        # print("after",(np.matmul(u_ * s_, vh_)).shape, np.matmul(u_ * s_, vh_))
        # print(s.shape, u.shape, vh.shape)
        # print(s_.shape,u_.shape,vh_.shape)
        # print("before",(np.matmul(u * s[:, None, :], vh)).shape,)
        # print("after",(np.matmul(u_ * s_, vh_)).shape)
       # Gr_ref=np.matmul(u_ * s_, vh_)#ur.dot(sr).dot(vhr)
        Gr_used = np.matmul(u_ *(s[:, None, :r]) , vh_)
        #print("Gr",Gr_ref.shape,Gr_ref)
       # print("Gr_used",Gr_used.shape,Gr_used[0])
        # group2 = np.array(imgs[i * N:(i + 1) * N]).T
        # u2, s2, vh2 = np.linalg.svd(group2, full_matrices=False)
        # u2_, s2_, vh2_ = u2[:, :, :r], s2[:, None, :r], vh2[:, :r, :]
        # Gr2 = np.matmul(u2_ * s2_, vh2_)  # ur.dot(sr).dot(vhr)
        # R2 = np.average(Gr2, axis=2)
        # print("S",Gr2.shape,R2.shape,R2)
        R=np.average(Gr_used,axis=0)
        # print(Gr)
      #  print("R",R.shape,R)
        Rs.append(R)
    return Rs
def low_rank_compression(Rs):
    hash=[]
    for R in Rs:
        D,highlevel=pywt.dwt2(R,'haar')
        print(len(D),D.shape,D)
        print(np.max(D),np.min(D))
        u=np.average(D)
        b=round(u+0.5)
        hash.append(b)
    print("hahs",hash)
    return hash








def gen_hash(video_path):
    N,r=2,2
    frame_samples = []
    clip = cv2.VideoCapture(video_path)
    frame_id = 0

    frames=[]
    while True:
        success, frame = clip.read()
        if not success:
            break

        frames.append(frame)
        frame_id += 1

    imgs = preprocessingLRH(frames)
    Rs=low_rank_calculation(imgs,N,r)
    hashes=low_rank_compression(Rs)
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
                db_path = path + '/' + file_name + '_compared_LRH_N=1.npy'

                # if os.path.exists(db_path):
                #     print("exist for video", video_path)
                #     continue
                print("gen for ", video_path)
                TD=gen_hash(video_path)
              #  np.save(db_path, TD)
    return


traverseVideo(dataset=1)
traverseVideo(dataset=4)