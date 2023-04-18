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
def baseline1(imgs):


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
    np.save('data/baseline1_'+str(resolution)+'.npy', [path_1920,frame_hash_values, sampling_lists])
# Return the luminance of Color c.
def blockmeans(frames,block_size):
    outputs=[]
    M=int(len(frames[0])/block_size)
    N=int(len(frames[0][0])/block_size)
    for frame in frames:
        output=[[-1 for j in range(N)] for i in range(M)]
        for m in range(M):
            for n in range(N):
                output[m][n]=mean2(frame[m*block_size:(m+1)*block_size,n*block_size:(n+1)*block_size])
        outputs.append(output)
    return outputs


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def preprocessing(imgs,sample_rate):
    frame_samples=[]

    frame_id=0
    for frame in imgs:

        if frame_id%sample_rate==0:
            b, g, r = cv2.split(frame)
            luminances = (.299 * r) + (.587 * g) + (.114 * b)
            frame_samples.append(luminances)
        frame_id+=1
  #  print("ee")
    return frame_samples

def feature_extraction(frame_samples,block_size):
    M=int(len(frame_samples[0])/block_size)
    N=int(len(frame_samples[0][0])/block_size)
    K=len(frame_samples)
    A=blockmeans(frame_samples,block_size)
  #  print("frame size",len(frame_samples[0]),len(frame_samples[0][0]))
  #  print("A size",len(A),len(A[0]),len(A[0][0]))
    H=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]
    V=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]

    for k in range(K):
        for i in range(M):
            for j in range(N-1):
                H[k][i][j]=A[k][i][j+1]-A[k][i][j]
            H[k][i][N-1] = A[k][i][0] - A[k][i][N-1]
    for k in range(K):
        for j in range(N):
            for i in range(M-1):
                V[k][i][j]=A[k][i+1][j]-A[k][i][j]
            V[k][M-1][j] = A[k][0][j] - A[k][M-1][j]

    return H,V

def normalization(x,a):
    L=len(x)
    #get Xc0(2),Xs0_1
    Xc0_2 = 0
    Xs0_1 = 0
    for n in range(L):
        Xc0_2 += x[n] * math.cos(math.pi * 2 * (n + 1 / 2) / L)
        Xs0_1 += x[n] * math.sin(math.pi * (1 + 1) * (n + 1 / 2) / L)
    if Xs0_1==0 and Xc0_2==0:
        print("errr")
        return -1
    shift_i=int((L*math.atan(Xs0_1/Xc0_2)+L*a)/(2*math.pi))%L


    return shift_i
def hash_extraction(H,V,a):
    M=len(H[0])
    N=len(H[0][0])
    K=len(H)
    H_2D=[[-1 for i in range(M)] for k in range(K)]
    V_2D=[[-1 for j in range(N)] for k in range(K)]

    H=np.array(H)
    V = np.array(V)

    for k in range(K):
        for i in range(M):
            H_2D[k][i]=normalization(H[k,i,:],a)
    for k in range(K):
        for j in range(N):

            V_2D[k][j]=normalization(V[k,:,j],a)


    return H_2D,V_2D






def PVH(imgs):
    sample_rate=5
    block_size=16
    a=math.pi/3

    frame_samples=preprocessing(imgs,sample_rate=1)


    H_3D,V_3D=feature_extraction(frame_samples,block_size)
  #  print("3d", H_3D[0][1][1])
    H_2D, V_2D=hash_extraction(H_3D,V_3D,a)


   # print("2d",len(H_2D),len(H_2D[0]))
    np.save('data/PVH_'+str(resolution)+'.npy', [path_1920, [H_2D, V_2D], []])
    print(np.max(np.array(V_2D)))
    return H_2D, V_2D
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
        if img_i != 0:
            f2 = imgs[img_i - 1]
            sampling_list_new = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num,))
            sampling_list_row_new = [int(sample_index / col_block_range) for sample_index in sampling_list_new]
            sampling_list_col_new = [int(sample_index % col_block_range) for sample_index in sampling_list_new]
            for i in range(sample_num):
                block_temp = f[sampling_list_row_new[i] * block_size:(sampling_list_row_new[i] + 1) * block_size,
                             sampling_list_col_new[i] * block_size:(sampling_list_col_new[i] + 1) * block_size]

                block_temp2 = f2[sampling_list_row_new[i] * block_size:(sampling_list_row_new[i] + 1) * block_size,
                              sampling_list_col_new[i] * block_size:(sampling_list_col_new[i] + 1) * block_size]
                cor_blocks.append(corr2(block_temp, block_temp2))
            sorted_blocks = sorted(cor_blocks, key=abs)
            i = 0
            sampling_list.extend(np.random.randint(low=0, high=row_block_range * col_block_range,
                                                   size=(sample_num - sample_insertion,)))
            while (len(sampling_list) < sample_num):
                index_temp = sampling_list_new[cor_blocks.index(sorted_blocks[i])]
                cor_blocks[cor_blocks.index(sorted_blocks[i])] = 2

                if index_temp in sampling_list:  # 此方法不好，对于block的corr2都差不多的部分frame而言 会sample0-19的block
                    i += 1
                    continue
                i += 1
                sampling_list.append(index_temp)

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
    np.save('data/baseline2_'+str(resolution)+'.npy', [path_1920,frame_hash_values,sampling_lists])
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
            sampling_list.extend(np.random.randint(low=0, high=row_block_range * col_block_range,
                                                   size=(sample_num - sample_insertion,)))
            while (len(sampling_list) < sample_num):
                index_temp = last_sampling_list[cor_blocks.index(sorted_blocks[i])]
                cor_blocks[cor_blocks.index(sorted_blocks[i])] = 2

                if index_temp in sampling_list:  # 此方法不好，对于block的corr2都差不多的部分frame而言 会sample0-19的block
                    i += 1
                    continue
                i += 1
                sampling_list.append(index_temp)

        else:
            sampling_list = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num,))

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
    np.save('data/CCSH_'+str(resolution)+'.npy', [path_1920,frame_hash_values, sampling_lists])
def ahash(imgs):
    method=imagehash.average_hash
    frame_hash_values = []
    for f in imgs:
        im=Image.fromarray(f)
        cur_frame_hash = method(im)

        frame_hash_values.append((cur_frame_hash))

    np.save('data/ahash_'+str(resolution)+'.npy', [path_1920, frame_hash_values])

v1='C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless/01_original.mp4'
v2='C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_010.mp4'
v3='C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_001.mp4'
v4='C:/Users/tangli/PycharmProjects/datas/videos/Shake_1920.mp4'
v5='C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
videos=[v1,v2,v3,v4,v5]
resolutions=[320,704,1280,1920,3840]
video_id=4
resolution,path_1920=resolutions[video_id],videos[video_id]
#path_1920 = 'C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_004.mp4'
#path_1920 = 'C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
#path_1920 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless/01_original.mp4'

#path_1920 = 'C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
time_start1 = process_time()
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
repeatTime=1
time_costs=[]
methods=[PVH,ahash,baseline1,baseline2,CCSH]
for i in range(5):
    method=methods[i]
    time_start = process_time()

    for i in range(repeatTime):
        method(imgs)
    time_end=process_time()

    time_costs.append((time_end-time_start)/(repeatTime*len(imgs)))

print("time costs:",time_costs)
#320*240  [0.0004650297619047619, 0.001402529761904762, 0.0013597470238095237, 0.0003385416666666667]
#[0.004207589285714286, 0.00036830357142857145, 0.0004464285714285714, 0.0015178571428571428, 0.0014806547619047618]
#[0.00587797619047619, 0.0004464285714285714, 0.0005208333333333333, 0.001636904761904762, 0.0018601190476190475]

#704*480  [0.00045504958246346555, 0.001453222860125261, 0.0014760568893528183, 0.0018952244258872652]
#[0.026146488469601676, 0.0021095387840670858, 0.0005306603773584906, 0.0017099056603773585, 0.0015461215932914046]
#                       [0.0019195492662473794, 0.00048480083857442346, 0.0016378406708595387, 0.0015461215932914046]

#1280*720  [0.000458984375, 0.0014936755952380952, 0.0014271763392857144, 0.005296688988095238]
#[0.07102864583333333, 0.004957217261904762, 0.0004650297619047619, 0.0014601934523809524, 0.001404389880952381]
#[0.08232886904761905, 0.005171130952380952, 0.0004836309523809524, 0.001636904761904762, 0.0015438988095238095]
#                     [0.0054910714285714285, 0.0004910714285714286, 0.0016555059523809524, 0.001566220238095238]
#1920*1080 [0.0004818281938325991, 0.0015461522577092511, 0.0014755988436123348, 0.012318023816079295] [0.00048354900881057266, 0.001525072274229075, 0.0014162307268722467, 0.012238436123348018]
#[0.2900519686123348, 0.02582943281938326, 0.0004818281938325991, 0.0015659416299559472, 0.0014885049559471366]
#[0.2772405011013216, 0.035810159691629956, 0.0006883259911894273, 0.0020305616740088107, 0.002013353524229075]
#                   [0.012587761563876651, 0.0005076404185022027, 0.001686398678414097, 0.0015831497797356828]
#time costs: [0.28988849118942733, 0.04043915198237885, 0.0005506607929515419, 0.0017036068281938326, 0.0015143171806167401]
#0.28,0.01273403,
# 3840:[0.0005130208333333333, 0.0015416666666666667, 0.0015442708333333333, 0.0482421875]
#[1.23046875, 0.13734375, 0.0005729166666666667, 0.0016666666666666668, 0.0016145833333333333]
#                                  [0.049921875, 0.0004947916666666667, 0.0016927083333333334, 0.0015625]