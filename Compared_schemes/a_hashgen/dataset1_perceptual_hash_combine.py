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


#-----------------------------------------------------------------------

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

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r
def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def preprocessing(video_path,sample_rate):
    frame_samples=[]
    clip = cv2.VideoCapture(video_path)
    frame_id=0
    while True:
        success, frame = clip.read()
        if not success:
            break
        if frame_id%sample_rate==0:
            b, g, r = cv2.split(frame)
            luminances = (.299 * r) + (.587 * g) + (.114 * b)
            frame_samples.append(luminances)
        frame_id+=1
  #  print("ee")
    return frame_samples
def feature_extraction_2D(frame_samples,block_size):
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

    return H[0],V[0]
def feature_extraction(frame_samples,block_size):
    M=int(len(frame_samples[0])/block_size)
    N=int(len(frame_samples[0][0])/block_size)
    K=len(frame_samples)
    A=blockmeans(frame_samples,block_size)
  #  print("frame size",len(frame_samples[0]),len(frame_samples[0][0]))
  #  print("A size",len(A),len(A[0]),len(A[0][0]))
    H=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]
    V=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]
    T=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]
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
    for i in range(M):
        for j in range(N):
            for k in range(K-1):
                T[k][i][j]=A[k+1][i][j]-A[k][i][j]
            T[K-1][i][j] = A[0][i][j] - A[K-1][i][j]
    return H,V,T
def DCT(x,i,m):
    L=len(x)
    X0=0
    output=[-1 for k in range(L)]

    Xc0_m = 0
    Xs0_m_1 = 0
    for n in range(L):
        Xc0_m += x[n] * math.cos(math.pi * m * (n + 1 / 2) / L)
        Xs0_m_1 += x[n] * math.sin(math.pi * (m + 1) * (n + 1 / 2) / L)
    Xci_m=math.sqrt(Xc0_m^2+Xs0_m_1^2)*math.cos(math.pi*m/L*i-math.atan(Xs0_m_1/Xc0_m))
    Xsi_m_1=math.sqrt(Xc0_m^2+Xs0_m_1^2)*math.cos(math.pi*m/L*i-math.atan(Xs0_m_1/Xc0_m)+math.pi/2)
    return Xci_m, Xsi_m_1
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
def hash_extraction(H,V,T,a):
    M=len(H[0])
    N=len(H[0][0])
    K=len(H)
    H_2D=[[-1 for i in range(M)] for k in range(K)]
    V_2D=[[-1 for j in range(N)] for k in range(K)]
    T_2D=[[-1 for j in range(N)] for i in range(M)]
    H=np.array(H)
    V = np.array(V)
    T = np.array(T)
    for k in range(K):
        for i in range(M):
            H_2D[k][i]=normalization(H[k,i,:],a)
    for k in range(K):
        for j in range(N):

            V_2D[k][j]=normalization(V[k,:,j],a)
    for i in range(M):
        for j in range(N):

            T_2D[i][j]=normalization(T[:,i,j],a)

    return H_2D,V_2D,T_2D

def hash_extraction_2D(H,V,a):
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


    return H_2D[0],V_2D[0]

def get_similarity(last_frame,cur_frame,last_sampling_list,block_range,sample_num,sample_corr,col_block_range,block_size=16):
    last_sampling_list_row = [int(sample_index / col_block_range) for sample_index in last_sampling_list]
    last_sampling_list_col = [int(sample_index % col_block_range) for sample_index in last_sampling_list]

    cor_values=[]
    cur_sampling_list=[]
    cur_sampling_list.extend(np.random.randint(low=0, high=block_range,
                                           size=(sample_num - sample_corr,)))
    for i in range(len(last_sampling_list)):
        block_temp = cur_frame[last_sampling_list_row[i] * block_size:(last_sampling_list_row[i] + 1) * block_size,
                     last_sampling_list_col[i] * block_size:(last_sampling_list_col[i] + 1) * block_size]
        last_block_temp=last_frame[last_sampling_list_row[i] * block_size:(last_sampling_list_row[i] + 1) * block_size,
                     last_sampling_list_col[i] * block_size:(last_sampling_list_col[i] + 1) * block_size]
        cor_values.append(corr2(block_temp, last_block_temp))
    sorted_values = sorted(cor_values, key=abs)


    i = 0
    print(sorted_values)
    while (len(cur_sampling_list) < sample_num):
        index_temp = last_sampling_list[cor_values.index(sorted_values[i])]
        cor_values[cor_values.index(sorted_values[i])] = 2
        # print(index_temp,sampling_list)
        if index_temp in cur_sampling_list:
            i += 1
            continue
        i += 1
        cur_sampling_list.append(index_temp)
  #  print("sorted_values",sorted_values)
    return cur_sampling_list,abs(min(sorted_values))
def gen_hash(video_path):
    sample_rate=1
    block_size=16
    a=math.pi/3
    sample_num=20
    sample_corr=3
    frame_samples=preprocessing(video_path,sample_rate)
    frame_num=0
    hashes=[]
    sampling_lists=[]
    simis=[]
    for frame in frame_samples:
        width = frame.shape[1]
        height = frame.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)
        moving_flag=False
        if frame_num==0:
            H_3D,V_3D=feature_extraction_2D([frame],block_size)
            H_2D, V_2D = hash_extraction_2D([H_3D], [V_3D], a)
            cur_hash=[H_2D, V_2D]
            last_frame=frame
            last_sampling_list= np.random.randint(low=0, high=row_block_range*col_block_range, size=(sample_num,))
            sampling_lists.append(last_sampling_list)
            last_hash=cur_hash
            hashes.append([frame_num,cur_hash])
        else:
          #  print("hh")

            sampling_list,simi_value=get_similarity(last_frame,frame,last_sampling_list,row_block_range*col_block_range,sample_num,sample_corr,col_block_range)
            sampling_lists.append(sampling_list)
            simis.append(simi_value)
            if simi_value < 0.9 :
                moving_flag = True
            if moving_flag:
                H_3D, V_3D = feature_extraction_2D([frame], block_size)
                H_2D, V_2D = hash_extraction_2D([H_3D], [V_3D], a)
                cur_hash =[ H_2D, V_2D]
                hashes.append([frame_num, cur_hash])

        frame_num+=1
    print(simis)
    p1 = os.path.basename(video_path)
    file_name = os.path.splitext(p1)[0]
    plt.figure()
    plt.plot(simis)
    plt.savefig("figs_corr/"+file_name+"_1.png")
  #  plt.show()
    return hashes,sampling_lists
def traverseVideo():
    data_path0 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless'
    data_path1 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossy_q10'
    data_path2 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossy_q20'
    data_path3 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossy_q30'
    data_path=[data_path0,data_path1,data_path2,data_path3]
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
                db_path = path + '/' + file_name + '_ccsh_pvh.npy'

            #    if os.path.exists(db_path):
                  #  print("exist for video", video_path)
                   # continue
                print("gen for ", video_path)
                hashes,sampling_lists= gen_hash(video_path)
                np.save(db_path, [hashes,sampling_lists])
    return

traverseVideo()

