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
#from thash.tucker_hash import tucker_hash
from itertools import combinations
from scipy.signal import convolve2d
from skimage.color import rgb2lab
import pywt
from time import process_time
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = np.array(TPR) - np.array(FPR)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, Youden_index

def getBenchmarks(tp, fp, fn, tn):

    precision=(tp / (tp + fp))
    recall=tp/ (tp + fn)
    F1=2 * tp / (2 * tp + fp + fn)
    iou=tp/ (tp + fp + fn)
    accuracy=((tp + tn) / (tp + fp + fn + tn))

    print("Precision={:.2%}".format(precision))
    print("Recall={:.2%}".format(recall))
    print("F1={:.2%}".format(F1))
    print("IoU={:.2%}".format(iou))
    print("Accuracy={:.2%}".format(accuracy))

    return precision, recall, F1, iou, accuracy
def drawRoc(tpr,fpr,figname):
    plt.figure()
    plt.plot(fpr,tpr)
    plt.savefig(figname)
def get5Metrics(labels, scores,level):
    AUC = getAuc(labels, scores)

    print("AUC:", AUC)

    thresholds = np.arange(0, np.max(np.array(scores)), 0.5)
    tps = []
    fps = []
    tns = []
    fns = []
    tpr=[]
    fpr=[]
    for threshold in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(scores)):
            if scores[i] >= threshold:
                if labels[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if labels[i] == 1:
                    fn += 1
                else:
                    tn += 1
        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    optimal_th, optimal_index = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    print("tpr=",tpr)
    print("fpr=",fpr)
    print("best threshold",optimal_th)
    precision, recall, F1, iou, accuracy = getBenchmarks(tps[optimal_index], fps[optimal_index], fns[optimal_index], tns[optimal_index])
    figname='figs/dataset1_roc_'+level+".png"
    drawRoc(tpr,fpr,figname)

def getMetrics(tp, fp, tn, fn):
    precision=(tp / (tp + fp))
    recall=(tp / (tp + fn))
    F1=(2 * tp / (2 * tp + fp + fn))
    iou=(tp / (tp + fp + fn))
    accuracy=((tp + tn) / (tp + fp + fn + tn))


    print("Precision={:.2%}".format(precision))
    print("Recall={:.2%}".format(recall))
    print("F1={:.2%}".format(F1))
    print("IoU={:.2%}".format(iou))
    print("Accuracy={:.2%}".format(accuracy))

    return precision,recall,F1,iou,accuracy
def get_frame_info(hd,segment_size,video_len):
    output=[]#[-1 for i in range(video_len)]
    label=[]

    for i in range(len(hd)):
        output.extend([hd[i] for j in range(segment_size)])
    output.extend([hd[-1] for j in range(video_len%segment_size)])
  #  print(len(output),video_len)
    return output
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
    np.save('data/baseline1.npy', [path_1920,frame_hash_values, sampling_lists])
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


def preprocessing_PVH(imgs,sample_rate):
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

def feature_extraction_PVH(frame_samples,block_size):
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
def hash_extraction_PVH(H,V,T,a):
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
def PVH(imgs,received_hashes,_):
    sample_rate=5
    block_size=16
    a=math.pi/3

    frame_samples=preprocessing_PVH(imgs,sample_rate=1)


    H_3D,V_3D,T_3D=feature_extraction_PVH(frame_samples,block_size)
  #  print("3d", H_3D[0][1][1])
    H_2D, V_2D, T_2D=hash_extraction_PVH(H_3D,V_3D,T_3D,a)

    hash_compare_PVH([H_2D, V_2D, T_2D], [H_2D, V_2D, T_2D])
   # print("2d",len(H_2D),len(H_2D[0]))

    return H_2D, V_2D, T_2D

def preprocessing_TD(img_array, size=(256, 256)):
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
def tucker_hash(img, **kwargs):

    l_matrix = preprocessing_TD(img)
    _, factors = tensor_decomposition(l_matrix, **kwargs)
    a_factor, b_factor, c_factor = factors[0], factors[1], factors[2]
    return make_hash(a_factor, b_factor, c_factor).reshape((-1, 8))
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
 #   print("u_matrix",u_matrix.shape,u_matrix[0:32,0:32])
 #   print("block_matrix",block_matrix.shape,block_matrix[0])
   # x2=np.lib.stride_tricks.as_strided(u_matrix, (q_block_size, q_block_size))
    r, c = u_matrix.shape
    size = u_matrix.itemsize
    ast = np.lib.index_tricks.as_strided
    x2 = ast(u_matrix,
                      shape=(r - q_block_size + 1, c - q_block_size + 1, q_block_size, q_block_size),
                      strides=(c * size, size, c * size, size))
    s1, s2 = x2.shape[2::]
   # print("mn",m,n)
    x2 = np.rollaxis(x2,0,1).reshape(-1,s1,s2)
  #  print("x2",x2.shape,x2[0][0])
    np.random.seed(0)
    # print("D_n2_seed",D_n2_seed)
    # D_n2=np.random.choice(D_n2_seed,D/Q*D/Q,replace=False)
    sample_seq=np.random.choice(x2.shape[0], n, replace=False)
    for i in range(len(sample_seq)):
        while sample_seq[i]//(c-q_block_size+1)%q_block_size==0:
            sample_seq[i]=np.random.choice(x2.shape[0], 1, replace=False)[0]
    block_matrix2 = x2[sample_seq]
   # print(np.random.choice(x2.shape[0], n, replace=False))
    np.random.seed(1)
    tensor_3D = np.concatenate((block_matrix, block_matrix2), axis=0)
    x = tl.tensor(tensor_3D, dtype="float")
    # print("l_matrix",l_matrix.shape,l_matrix)
    # print("u_matirx",u_matrix.shape,u_matrix)
    # print("tensor_3D",tensor_3D.shape,tensor_3D)
    # print("block_matrix", block_matrix2.shape, block_matrix2)
    return tucker(x, rank=[i_core, j_core, k_core], random_state=random_state)


def make_hash(a_factor, b_factor, c_factor):
    a_p = a_factor.mean(axis=1)
    a_h = (a_p >= a_p.mean()).astype(int)
    b_p = b_factor.mean(axis=1)
    b_h = (b_p >= b_p.mean()).astype(int)
    c_p = c_factor.mean(axis=1)
    c_h = (c_p >= c_p.mean()).astype(int)
    return np.hstack((a_h, b_h, c_h))

def hash_compare_TD(hashes1,hashes2):
    D=[]
    for i in range(len(hashes1)):
        D.append(np.count_nonzero(hashes2[i]!=hashes1[i]))
    return D
def TD(imgs,received_hashes,_):

    frame_id = 0
    hashes=[]
    for frame in imgs:

        hash=tucker_hash(frame)
        frame_id += 1
        hashes.append(hash)
    hash_compare_TD(hashes,hashes)
    return hashes
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
        print(np.max(D),np.min(D))
        u=np.average(D)
        b=round(u-1)
        hash.append(b)
   # print("hahs",hash)
    return hash



def hash_compare_LRH(H_set1,H_set2):
    dif1 = H_set1 - np.average(H_set1)
    dif2 = H_set2 - np.average(H_set2)
    #  print("dif1",dif1)
    #  print("dif2",dif2)
    elp = 0.0000000000000001
    #  print(dif1)
    #   print(dif1*dif1)
    frame_dif = H_set1 - H_set2
    #   print("H_set1",H_set1)
    video_corr = np.sum(dif1 * dif2) / (np.sqrt(np.sum(dif1 * dif1)) * np.sqrt(np.sum(dif2 * dif2)) + elp)

    return frame_dif, video_corr




def LRH(frames,received_hashes,_):
    N,r=16,2


    imgs = preprocessingLRH(frames)
    Rs=low_rank_calculation(imgs,N,r)
    hashes=low_rank_compression(Rs)
    hash_compare_LRH(np.array(hashes),np.array(hashes))
    return hashes
def hash_compare_PVH(hashes1,hashes2):
    H_2D_1,V_2D_1,T_2D_1=hashes1
    H_2D_2, V_2D_2, T_2D_2 =hashes2
    K=len(H_2D_1)
    M=len(H_2D_1[0])
    N=len(V_2D_1[0])
    Dh=[[-1 for i in range(M)] for k in range(K)]
    Dv=[[-1 for j in range(N)] for k in range(K)]

    for k in range(K):
        for i in range(M):
            Dh[k][i]=abs(H_2D_1[k][i]-H_2D_2[k][i])
    for k in range(K):
        for j in range(N):
            Dv[k][j]=abs(V_2D_1[k][j]-V_2D_2[k][j])
    Arr=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]
    for k in range(K):
        for i in range(M):
            for j in range(N):
                if Dh[k][i]>0 and Dv[k][j]>0:
                    Arr[k][i][j]=1
                else:
                    Arr[k][i][j] = 0
    s=5
    Q=int(K/s)
    Arr1=[[[-1 for j in range(N)] for i in range(M)] for q in range(Q)]
    for q in range(Q):
        for i in range(M):
            for j in range(N):
                su=0
                for k in range((q-1)*s,q*s):
                    su+=Arr[k][i][j]

                if su==s:
                    Arr1[q][i][j] = 1
                else:
                    Arr1[q][i][j] = 0
    Dq=[0 for i in range(Q)]
    for q in range(Q):
        for i in range(M):
            for j in range(N):
                Dq[q]+=Arr1[q][i][j]

  #  print(Dq)
    return Dq
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
def ahash(imgs,hashes_sender,sampling_lists_sender):
    method=imagehash.average_hash
    frame_hash_values = []
    for f in imgs:
        im=Image.fromarray(f)
        cur_frame_hash = method(im)

        frame_hash_values.append((cur_frame_hash))

v1='C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless/01_original.mp4'
v2='C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_010.mp4'
v3='C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_001.mp4'
v4='C:/Users/tangli/PycharmProjects/datas/videos/Shake_1920.mp4'
v5='C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
videos=[v1,v2,v3,v4,v5]
resolutions=[320,704,1280,1920,3840]
video_id=2
resolution,video_path=resolutions[video_id],videos[video_id]
#video_path='C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless/01_original.mp4'
#video_path='C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_010.mp4'
path,hashes_CCSH,sampling_lists_CCSH=np.load('data/CCSH_'+str(resolution)+'.npy', allow_pickle=True)
path,hashes_baseline2,sampling_lists_baseline2=np.load('data/baseline2_'+str(resolution)+'.npy', allow_pickle=True)
path,hashes_baseline1,sampling_lists_baseline1=np.load('data/baseline1_'+str(resolution)+'.npy', allow_pickle=True)


#_,hashes_PVH,_=np.load('data/PVH_'+str(resolution)+'.npy', allow_pickle=True)
#print("hashes_PVH")
hashes_PVH=np.load(videos[2].replace('.mp4','_compared_PVH.npy'),allow_pickle=True)
path,hashes_ahash=np.load('data/ahash_'+str(resolution)+'.npy', allow_pickle=True)
hashes_TD=np.load(videos[2].replace('.mp4','_compared_TD2n.npy'),allow_pickle=True)
hashes_LRH=np.load(videos[2].replace('.mp4','_compared_LRH.npy'),allow_pickle=True)
time_start1 = process_time()
#path_1920 = 'C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
clip = cv2.VideoCapture(video_path)

group_size=16
imgs0 = []
while True:
    success, frame = clip.read()
    if not success:
        break
    imgs0.append(frame)
imgs=imgs0[:group_size*(len(imgs0)%group_size)-1]
time_end1=process_time()
print(time_end1-time_start1)
print(len(imgs),len(imgs)%group_size)
repeatTime=5
time_costs=[]
methods=[PVH,TD,LRH,ahash,baseline1,baseline2,CCSH]
hashes=[hashes_PVH,hashes_TD,hashes_LRH,hashes_ahash,hashes_baseline1,hashes_baseline2,hashes_CCSH]
sampling_lists=[[],[],[],[],sampling_lists_baseline1,sampling_lists_baseline2,sampling_lists_CCSH]
for j in range(2,4):
    method=methods[j]
    hashes_sender=hashes[j]
    sampling_lists_sender=sampling_lists[j]
    time_start = process_time()
    for i in range(repeatTime):
        method(imgs,hashes_sender,sampling_lists_sender)
    time_end=process_time()
    time_costs.append((time_end-time_start)/(repeatTime*len(imgs)))
print(time_costs)
#320*240:  [0.005580357142857143, 0.1740327380952381, 0.12105654761904762, 0.0005208333333333333, 0.0005208333333333333, 0.0005952380952380953, 0.0005208333333333333]
#704*480: [0.031086215932914045, 0.1804245283018868, 0.12044680293501048, 0.0022929769392033544, 0.0005241090146750524, 0.0004913522012578616, 0.0005568658280922431]
#1280*720: [0.09207589285714286, 0.24813988095238096, 0.1263764880952381, 0.006882440476190476, 0.0005580357142857143, 0.0005022321428571428, 0.0005208333333333333]
#1920*1080: [0.214375, 0.40239583333333334, 0.12979166666666667, 0.014583333333333334, 0.0005208333333333333, 0.000625, 0.0005208333333333333]
#3840*2160: [1.1598958333333333, 2.1084375, 0.18979166666666666, 0.0596875, 0.000625, 0.0005208333333333333, 0.0005729166666666667]
#group size=16
#320*240:[0.06139112903225807, 0.0009072580645161291]
#704*480:[0.11320954106280193, 0.0028381642512077293]
#1280*720:[0.11589566929133858, 0.0073818897637795275]
#1920*1080:[0.13702552356020942, 0.01775196335078534]
#3840*2160:[0.18520942408376964, 0.0700261780104712]

#group size=1
#320*240:  [0.12247242647058823, 0.0023634453781512603]
#704*480: [0.1283153886554622, 0.0026260504201680674]
#1280*720:[0.1345537842669845, 0.006257449344457688]
#1920*1080:[0.14553720735785952, 0.014945652173913044]
#3840*2160: [0.18436454849498327, 0.05847617056856187]