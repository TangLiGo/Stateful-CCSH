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






def PVH(imgs):
    sample_rate=5
    block_size=16
    a=math.pi/3

    frame_samples=preprocessing(imgs,sample_rate=1)


    H_3D,V_3D,T_3D=feature_extraction(frame_samples,block_size)
  #  print("3d", H_3D[0][1][1])
    H_2D, V_2D, T_2D=hash_extraction(H_3D,V_3D,T_3D,a)


   # print("2d",len(H_2D),len(H_2D[0]))
    np.save('data/PVH.npy', [path_1920, [H_2D, V_2D, T_2D], []])
    print(np.max(np.array(V_2D)))
    return H_2D, V_2D, T_2D
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
    np.save('data/baseline2.npy', [path_1920,frame_hash_values,sampling_lists])
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
    np.save('data/CCSH.npy', [path_1920,frame_hash_values, sampling_lists])
def ahash(imgs):
    method=imagehash.average_hash
    frame_hash_values = []
    for f in imgs:
        im=Image.fromarray(f)
        cur_frame_hash = method(im)

        frame_hash_values.append((cur_frame_hash))

    np.save('data/ahash.npy', [path_1920, frame_hash_values])

#path_1920 = 'C:/Users/tangli/Desktop/research2_dataset/dataset4_revised/Lossless/Forged_010.mp4'
#path_1920 = 'C:/Users/tangli/PycharmProjects/datas/videos/Shake_3840.mp4'
path_1920 = 'C:/Users/tangli/Desktop/research2_dataset/dataset1/h264_lossless/01_original.mp4'

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
for i in range(len(methods)):
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
#[0.025271881551362684, 0.0019162735849056604, 0.0004585953878406709, 0.0014576781970649896, 0.0015559486373165619]
#[0.029284591194968554, 0.001899895178197065, 0.0004585953878406709, 0.001605083857442348, 0.0014085429769392034]
#1280*720  [0.000458984375, 0.0014936755952380952, 0.0014271763392857144, 0.005296688988095238]
#[0.07102864583333333, 0.004957217261904762, 0.0004650297619047619, 0.0014601934523809524, 0.001404389880952381]
#[0.08232886904761905, 0.005171130952380952, 0.0004836309523809524, 0.001636904761904762, 0.0015438988095238095]
#[0.08227306547619048]
#1920*1080 [0.0004818281938325991, 0.0015461522577092511, 0.0014755988436123348, 0.012318023816079295] [0.00048354900881057266, 0.001525072274229075, 0.0014162307268722467, 0.012238436123348018]
#[0.2900519686123348, 0.02582943281938326, 0.0004818281938325991, 0.0015659416299559472, 0.0014885049559471366]
#[0.2772405011013216, 0.035810159691629956, 0.0006883259911894273, 0.0020305616740088107, 0.002013353524229075]
#0.28,0.01273403,
# 3840:[0.0005130208333333333, 0.0015416666666666667, 0.0015442708333333333, 0.0482421875]
#[1.23046875, 0.13734375, 0.0005729166666666667, 0.0016666666666666668, 0.0016145833333333333]