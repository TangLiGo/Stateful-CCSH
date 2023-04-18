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
from metrics import *
from sklearn.metrics import roc_auc_score, roc_curve
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



def data_processing(nums_forgery,nums_compression):
    video_len = [210, 329, 313, 319, 583, 262, 412, 274, 554, 239]
    forged_info = [[1, 210], [120, 162], [233, 313], [61, 122], [1, 190], [211, 262], [90, 117], [133, 160], [169, 359],
                   [104, 149]]

    frame_labels=[]
    frame_scores=[]
    video_scores=[]
    video_labels=[]

    segment_size=5
    for i in range(1,len(nums_forgery)):  # 第几组
        for j in range(len(nums_forgery[i])):  # 第几个
            processed_hd=get_frame_info(nums_forgery[i][j],segment_size,video_len[j])
            frame_scores.extend(processed_hd)
            video_scores.append(np.max(np.array(nums_forgery[i][j])))
          #  print(forged_info[j],len(nums_forgery[i][j]))
            processed_label=[0 for j in range(forged_info[j][0])]+[1 for j in range(forged_info[j][0],forged_info[j][1])]+[0 for j in range(forged_info[j][1],video_len[j])]

            frame_labels.extend(processed_label )
            if len(processed_label)!=video_len[j] or video_len[j]!=len(processed_hd):
                print("error forgery",j,len(processed_label),video_len[j],len(processed_hd))
            video_labels.append(1)

    for i in range(len(nums_compression)):
        for j in range(len(nums_compression[i])):
            processed_hd = get_frame_info(nums_compression[i][j], segment_size, video_len[j])
            frame_scores.extend(processed_hd)

            video_scores.append(np.max(np.array(nums_compression[i][j])))
            processed_label=[0 for k in range(video_len[j])]
            frame_labels.extend(processed_label)
            if len(processed_label)!=video_len[j] or video_len[j]!=len(processed_hd):
                print("error compression",j,len(processed_label),video_len[j],len(processed_hd))
            video_labels.append(0)
    print("video_scores",video_scores)
    print("Get info for videolevel")
    get5Metrics(video_labels,video_scores,'videolevel')
    print("Get info for framelevel")
    get5Metrics(frame_labels,frame_scores,'framelevel')

#-----------------------------------------------------------------------

# Return the luminance of Color c.
def blockmeans(blocks):
    outputs=[]
    for block_set in blocks:
        mean_frame=[]
        for block in block_set:
            mean_frame.append(mean2(block))
        outputs.append(mean_frame)
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
            luminances = r#(.299 * r) + (.587 * g) + (.114 * b)
            frame_samples.append(luminances)
        frame_id+=1
  #  print("ee")
    return frame_samples

def feature_extraction(block_samples,sampling_lists,M,N,a):


    K=len(block_samples)
    A2=blockmeans(block_samples)
    A=[[[-1 for j in range(N)] for i in range(M)] for k in range(K)]
    for k in range(K):
        for i in range(len(sampling_lists[k])):
            row=int(sampling_lists[k][i] / N)
            col=int(sampling_lists[k][i] % N)
        #    print(len(A2),len(A2[0]),k,i,M,N,K,row,col)
            A[k][row][col]=A2[k][i]
    A=np.array(A)
  #  print("frame size",len(frame_samples[0]),len(frame_samples[0][0]))
  #  print("A size",len(A),len(A[0]),len(A[0][0]))
    H_2D = [[-1 for i in range(M)] for k in range(K)]
    V_2D = [[-1 for j in range(N)] for k in range(K)]
    T_2D = [[-1 for j in range(N)] for i in range(M)]
    for k in range(K):
        for i in range(M):
          #  print("A[k, i, :]",A[k, i, :])
            H_2D[k][i] = normalization(A[k, i, :], a)
    for k in range(K):
        for j in range(N):
            V_2D[k][j] = normalization(A[k, :, j], a)
    for i in range(M):
        for j in range(N):
            T_2D[i][j] = normalization(A[:, i, j], a)

    return H_2D,V_2D,T_2D,A

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




def blocks_extraction(frames,sample_num,sample_corr,block_size=16):
    frame_number=0
    sampling_lists=[]
    sampled_blocks=[]
    for f in frames:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)

        cor_blocks = []
        sampling_list = []
        if frame_number != 0:

            # print("last",last_sampling_list_row)
            for i in range(sample_num):
                block_temp = f[last_sampling_list_row[i] * block_size:(last_sampling_list_row[i] + 1) * block_size,
                             last_sampling_list_col[i] * block_size:(last_sampling_list_col[i] + 1) * block_size]

                cor_blocks.append(corr2(block_temp, last_img_blocks[i]))
            sorted_blocks = sorted(cor_blocks, key=abs)
            i = 0
            sampling_list.extend(np.random.randint(low=0, high=row_block_range * col_block_range,
                                                   size=(sample_num - sample_corr,)))
            while (len(sampling_list) < sample_num):
                index_temp = last_sampling_list[cor_blocks.index(sorted_blocks[i])]
                cor_blocks[cor_blocks.index(sorted_blocks[i])] = 2
                # print(index_temp,sampling_list)
                if index_temp in sampling_list:  # 此方法不好，对于block的corr2都差不多的部分frame而言 会sample0-19的block
                    i += 1
                    continue
                i += 1
                sampling_list.append(index_temp)

        else:
            sampling_list = np.random.randint(low=0, high=row_block_range * col_block_range, size=(sample_num,))

        # print("col",col_block_range,row_block_range)
        frame_number += 1
        sampling_lists.append(sampling_list)

        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]
      #  print("cur",sampling_list_row,sampling_list_col)
        last_sampling_list_col = sampling_list_col
        last_sampling_list_row = sampling_list_row
        last_sampling_list = sampling_list

        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]

            blocks.append(block)
        last_img_blocks = blocks
        sampled_blocks.append(blocks)


    return sampled_blocks,sampling_lists,row_block_range,col_block_range
def gen_hash_sender(video_path):
    sample_rate=1
    block_size=16
    a=math.pi/3
    sample_num=20
    sample_corr=3
    frame_samples=preprocessing(video_path,sample_rate)
    sampled_blocks,sampling_lists,M,N=blocks_extraction(frame_samples,sample_num,sample_corr,block_size)
    print(len(sampled_blocks),len(sampled_blocks[0]),len(sampled_blocks[0][0]))
    hashes=feature_extraction(sampled_blocks,sampling_lists,M,N,a)

    p1 = os.path.basename(video_path)
    file_name = os.path.splitext(p1)[0]
   # plt.figure()
   # plt.plot(simis)
  #  plt.savefig("figs_corr/"+file_name+"_1.png")
  #  plt.show()
    return hashes,sampling_lists
def blocks_extraction_pure(frames,sampling_lists,block_size):
    frame_number=0

    sampled_blocks=[]
    for f in frames:
        width = f.shape[1]
        height = f.shape[0]
        row_block_range = int(height / block_size)
        col_block_range = int(width / block_size)

        cor_blocks = []
        sampling_list =sampling_lists[frame_number]

        sampling_list_row = [int(sample_index / col_block_range) for sample_index in sampling_list]
        sampling_list_col = [int(sample_index % col_block_range) for sample_index in sampling_list]
        blocks = []
        for i in range(len(sampling_list_row)):
            block = f[sampling_list_row[i] * block_size:(sampling_list_row[i] + 1) * block_size,
                    sampling_list_col[i] * block_size:(sampling_list_col[i] + 1) * block_size]

            blocks.append(block)
        sampled_blocks.append(blocks)
        frame_number += 1
    return sampled_blocks,row_block_range,col_block_range
def gen_hash_receiver(video_path,sampling_lists):
  #  print("re" ,video_path)
    sample_rate=1
    block_size=16
    a=math.pi/3
    frame_samples=preprocessing(video_path,sample_rate)
    sampled_blocks,M,N=blocks_extraction_pure(frame_samples,sampling_lists,block_size)

    hashes_receiver=feature_extraction(sampled_blocks,sampling_lists,M,N,a)

    return hashes_receiver
def hash_compare(hashes_sender,hashes_receiver):
    H_2D_1,V_2D_1,T_2D_1=hashes_sender
    H_2D_2, V_2D_2, T_2D_2 = hashes_receiver
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
            if 'original' in file and ('.mp4' in file or '.avi' in file):
                original_videos[i].append(video_path)
            elif 'forged' in file and ('.mp4' in file or '.avi' in file):
                forged_videos[i].append(video_path)

    hds_forgery=[]
    for i in range(len(original_videos)):
        for j in range(i,len(forged_videos)):
            sub_hds_forgery = []
            for k in range(len(original_videos[i])):
                print(original_videos[i][k], forged_videos[j][k])
                hashes, sampling_lists = gen_hash_sender(original_videos[i][k])
                hashes_receiver=gen_hash_receiver(forged_videos[j][k],sampling_lists)
                print( (hashes[3]!=hashes_receiver[3]).any())
                index=np.arange(0,len(hashes[3]))
                if ((hashes[3][130:140]!=hashes_receiver[3][130:140]).any()):
                    print(index[hashes!=hashes_receiver])

                sub_hds_forgery.append(hash_compare( hashes[:3], hashes_receiver[:3]))
            hds_forgery.append(sub_hds_forgery)

    hds_compression=[]
    for i in range(len(original_videos)-1):
        for j in range(i+1,len(original_videos)):
            sub_hds_compression = []
            for k in range(len(original_videos[i])):
                print(original_videos[i][k], original_videos[j][k])
                hashes, sampling_lists = gen_hash_sender(original_videos[i][k])
                hashes_receiver=gen_hash_receiver(original_videos[j][k],sampling_lists)
                sub_hds_compression.append(hash_compare( hashes[:3], hashes_receiver[:3]))
            hds_compression.append(sub_hds_compression)
    return hds_forgery,hds_compression





hds_forgery,hds_compression=traverseVideo()
np.save('data/dataset1_authen_ccsh_pvh.npy',[hds_forgery,hds_compression])
hds_forgery,hds_compression=np.load('data/dataset1_authen_ccsh_pvh.npy',allow_pickle=True)
print(hds_forgery,hds_compression)
data_processing(hds_forgery,hds_compression)


