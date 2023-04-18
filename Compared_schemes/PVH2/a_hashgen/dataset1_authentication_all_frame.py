import hashlib
import numpy as np
from moviepy.editor import VideoFileClip

import imagehash
from PIL import Image
import cv2
import distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os

import math
import timeit
from time import process_time
import sys
from metrics import *

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
def data_processing(nums_forgery,nums_compression):
   # forged_info = [[100, 133], [111, 187], [99, 172], [95, 181], [64, 72], [1, 32], [103, 143], [120, 138], [77, 131],
              #     [59, 75], [82, 126], [66, 99], [6, 48], [50, 70], [107, 132], [1, 30]]
    #video_len = [210, 329, 313, 319, 583, 262, 412, 274, 554, 239]
    forged_info =[[0, 42], [24, 33], [46, 63], [12, 25], [0, 38], [42, 53], [18, 24], [26, 32], [33, 72], [20, 30]]
    print(len(nums_forgery[0][1]),nums_forgery[0][1])
    frame_labels=[]
    frame_scores=[]
    video_scores=[]
    video_labels=[]


    for i in range(len(nums_forgery)):  # 第几组
        for j in range(1,len(nums_forgery[i])):  # 第几个
            print("len",j,len(nums_forgery[i][j]))
            frame_scores.extend(nums_forgery[i][j])
            video_scores.append(np.max(np.array(nums_forgery[i][j])))
           # print(forged_info[j],len(nums_forgery[i][j]))
            frame_labels.extend([0 for j in range(forged_info[j][0])]+[1 for j in range(forged_info[j][0],forged_info[j][1])]+[0 for j in range(forged_info[j][1],len(nums_forgery[i][j]))])
            video_labels.append(1)

    for i in range(len(nums_compression)):
        for j in range(1,len(nums_compression[i])):
            frame_scores.extend(nums_compression[i][j])
            video_scores.append(np.max(np.array(nums_compression[i][j])))

            frame_labels.extend([0 for k in range(len(nums_compression[i][j]))])
            video_labels.append(0)
    print("video_scores",video_scores)
    print("Get info for videolevel")
    get5Metrics(video_labels,video_scores,'videolevel')
    print("Get info for framelevel")
    get5Metrics(frame_labels,frame_scores,'framelevel')








def hash_compare(video_path1,video_path2):
    H_2D_1,V_2D_1,T_2D_1=np.load(video_path1,allow_pickle=True)
    H_2D_2, V_2D_2, T_2D_2 = np.load(video_path2, allow_pickle=True)
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

    print("Dq",len(H_2D_1),len(Dq),Dq)
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
            if 'original' in file and "_compared_perceptualhash_v2" in file:
                original_videos[i].append(video_path)
            elif 'forged' in file and "_compared_perceptualhash_v2" in file:
                forged_videos[i].append(video_path)
    print(forged_videos[1])
    hds_forgery=[]
    for i in range(len(original_videos)):
        for j in range(i,len(forged_videos)):
            sub_hds_forgery = []
            for k in range(len(original_videos[i])):
              #  print(i,k,j,len(original_videos),len(original_videos[0]),len(forged_videos),len(forged_videos[0]))
                print(original_videos[i][k], forged_videos[j][k])
                sub_hds_forgery.append(hash_compare( original_videos[i][k], forged_videos[j][k]))
              #  print("sub_hds_forgery",sub_hds_forgery)
            hds_forgery.append(sub_hds_forgery)

    hds_compression=[]
    for i in range(len(original_videos)-1):
        for j in range(i+1,len(original_videos)):
            sub_hds_compression = []
            for k in range(len(original_videos[i])):
                print(original_videos[i][k], original_videos[j][k])
                sub_hds_compression.append(hash_compare( original_videos[i][k], original_videos[j][k]))
            hds_compression.append(sub_hds_compression)
    return hds_forgery,hds_compression

hds_forgery,hds_compression=traverseVideo()

np.save('data/dataset1_authen_sample_frame.npy',[hds_forgery,hds_compression])
data_processing(hds_forgery,hds_compression)