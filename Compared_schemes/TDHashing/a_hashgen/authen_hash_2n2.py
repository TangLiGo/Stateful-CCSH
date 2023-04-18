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
  #  print("Accuracy={:.2%}".format(accuracy))

    return precision, recall, F1, iou, accuracy
def drawRoc(tpr,fpr,figname):
    plt.figure()
    plt.plot(fpr,tpr)
    plt.savefig(figname)
def get5Metrics(labels, scores,level):
    AUC = getAuc(labels, scores)

    print("AUC:", AUC)

    thresholds = np.arange(0, 4, 0.1)
    print("thresholds",thresholds)
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
   # precision, recall, F1, iou, accuracy = getBenchmarks(tps[optimal_index], fps[optimal_index], fns[optimal_index], tns[optimal_index])
    drawBenchmarks(tps,fps,fns,tns,thresholds,'figs/benchmark_'+level+'.png')
    print("reach video level threshold")
    getBenchmarks(tps[-9], fps[-9], fns[-9], tns[-9])
    figname='figs/dataset1_roc_'+level+".png"
    drawRoc(tpr,fpr,figname)
    #Precision=57.00%
#Recall=82.13%
#F1=67.29%
#IoU=50.71%

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
def data_processing(nums_forgery,nums_compression,dataset):
    if dataset==1:
        video_len = [210, 329, 313, 319, 583, 262, 412, 274, 554, 239]
        forged_info = [[1, 210], [120, 162], [233, 313], [61, 122], [1, 190], [211, 262], [90, 117], [133, 160],
                       [169, 359],
                       [104, 149]]
    elif dataset==4:
        video_len = [840, 933, 858, 908, 359, 479, 736, 698, 666, 477, 627, 491, 701, 394, 679, 448]
        forged_info = [[500, 662], [551, 932], [495, 858], [473, 903], [317, 359], [1, 155], [511, 710], [596, 685],
                       [383, 654],
                       [294, 373], [410, 627], [328, 491], [29, 239], [250, 348], [531, 659], [1, 149]]


   # forged_info = [[100, 133], [111, 187], [99, 172], [95, 181], [64, 72], [1, 32], [103, 143], [120, 138], [77, 131],
              #     [59, 75], [82, 126], [66, 99], [6, 48], [50, 70], [107, 132], [1, 30]]


    frame_labels=[]
    frame_scores=[]
    video_scores=[]
    video_labels=[]


    for i in range(len(nums_forgery)):  # 第几组
        for j in range(len(nums_forgery[i])):  # 第几个
          #  print("len",j,len(nums_forgery[i][j]))
          #   plt.figure()
          #   plt.plot(nums_forgery[i][j])
          #   plt.show()
            frame_scores.extend(nums_forgery[i][j])
            video_scores.append(np.max(np.array(nums_forgery[i][j])))
         #   print(forged_info[j],len(nums_forgery[i][j]))
            frame_labels.extend([0 for j in range(forged_info[j][0])]+[1 for j in range(forged_info[j][0],forged_info[j][1])]+[0 for j in range(forged_info[j][1],len(nums_forgery[i][j]))])
            video_labels.append(1)

    for i in range(len(nums_compression)):
        for j in range(len(nums_compression[i])):
            frame_scores.extend(nums_compression[i][j])
            video_scores.append(np.max(np.array(nums_compression[i][j])))
            # plt.figure()
            # plt.plot(nums_compression[i][j])
            # plt.show()
            frame_labels.extend([0 for k in range(len(nums_compression[i][j]))])
            video_labels.append(0)
   # print("video_scores",video_scores)
    print("Get info for videolevel")
    get5Metrics(video_labels,video_scores,'videolevel')
    print("Get info for framelevel")
    get5Metrics(frame_labels,frame_scores,'framelevel')

def hash_compare(video_path1,video_path2):
    H_set1=np.load(video_path1,allow_pickle=True)
    H_set2 = np.load(video_path2, allow_pickle=True)
    D=[]
    for i in range(len(H_set1)):
        D.append(np.count_nonzero(H_set1[i]!=H_set2[i]))
    return D
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
            if ('Real' in file or 'original' in file) and "_compared_TD2n.npy" in file:
                original_videos[i].append(video_path)
            elif ('Forged' in file or 'forged' in file)and "_compared_TD2n.npy" in file:
                forged_videos[i].append(video_path)
  #  print(forged_videos[1])
    hds_forgery=[]
    for i in range(len(original_videos)):
        for j in range(i,len(forged_videos)):
            sub_hds_forgery = []
            for k in range(len(original_videos[i])):
              #  print(i,k,j,len(original_videos),len(original_videos[0]),len(forged_videos),len(forged_videos[0]))
               # print(original_videos[i][k], forged_videos[j][k])
                sub_hds_forgery.append(hash_compare( original_videos[i][k], forged_videos[j][k]))
              #  print("sub_hds_forgery",sub_hds_forgery)
            hds_forgery.append(sub_hds_forgery)

    hds_compression=[]
    for i in range(len(original_videos)-1):
        for j in range(i+1,len(original_videos)):
            sub_hds_compression = []
            for k in range(len(original_videos[i])):
              #  print(original_videos[i][k], original_videos[j][k])
                sub_hds_compression.append(hash_compare( original_videos[i][k], original_videos[j][k]))
            hds_compression.append(sub_hds_compression)
    return hds_forgery,hds_compression
print("Authenticate for dataset 1")
hds_forgery,hds_compression=traverseVideo(dataset=1)
np.save('data/dataset1_TD2n_hash_distances.npy',[hds_forgery,hds_compression])
#hds_forgery,hds_compression=np.load('data/dataset1_TD2n_hash_distances.npy',allow_pickle=True)

data_processing(hds_forgery,hds_compression,dataset=1)

hds_forgery,hds_compression=traverseVideo(dataset=4)
np.save('data/dataset4_TD2n_hash_distances.npy',[hds_forgery,hds_compression])
#hds_forgery,hds_compression=np.load('data/dataset4_TD2n_hash_distances.npy',allow_pickle=True)
print("Authenticate for dataset 4")
data_processing(hds_forgery,hds_compression,dataset=4)