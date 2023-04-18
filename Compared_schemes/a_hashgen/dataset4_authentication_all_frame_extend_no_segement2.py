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


def SlidingAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


def self_filter(inputs, low_threshold):
    median_value = np.median(inputs)
    for index, temp in enumerate(inputs):

        if temp - median_value < low_threshold:
            inputs[index] = 0
    return inputs


def forged_frame_statistic(forged_frame_info, frame_len, window_size, inputs):
    forged_frame_detected = []
    for index, value in enumerate(inputs):
        if value > 0:
            forged_frame_detected.extend(
                range(index * window_size - int(window_size / 2), index * window_size + int(window_size / 2)))

    true_positives = 0
    false_positives = 0

    for index in forged_frame_detected:
        if index >= 0 and index < frame_len:
            # if index < forged_frame_info[0] - 1 - int(window_size / 2) or index > forged_frame_info[1] - 1 + int(
            #        window_size / 2):  # 正确帧被判断为篡改帧的情况
            if index < forged_frame_info[0] - 1 or index > forged_frame_info[1] - 1:  # 正确帧被判断为篡改帧的情况
                false_positives += 1
            else:  # 篡改帧被检测到的情况
                true_positives += 1

    false_negatives = forged_frame_info[1] - forged_frame_info[0] + 1 - true_positives

    return true_positives, false_positives, false_negatives


def false_negatives_statistic(window_size, results, frame_n):
    false_frames = []
    for index, value in enumerate(results):
        if value > 0:
            false_frames.extend(
                range(max(0, index * window_size - int(window_size / 2)),
                      min(frame_n, index * window_size + int(window_size / 2))))
    false_negatives = len(false_frames)
    return false_negatives





def getBestThreshold(tps, fps, fns, low_thresholds):
    precision = []
    recall = []
    F1 = []
    iou = []
    for k in range(len(tps)):
        if tps[k] == 0:
            precision.append(0)
            recall.append(0)
            F1.append(0)
            iou.append(0)
        else:
            precision.append(tps[k] / (tps[k] + fps[k]))
            recall.append(tps[k] / (tps[k] + fns[k]))
            F1.append(2 * tps[k] / (2 * tps[k] + fps[k] + fns[k]))
            iou.append(tps[k] / (tps[k] + fps[k] + fns[k]))
    best_threshold_index = F1.index(max(F1))
    best_threshold = low_thresholds[best_threshold_index]
    boundline = max(F1) * 0.97
    range_index = []
    # for i in range(len(F1)):
    #  if F1[i]>boundline and range_index==[]:
    #     range_index.append(i)
    #  elif F1[i]<boundline and len(range_index)==1:
    #      range_index.append(i)
    for i in range(len(F1)):
        if F1[i] > boundline:
            range_index.append(i)
            break
    for i in range(len(F1)):
        if F1[len(F1) - 1 - i] > boundline:
            range_index.append(len(F1) - 1 - i)
            break
    best_range = [low_thresholds[range_index[0]], low_thresholds[range_index[1]]]

    return best_threshold_index, [range_index[0], range_index[1]]


def data_processing(nums_forgery, nums_compression):
    video_len = [840, 933, 858, 908, 359, 479, 736, 698, 666, 477, 627, 491, 701, 394, 679, 448]
    forged_info = [[500, 662], [551, 932], [495, 858], [473, 903], [317, 359], [1, 155], [511, 710], [596, 685],
                   [383, 654],
                   [294, 373], [410, 627], [328, 491], [29, 239], [250, 348], [531, 659], [1, 149]]  # 954
    forged_video_num = 0
    compressed_video_num = 0
    forged_frame_num = 0
    forged_video_frame_num = 0  # The number of forged videos' frames
    compressed_video_frame_num = 0  # The number of original videos' frames

    benchmarks_train_revised_fl_data = 1
    for i in range(len(nums_forgery)):  # 第几组
        for j in range(len(nums_forgery[i])):  # 第几个
            #  if j==3 or j==14 or j==15:
            #     continue
            forged_video_num += 1
            forged_video_frame_num += video_len[j]
            forged_frame_num += forged_info[j][1] - forged_info[j][0] + 1
   # print(forged_frame_num)
    for i in range(len(nums_compression)):
        for j in range(len(nums_compression[i])):
            compressed_video_num += 1
            compressed_video_frame_num += video_len[j]

    window_size = 10
    tps_framelevel = []  # The matrix to store the num of correctly detected forged videos       true positives & judged positives
    fps_framelevel = []  # The matrix to store the num of falsely judged forged videos          true negatives & judged positives
    fns_framelevel = []  # The matrix to store the num of falsely judged tampered videos        true positives & judged negatives
    tps_framelevel_probability = []
    fps_framelevel_probability = []

    tps_videolevel = []  # The matrix to store the num of correctly detected forged videos       true positives & judged positives
    fps_videolevel = []  # The matrix to store the num of falsely judged forged videos          true negatives & judged positives
    fns_videolevel = []  # The matrix to store the num of falsely judged tampered videos        true positives & judged negatives
    tps_videolevel_probability = []
    fps_videolevel_probability = []
    low_thresholds=np.arange(0,250,1)
    for low_threshold in low_thresholds:
        tp_framelevel = 0
        fp_framelevel = 0
        fn_framelevel = 0
        tp_videolevel = 0
        fp_videolevel = 0
        fn_videolevel = 0
        for i in range(len(nums_forgery)):
            for j in range(len(nums_forgery[i])):
                #   if j == 3 or j == 14 or j == 15:
                #     continue
                num = np.array(nums_forgery[i][j])
                filter_result = SlidingAverage(np.array(num).copy(), window_size)
                result = self_filter(filter_result, low_threshold)

                tp, fp, fn = forged_frame_statistic(forged_info[j], video_len[j], window_size, result)
                tp_framelevel += tp
                fp_framelevel += fp
                fn_framelevel += fn
                if tp + fp > 0:
                    tp_videolevel += 1
        for i in range(len(nums_compression)):
            for j in range(len(nums_compression[i])):
                frame_n = len(nums_compression[i][j])
                num = np.array(nums_compression[i][j])
                filter_result = SlidingAverage(np.array(num).copy(), window_size)
                result = self_filter(filter_result, low_threshold)
                tt = false_negatives_statistic(window_size, result, frame_n)
                fp_framelevel += tt
                if tt > 0:
                    fp_videolevel += 1
        tps_framelevel.append(tp_framelevel)
        fns_framelevel.append(fn_framelevel)
        fps_framelevel.append(fp_framelevel)
        tps_framelevel_probability.append(tp_framelevel / forged_frame_num)
        fps_framelevel_probability.append(
            fp_framelevel / (compressed_video_frame_num + forged_video_frame_num - forged_frame_num))

        tps_videolevel.append(tp_videolevel)
        fns_videolevel.append(forged_video_num - tp_videolevel)
        fps_videolevel.append(fp_videolevel)
        tps_videolevel_probability.append(tp_videolevel / forged_video_num)
        fps_videolevel_probability.append(fp_videolevel / compressed_video_num)

    optimal_th, optimal_index = Find_Optimal_Cutoff(TPR=tps_videolevel_probability, FPR=fps_videolevel_probability, threshold=low_thresholds)

    print("best threshold",optimal_th)
    precision, recall, F1, iou = getBenchmarks(tps_videolevel[optimal_index], fps_videolevel[optimal_index], fns_videolevel[optimal_index])


    optimal_th, optimal_index = Find_Optimal_Cutoff(TPR=tps_framelevel_probability, FPR=fps_framelevel_probability, threshold=low_thresholds)

    print("best threshold",optimal_th)
    precision, recall, F1, iou = getBenchmarks(tps_framelevel[optimal_index], fps_framelevel[optimal_index], fns_framelevel[optimal_index])


    return [fps_framelevel_probability, tps_framelevel_probability, tps_framelevel, fps_framelevel, fns_framelevel], [
        fps_videolevel_probability, tps_videolevel_probability, tps_videolevel, fps_videolevel, fns_videolevel]


def getBenchmarks(tp, fp, fn):

    precision=(tp / (tp + fp))
    recall=tp/ (tp + fn)
    F1=2 * tp / (2 * tp + fp + fn)
    iou=tp/ (tp + fp + fn)


    print("Precision={:.2%}".format(precision))
    print("Recall={:.2%}".format(recall))
    print("F1={:.2%}".format(F1))
    print("IoU={:.2%}".format(iou))


    return precision, recall, F1, iou
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
def get_frame_info(hd,segment_size,video_len):
    output=[]#[-1 for i in range(video_len)]
    label=[]

    for i in range(len(hd)):
        output.extend([hd[i] for j in range(segment_size)])
    output.extend([hd[-1] for j in range(video_len%segment_size)])
  #  print(len(output),video_len)
    return output












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
    s=1
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

    print(len(Dq))
    return Dq



hds_forgery,hds_compression=np.load('data/dataset4_authen_all_frame_no_segment.npy',allow_pickle=True)
data_processing(hds_forgery,hds_compression)