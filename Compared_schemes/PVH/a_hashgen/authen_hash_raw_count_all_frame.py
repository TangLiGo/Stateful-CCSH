import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import math
'''
算术平均滤波法
'''
import os
from matplotlib.pyplot import MultipleLocator

def SlidingAverage(inputs,per):
	if np.shape(inputs)[0] % per != 0:
		lengh = np.shape(inputs)[0] / per
		for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
			inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
	inputs = inputs.reshape((-1,per))
	tmpmean = inputs[0].mean()
	mean = []
	for tmp in inputs:
		mean.append((tmpmean+tmp.mean())/2)
		tmpmean = tmp.mean()
	return mean



def self_filter(inputs,low_threshold):

    for index,temp in enumerate(inputs):

        if temp<low_threshold:
            inputs[index]=0
    # median_value=np.median(inputs)
    # for index,temp in enumerate(inputs):
    #
    #     if temp-median_value<low_threshold:
    #         inputs[index]=0
    return inputs

def forged_frame_statistic(forged_frame_info,inputs):
    forged_frame_detected=[]
    for index, value in enumerate(inputs):
        if value>0:
            forged_frame_detected.append(index)
    true_positives=0
    false_positives=0

    for index in forged_frame_detected:
        if index < forged_frame_info[0] - 1 or index > forged_frame_info[1] - 1:  # 正确帧被判断为篡改帧的情况
            false_positives += 1
        else:  # 篡改帧被检测到的情况
            true_positives += 1

    false_negatives=forged_frame_info[1]-forged_frame_info[0]+1-true_positives

    return true_positives,false_positives,false_negatives

def false_negatives_statistic(results):

    false_frames=[]
    for index, value in enumerate(results):
        if value>0:
            false_frames.append(index)
    false_negatives=len(false_frames)
    return false_negatives

def benchmarks_process(tps,fps,fns,low_thresholds,img_path):
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
    precision_best=round(precision[best_threshold_index],4)
    recall_best=round(recall[best_threshold_index],4)
    F1_best=round(F1[best_threshold_index],4)
    iou_best=round(iou[best_threshold_index],4)
    print("best_threshold=", best_threshold)
    print("Precision={:.2%}".format(precision[best_threshold_index]))
    print("Recall={:.2%}".format(recall[best_threshold_index]))
    print("F1={:.2%}".format(F1[best_threshold_index]))
    print("IoU={:.2%}".format(iou[best_threshold_index]))

    pl.title("benchmarks")
    pl.figure()
    pl.plot(low_thresholds, precision, label="precision")
    pl.plot(low_thresholds, recall, label="recall")
    pl.plot(low_thresholds, F1, label="F1")
    pl.plot(low_thresholds, iou, label="iou")
    plt.legend()
    pl.savefig(img_path)
    return precision_best,recall_best,F1_best,iou_best
def get_frame_info(hd,segment_size,video_len):
    output=[]#[-1 for i in range(video_len)]
    label=[]

    for i in range(len(hd)):
        output.extend([hd[i] for j in range(segment_size)])
    output.extend([hd[-1] for j in range(video_len%segment_size)])
  #  print(len(output),video_len)
    return output
def data_processing(nums_forgery,nums_compression,dataset):
    if dataset==1:
        video_len = [210, 329, 313, 319, 583, 262, 412, 274, 554, 239]
        forged_info = [[1, 210], [120, 162], [233, 313], [61, 122], [1, 190], [211, 262], [90, 117], [133, 160],
                       [169, 359],
                       [104, 149]]
        start=1
    elif dataset==4:
        video_len = [840, 933, 858, 908, 359, 479, 736, 698, 666, 477, 627, 491, 701, 394, 679, 448]
        forged_info = [[500, 662], [551, 932], [495, 858], [473, 903], [317, 359], [1, 155], [511, 710], [596, 685],
                       [383, 654],
                       [294, 373], [410, 627], [328, 491], [29, 239], [250, 348], [531, 659], [1, 149]]
        start=0
    forged_video_num = 0
    compressed_video_num = 0
    forged_frame_num = 0
    forged_video_frame_num = 0  # The number of forged videos' frames
    compressed_video_frame_num = 0  # The number of original videos' frames
    for i in range(len(nums_forgery)):  # 第几组
        for j in range(start,len(nums_forgery[i])):  # 第几个
          #  if j==3 or j==14 or j==15:
           #     continue
            forged_video_num += 1
            forged_video_frame_num += video_len[j]
            forged_frame_num += forged_info[j][1] - forged_info[j][0] + 1
    for i in range(len(nums_compression)):
        for j in range(len(nums_compression[i])):
            compressed_video_num += 1
            forged_video_frame_num += video_len[j]
    low_thresholds = np.arange(0, 100, 1)
  #  print("low_thresholds",np.max(nums_forgery), low_thresholds)

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
    segment_size=5
    for low_threshold in low_thresholds:
        tp_framelevel = 0
        fp_framelevel = 0
        fn_framelevel = 0
        tp_videolevel = 0
        fp_videolevel = 0
        fn_videolevel = 0
        for i in range(len(nums_forgery)):
            for j in range(start,len(nums_forgery[i])):
             #   if j == 3 or j == 14 or j == 15:
               #     continue
                num =get_frame_info(nums_forgery[i][j],segment_size,video_len[j])
                #print(len(nums_forgery[i][j]))
              #  filter_result = SlidingAverage(np.array(num).copy(), window_size)
                result = self_filter(num, low_threshold)
                # plt.figure()
                # plt.plot(num)
                # plt.show()
                tp, fp, fn = forged_frame_statistic(forged_info[j], result)
                tp_framelevel += tp
                fp_framelevel += fp
                fn_framelevel += fn
                if tp+fp>0:
                    tp_videolevel+=1
        for i in range(len(nums_compression)):
            for j in range(len(nums_compression[i])):
                num = get_frame_info(nums_compression[i][j],segment_size,video_len[j])
               # filter_result = SlidingAverage(np.array(num).copy(), window_size)
                result = self_filter(num,  low_threshold)
                # plt.figure()
                # plt.plot(num)
                # plt.show()
                tt=false_negatives_statistic(result)
                fp_framelevel+=tt
                if  tt> 0:
                    fp_videolevel += 1
        tps_framelevel.append(tp_framelevel)
        fns_framelevel.append(fn_framelevel)
        fps_framelevel.append(fp_framelevel)
        tps_framelevel_probability.append(tp_framelevel / forged_frame_num)
        fps_framelevel_probability.append(fp_framelevel / (forged_video_frame_num - forged_frame_num))

        tps_videolevel.append(tp_videolevel)
        fns_videolevel.append(forged_video_num - tp_videolevel)
        fps_videolevel.append(fp_videolevel)
        tps_videolevel_probability.append(tp_videolevel / forged_video_num)
        fps_videolevel_probability.append(fp_videolevel / compressed_video_num)

    print("video level")
    img_path="figs/dataset"+str(dataset)+"_videolevel.pdf"
    precision_videolevel,recall_videolevel,F1_videolevel,iou_videolevel=benchmarks_process(tps_videolevel, fps_videolevel, fns_videolevel, low_thresholds, img_path)

    print("frame level")
    img_path="figs/dataset"+str(dataset)+"_framelevel.pdf"
    precision_framelevel,recall_framelevel,F1_framelevel,iou_framelevel=benchmarks_process(tps_framelevel,fps_framelevel,fns_framelevel,low_thresholds,img_path)

    return [fps_framelevel_probability, tps_framelevel_probability,precision_framelevel,recall_framelevel,F1_framelevel,iou_framelevel],[fps_videolevel_probability,tps_videolevel_probability,precision_videolevel,recall_videolevel,F1_videolevel,iou_videolevel]


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
    segment_size=5
    Q=int(K/segment_size)
    Arr1=[[[-1 for j in range(N)] for i in range(M)] for q in range(Q)]
    for q in range(Q):
        for i in range(M):
            for j in range(N):
                su=0
                for k in range((q-1)*segment_size,q*segment_size):
                    su+=Arr[k][i][j]

                if su==segment_size:
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
            if ('Real' in file or 'original' in file) and "compared_all_frame_perceptualhash.npy" in file:
                original_videos[i].append(video_path)
            elif ('Forged' in file or 'forged' in file)and "compared_all_frame_perceptualhash.npy" in file:
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
#np.save('data/dataset1_PVH_hash_distances.npy',[hds_forgery,hds_compression])
#hds_forgery,hds_compression=np.load('data/dataset1_TD2n_hash_distances.npy',allow_pickle=True)

data_processing(hds_forgery,hds_compression,dataset=1)

hds_forgery,hds_compression=traverseVideo(dataset=4)
#np.save('data/dataset4_PVH_hash_distances.npy',[hds_forgery,hds_compression])
#hds_forgery,hds_compression=np.load('data/dataset4_TD2n_hash_distances.npy',allow_pickle=True)
print("Authenticate for dataset 4")
data_processing(hds_forgery,hds_compression,dataset=4)