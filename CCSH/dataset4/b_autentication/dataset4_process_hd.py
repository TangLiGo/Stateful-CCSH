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




def self_filter(inputs,high_threshold,low_threshold):
    median_value=np.median(inputs)
    for index,temp in enumerate(inputs):
        if median_value>high_threshold:
            break
        if temp-median_value<low_threshold:
            inputs[index]=0
    return inputs

def forged_frame_statistic(forged_frame_info,frame_len,window_size,inputs):
    forged_frame_detected=[]
    for index, value in enumerate(inputs):
        if value>0:
            forged_frame_detected.extend(range(index*window_size-int(window_size/2),index*window_size+int(window_size/2)))

    true_positives=0
    false_positives=0

    for index in forged_frame_detected:
        if index >= 0 and index < frame_len:
            #if index < forged_frame_info[0] - 1 - int(window_size / 2) or index > forged_frame_info[1] - 1 + int(
            #        window_size / 2):  # 正确帧被判断为篡改帧的情况
            if index <  forged_frame_info[0] - 1 or index > forged_frame_info[1] - 1:  # 正确帧被判断为篡改帧的情况
                false_positives += 1
            else:  # 篡改帧被检测到的情况
                true_positives += 1


    false_negatives=forged_frame_info[1]-forged_frame_info[0]+1-true_positives

    return true_positives,false_positives,false_negatives

def false_negatives_statistic(window_size,results):

    false_frames=[]
    for index, value in enumerate(results):
        if value>0:
            false_frames.extend(
                range(index * window_size - int(window_size / 2), index * window_size + int(window_size / 2)))
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
def data_processing(nums_forgery,nums_compression,hash_len,high_threshold,hash_name):
    video_len = [840, 933, 858, 908, 359, 479, 736, 698, 666, 477,627,491,701,394,679,448]
    forged_info = [[500, 662], [551, 932], [495, 858], [473, 903], [317, 359], [1, 155], [511, 710], [596, 685], [383, 654],
                   [294, 373],[410,627],[328,491],[29,239],[250,348],[531,659],[1,149]]  # 954
    forged_video_num = 0
    compressed_video_num = 0
    forged_frame_num = 0
    forged_video_frame_num = 0  # The number of forged videos' frames
    compressed_video_frame_num = 0  # The number of original videos' frames
    for i in range(len(nums_forgery)):  # 第几组
        for j in range(len(nums_forgery[i])):  # 第几个
          #  if j==3 or j==14 or j==15:
           #     continue
            forged_video_num += 1
            forged_video_frame_num += video_len[j]
            forged_frame_num += forged_info[j][1] - forged_info[j][0] + 1
    for i in range(len(nums_compression)):
        for j in range(len(nums_compression[i])):
            compressed_video_num += 1
    low_thresholds = np.arange(0, hash_len, 0.05)
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
                result = self_filter(filter_result, high_threshold, low_threshold)

                tp, fp, fn = forged_frame_statistic(forged_info[j], video_len[j], window_size, result)
                tp_framelevel += tp
                fp_framelevel += fp
                fn_framelevel += fn
                if tp+fp>0:
                    tp_videolevel+=1
        for i in range(len(nums_compression)):
            for j in range(len(nums_compression[i])):
                num = np.array(nums_compression[i][j])
                filter_result = SlidingAverage(np.array(num).copy(), window_size)
                result = self_filter(filter_result, high_threshold, low_threshold)
                if false_negatives_statistic(window_size, result) > 0:
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

    print("frame level")
    img_path="../result/framelevel_"+hash_name+".pdf"
    precision_framelevel,recall_framelevel,F1_framelevel,iou_framelevel=benchmarks_process(tps_framelevel,fps_framelevel,fns_framelevel,low_thresholds,img_path)
    print("video level")
    img_path = "../result/videolevel_" + hash_name + ".pdf"
    precision_videolevel,recall_videolevel,F1_videolevel,iou_videolevel=benchmarks_process(tps_videolevel, fps_videolevel, fns_videolevel, low_thresholds, img_path)

    return [fps_framelevel_probability, tps_framelevel_probability,precision_framelevel,recall_framelevel,F1_framelevel,iou_framelevel],[fps_videolevel_probability,tps_videolevel_probability,precision_videolevel,recall_videolevel,F1_videolevel,iou_videolevel]





ahash_samples_path=[]
ahash_samples_insertion_path=[]
ahash_samples_compression_path=[]
ahash_samples_insertion_compression_path=[]
data_path="../data/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'hds_ahash_sample_advanced' in file :

            if "compression" in file:
                ahash_samples_compression_path.append(video_path)
            else:
                ahash_samples_path.append(video_path)

hds_ahash = np.load('../data/hds_ahash.npy', allow_pickle=True)
hds_ahash_compression = np.load('../data/hds_ahash_compression.npy', allow_pickle=True)

hds_ahash_samples=[]
hds_ahash_samples_compression=[]

sample_names=[]
sample_names_insertion=[]
for path in ahash_samples_path:
    hds_ahash_samples.append(np.load(path, allow_pickle=True))
    (subpath, temp_file_name) = os.path.split(path)
    (output_file_name,extension)=os.path.splitext(temp_file_name)
    sample_names.append(output_file_name)
for path in ahash_samples_compression_path:
    hds_ahash_samples_compression.append(np.load(path, allow_pickle=True))

#hds_ahash_sample = np.load('../../data/hds_ahash_sample.npy', allow_pickle=True)
#hds_ahash_sample_compression = np.load('../../data/hds_ahash_sample_compression.npy', allow_pickle=True)

print(ahash_samples_path)
print(ahash_samples_compression_path)

sample_nums = range(6)

hash_lens=[[16,16,16,16,16],[14,5,5,5],[21,9,9,9]]
high_threshold=[[5,5,5,5,5],[8,2.5,2.5,2.5],[5,2,2,2]]
hash_names=[["ahash","ahash_sample","ahash_sample_advanced","ahash_sample_similarity_area","ahash_sample_similarity_block"],["videohash","videohash_sample","videohash_sample_similarity_area","videohash_sample_similarity_block"],["blockhash_even","blockhash_sample","blockhash_sample_similarity_area","blockhash_sample_similarity_block"]]
hds=[[hds_ahash,hds_ahash_samples]]
hds_compression=[[hds_ahash_compression,hds_ahash_samples_compression]]


ahash_framelevel,ahash_videolevel=data_processing(hds_ahash ,hds_ahash_compression,16, 16,'ahash')
fpr_ahash_framelevel,tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel=ahash_framelevel
fpr_ahash_videolevel,tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel=ahash_videolevel

#ahash_sample_framelevel,ahash_sample_videolevel=data_processing(hds_ahash_sample ,hds_ahash_sample_compression,16, 16,'ahash_sample')
#fpr_ahash_sample_framelevel,tpr_ahash_sample_framelevel,precision_ahash_sample_framelevel,recall_ahash_sample_framelevel,F1_ahash_sample_framelevel,iou_ahash_sample_framelevel=ahash_sample_framelevel
#fpr_ahash_sample_videolevel,tpr_ahash_sample_videolevel,precision_ahash_sample_videolevel,recall_ahash_sample_videolevel,F1_ahash_sample_videolevel,iou_ahash_sample_videolevel=ahash_sample_videolevel

np.save('../data/roc_videolevel_data_ahash.npy', np.array([fpr_ahash_videolevel, tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel],dtype=object))
np.save('../data/roc_framelevel_data_ahash.npy', np.array([fpr_ahash_framelevel, tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel],dtype=object))

# np.save('../../data/roc_videolevel_data_ahash_sample.npy', np.array([fpr_ahash_sample_videolevel, tpr_ahash_sample_videolevel],dtype=object))
#np.save('../../data/roc_framelevel_data_ahash_sample.npy', np.array([fpr_ahash_sample_framelevel, tpr_ahash_sample_framelevel],dtype=object))

fpr_framelevel = []
tpr_framelevel = []
precision_framelevel=[]
recall_framelevel=[]
F1_framelevel=[]
iou_framelevel=[]


fpr_videolevel = []
tpr_videolevel = []
precision_videolevel=[]
recall_videolevel=[]
F1_videolevel=[]
iou_videolevel=[]

for i in range(len(hds_ahash_samples)):

    print("processing",sample_names[i])
    framelevel_data,videolevel_data= data_processing(hds_ahash_samples[i],hds_ahash_samples_compression[i], 16, 16,
                                         sample_names[i])
    fpr_framelevel_temp, tpr_framelevel_temp, precision_framelevel_temp, recall_framelevel_temp, F1_framelevel_temp, iou_framelevel_temp=framelevel_data
    fpr_videolevel_temp, tpr_videolevel_temp, precision_videolevel_temp, recall_videolevel_temp, F1_videolevel_temp, iou_videolevel_temp=videolevel_data
    if i<10:
        np.save('../data/roc_videolevel_0' + str(i) + '_deletion.npy', np.array(
            [fpr_videolevel_temp, tpr_videolevel_temp, precision_videolevel_temp, recall_videolevel_temp,
             F1_videolevel_temp, iou_videolevel_temp], dtype=object))
        np.save('../data/roc_framelevel_0' + str(i) + '_deletion.npy', np.array(
            [fpr_framelevel_temp, tpr_framelevel_temp, precision_framelevel_temp, recall_framelevel_temp,
             F1_framelevel_temp, iou_framelevel_temp], dtype=object))
    else:
        np.save('../data/roc_videolevel_' + str(i) + '_deletion.npy', np.array(
            [fpr_videolevel_temp, tpr_videolevel_temp, precision_videolevel_temp, recall_videolevel_temp,
             F1_videolevel_temp, iou_videolevel_temp], dtype=object))
        np.save('../data/roc_framelevel_' + str(i) + '_deletion.npy', np.array(
            [fpr_framelevel_temp, tpr_framelevel_temp, precision_framelevel_temp, recall_framelevel_temp,
             F1_framelevel_temp, iou_framelevel_temp], dtype=object))




    fpr_framelevel.append(fpr_framelevel_temp)
    tpr_framelevel.append(tpr_framelevel_temp)
    precision_framelevel.append(precision_framelevel_temp)
    recall_framelevel.append(recall_framelevel_temp)
    F1_framelevel.append(F1_framelevel_temp)
    iou_framelevel.append(iou_framelevel_temp)


    fpr_videolevel.append(fpr_videolevel_temp)
    tpr_videolevel.append(tpr_videolevel_temp)
    precision_videolevel.append(precision_videolevel_temp)
    recall_videolevel.append(recall_videolevel_temp)
    F1_videolevel.append(F1_videolevel_temp)
    iou_videolevel.append(iou_videolevel_temp)


    pl.figure()
    pl.plot(fpr_ahash_framelevel,tpr_ahash_framelevel,label='ahash')
    pl.plot( fpr_framelevel[0], tpr_framelevel[0], label='baseline')
    pl.plot(fpr_framelevel_temp, tpr_framelevel_temp, label='CCSHG')
  #  pl.plot(fpr_framelevel_insertion_temp, tpr_framelevel_insertion_temp, label='CCSHG-insertion')
    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("../result/d1_roc_fl_"+str(i)+".pdf")

    pl.figure()
    pl.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash')
    pl.plot(fpr_videolevel[0],  tpr_videolevel[0],  label='baseline')
    pl.plot(fpr_videolevel_temp, tpr_videolevel_temp, label='CCSHG')
  #  pl.plot(fpr_videolevel_insertion_temp, tpr_videolevel_insertion_temp, label='CCSHG-insertion')
    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("../result/d1_roc_vl_"+str(i)+".pdf")

