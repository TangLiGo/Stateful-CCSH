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

def benchmarks_process(tps,fps,fns,low_thresholds,img_name):
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
    boundline = max(F1) * 0.99
    range_index = []

    for i in range(len(F1)):
        if F1[i] > boundline:
            range_index.append(i)
            break
    for i in range(len(F1)):
        if F1[len(F1) - 1 - i] > boundline:
            range_index.append(len(F1) - 1 - i)
            break
    best_range=[low_thresholds[range_index[0]], low_thresholds[range_index[1]]]
    print("The best th", best_range)
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
    pl.vlines(low_thresholds[range_index[0]], 0, 1, color='black', linestyles='dashed', label='bound')
    pl.vlines(low_thresholds[range_index[1]], 0, 1, color='black', linestyles='dashed', label='bound')
    plt.legend()
    pl.savefig('../result/range_'+img_name+'.png')
    return precision_best,recall_best,F1_best,iou_best,best_range

def data_processing(nums_forgery,nums_compression,hash_len,high_threshold):
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


        tps_videolevel.append(tp_videolevel)
        fns_videolevel.append(forged_video_num - tp_videolevel)
        fps_videolevel.append(fp_videolevel)


    return [tps_framelevel, fps_framelevel, fns_framelevel, forged_frame_num,
            forged_video_frame_num - forged_frame_num], [tps_videolevel, fps_videolevel, fns_videolevel,
                                                         forged_video_num, compressed_video_num]


def data_read(data_path):
    ahash_samples_path = []
   
    ahash_samples_compression_path = []


    for file in os.listdir(data_path):
        video_path = os.path.join(data_path, file)
        video_path = video_path.replace('\\', '/')
        if 'hds_ahash_sample_advanced' in file:
                if "compression" in file:
                    ahash_samples_compression_path.append(video_path)
                else:
                    ahash_samples_path.append(video_path)

    print(ahash_samples_path)
    print(ahash_samples_compression_path)
    hds_ahash = np.load(data_path+'hds_ahash.npy', allow_pickle=True)
    hds_ahash_compression = np.load(data_path+'hds_ahash_compression.npy', allow_pickle=True)

    hds_ahash_samples = []
    hds_ahash_samples_compression = []

    sample_names = []
    
    for path in ahash_samples_path:
        hds_ahash_samples.append(np.load(path, allow_pickle=True))
        (subpath, temp_file_name) = os.path.split(path)
        (output_file_name, extension) = os.path.splitext(temp_file_name)
        sample_names.append(output_file_name)
    for path in ahash_samples_compression_path:
        hds_ahash_samples_compression.append(np.load(path, allow_pickle=True))
    return hds_ahash,hds_ahash_compression,hds_ahash_samples,hds_ahash_samples_compression

low_thresholds = np.arange(0, 16, 0.05)
path='../data/'

hds_ahash,hds_ahash_compression,hds_ahash_samples,hds_ahash_samples_compression=data_read(path)


ahash_framelevel,ahash_videolevel=data_processing(hds_ahash,hds_ahash_compression,16, 16)
tps_ahash_framelevel,fps_ahash_framelevel,fns_ahash_framelevel,forged_frame_num,real_frame_num=ahash_framelevel
tps_ahash_videolevel,fps_ahash_videolevel,fns_ahash_videolevel,forged_video_num,real_video_num=ahash_videolevel

precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel,range_ahash_framelevel=benchmarks_process(tps_ahash_framelevel,fps_ahash_framelevel,fns_ahash_framelevel,low_thresholds,'fl_ahash')
precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel,range_ahash_videolevel=benchmarks_process(tps_ahash_videolevel,fps_ahash_videolevel,fns_ahash_videolevel,low_thresholds,'vl_ahash')
tpr_ahash_videolevel=[round(x/(forged_video_num),4) for x in tps_ahash_videolevel]
fpr_ahash_videolevel=[round(x/(real_video_num),4) for x in fps_ahash_videolevel]

tpr_ahash_framelevel=[round(x/(forged_frame_num),4) for x in tps_ahash_framelevel]
fpr_ahash_framelevel=[round(x/(real_frame_num),4) for x in fps_ahash_framelevel]
np.save('../data/benchrmarks_videolevel_data_ahash.npy', np.array([tps_ahash_videolevel,fps_ahash_videolevel,fns_ahash_videolevel,[forged_video_num + forged_video_num,real_video_num + real_video_num]],dtype=object))
np.save('../data/benchrmarks_framelevel_data_ahash.npy', np.array([tps_ahash_framelevel,fps_ahash_framelevel,fns_ahash_framelevel,[forged_frame_num + forged_frame_num,real_frame_num + real_frame_num]],dtype=object))


fpr_framelevel_set = []
tpr_framelevel_set = []
precision_framelevel_set=[]
recall_framelevel_set=[]
F1_framelevel_set=[]
iou_framelevel_set=[]

fpr_videolevel_set = []
tpr_videolevel_set = []
precision_videolevel_set=[]
recall_videolevel_set=[]
F1_videolevel_set=[]
iou_videolevel_set=[]

range_framelevel_set=[]
range_videolevel_set=[]




for i in range(len(hds_ahash_samples)):
    print("processing ",i)
    framelevel, videolevel = data_processing(hds_ahash_samples[i], hds_ahash_samples_compression[i], 16, 16,)
    tps_framelevel, fps_framelevel, fns_framelevel, forged_frame_num, real_frame_num = framelevel
    tps_videolevel, fps_videolevel, fns_videolevel, forged_video_num, real_video_num = videolevel


    precision_framelevel, recall_framelevel, F1_framelevel, iou_framelevel, range_framelevel = benchmarks_process(
        tps_framelevel, fps_framelevel, fns_framelevel, low_thresholds, 'fl_sample_advanced_'+str(i)+'')
    range_framelevel_set.append(range_framelevel)
    precision_videolevel, recall_videolevel, F1_videolevel, iou_videolevel, range_videolevel = benchmarks_process(
        tps_videolevel, fps_videolevel, fns_videolevel, low_thresholds, 'vl_sample_advanced_'+str(i)+'')
    range_videolevel_set.append(range_videolevel)
    tpr_videolevel = [round(x / (forged_video_num ), 4) for x in tps_videolevel]
    fpr_videolevel = [round(x / (real_video_num ), 4) for x in fps_videolevel]

    tpr_framelevel = [round(x / (forged_frame_num ), 4) for x in tps_framelevel]
    fpr_framelevel = [round(x / (real_frame_num ), 4) for x in fps_framelevel]
    np.save('../data/benchrmarks_videolevel_' + str(i) + '.npy', np.array(
        [tps_videolevel, fps_videolevel, fns_videolevel,[forged_video_num ,real_video_num ]], dtype=object))
    np.save('../data/benchrmarks_framelevel_' + str(i) + '.npy', np.array(
        [tps_framelevel, fps_framelevel, fns_framelevel,[forged_frame_num ,real_frame_num ]], dtype=object))
    fpr_framelevel_set.append(fpr_framelevel)
    tpr_framelevel_set.append(tpr_framelevel)
    precision_framelevel_set.append(precision_framelevel)
    recall_framelevel_set.append(recall_framelevel)
    F1_framelevel_set.append(F1_framelevel)
    iou_framelevel_set.append(iou_framelevel)

    fpr_videolevel_set.append(fpr_videolevel)
    tpr_videolevel_set.append(tpr_videolevel)
    precision_videolevel_set.append(precision_videolevel)
    recall_videolevel_set.append(recall_videolevel)
    F1_videolevel_set.append(F1_videolevel)
    iou_videolevel_set.append(iou_videolevel)
