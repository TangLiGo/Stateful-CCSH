import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import math
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

    benchmark_sums = [i + j + k + p for i, j, k, p in zip(precision, recall, F1, iou)]
    best_threshold_index = benchmark_sums.index(max(benchmark_sums))
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

    # The set to store the in formation of tampered frames' location
    forged_info1 = [[2, 31], [2, 31], [2, 31], [2, 31], [1, 30], [1, 30], [1, 20], [1, 30], [1, 30], [10, 30], [1, 30],
                    [1, 14], [1, 30], [1, 30], [1, 14], [1, 30], [1, 30], [1, 14], [1, 30], [1, 30], [1, 30], [1, 30]]
    forged_info2 = [[1, 25], [1, 30], [1, 30], [136, 165], [1, 25], [1, 25], [1, 25], [1, 25], [168, 192], [1, 25],
                    [1, 25], [1, 25], [1, 50], [1, 30], [1, 30], [1, 30], [1, 25], [1, 25], [1, 25], [1, 25], [1, 25]]
    forged_info3 = [[1, 25], [1, 25], [1, 19], [1, 25], [3, 27], [1, 25], [1, 25], [55, 79], [1, 25], [1, 25], [1, 21],
                    [1, 21], [1, 25], [1, 19], [1, 25]]
    forged_info4 = [[3, 27], [3, 27], [1, 25], [1, 25], [51, 75], [100, 124], [1, 50], [209, 258], [3, 27], [1, 25],
                    [3, 27], [3, 27], [3, 27], [3, 27], [3, 27]]
    forged_info5 = [[1, 18], [1, 18], [1, 16], [1, 18], [1, 16], [1, 20], [6, 25], [1, 20], [1, 20], [1, 20], [1, 10],
                    [1, 20], [2, 26], [29, 48], [1, 20], [1, 10], [1, 16], [6, 21], [6, 21], [1, 20], [1, 20], [1, 16],
                    [1, 16]]
    forged_info6 = [[2, 56], [1, 55], [10, 40], [10, 64], [50, 82], [50, 99], [50, 86], [50, 87], [50, 87], [1, 25],
                    [1, 25], [1, 39], [1, 40], [46, 85], [1, 51], [51, 100], [1, 50], [1, 25], [1, 25], [1, 50],
                    [1, 42], [27, 76], [1, 50], [30, 79], [1, 50], [1, 40], [1, 55], [1, 22], [1, 25], [1, 25], [1, 55],
                    [1, 50]]
    forged_info = [forged_info1, forged_info2, forged_info3, forged_info4, forged_info5, forged_info6]

    # Compute the number of videos' frames all for benchmarks
    video_len = [172, 334, 98, 259, 554, 104]
    forged_video_n = [22, 21, 15, 15, 23, 32]  # The number of videos for forgery detection
    frame_lens_forgery = []
    for i in range(6):
        frame_lens_forgery.append([video_len[i] for k in range(forged_video_n[i])])
    frame_num = 0  # The number of forged videos' frames

    forged_frame_num = 0

    for i in range(len(nums_forgery)):#2-compressed forgery or uncompressed forgery
        for j in range(len(nums_forgery[i])):#video group
            for k in range(len(nums_forgery[i][j])):#video
                frame_num +=len(nums_forgery[i][j][k])
                #print(i,j,k,len(nums_forgery[i][j][k]))

    # Compute the number of forged frames in all videos
    for i in range(len(forged_info)):
        for j in range(len(forged_info[i])):
            forged_frame_num += forged_info[i][j][1] - forged_info[i][j][0] + 1
    forged_frame_num=2*forged_frame_num*int(len(nums_forgery[0])/6)


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


    video_for_forgery = 0
    video_for_compression = 0

    for i in range(len(nums_forgery)):
        for j in range(len(nums_forgery[i])):
            for k in range(len(nums_forgery[i][j])):
                video_for_forgery += 1

    for i in range(len(nums_compression)):
        video_for_compression += 1

    for low_threshold in low_thresholds:
        tp_framelevel = 0
        fp_framelevel = 0
        fn_framelevel = 0
        tp_videolevel = 0
        fp_videolevel = 0
        fn_videolevel = 0
        for i in range(len(nums_forgery)):
            for j in range(len(nums_forgery[i])):
                for k in range(len(nums_forgery[i][j])):

                    num = np.array(nums_forgery[i][j][k])

                    filter_result = SlidingAverage(np.array(num).copy(), window_size)

                    result = self_filter(filter_result, high_threshold, low_threshold)

                    tp, fp, fn = forged_frame_statistic(forged_info[j%6][k], frame_lens_forgery[j%6][k],
                                                                        window_size, result)
                    tp_framelevel += tp
                    fp_framelevel += fp
                    fn_framelevel += fn
                    if tp + fp > 0:
                        tp_videolevel += 1
        for i in range(len(nums_compression)):
            num = np.array(nums_compression[i])
            filter_result = SlidingAverage(np.array(num).copy(), window_size)
            result = self_filter(filter_result, high_threshold, low_threshold)
            if false_negatives_statistic(window_size, result) > 0:
                fp_videolevel += 1

        tps_framelevel.append(tp_framelevel)
        fns_framelevel.append(fn_framelevel)
        fps_framelevel.append(fp_framelevel)
        tps_framelevel_probability.append(tp_framelevel / forged_frame_num)
        fps_framelevel_probability.append(fp_framelevel / (frame_num - forged_frame_num))

        tps_videolevel.append(tp_videolevel)
        fns_videolevel.append(video_for_forgery - tp_videolevel)
        fps_videolevel.append(fp_videolevel)
        tps_videolevel_probability.append(tp_videolevel / video_for_forgery)
        fps_videolevel_probability.append(fp_videolevel / video_for_compression)




    print("frame level")
    img_path="../result/framelevel_"+hash_name+".pdf"
    precision_framelevel,recall_framelevel,F1_framelevel,iou_framelevel=benchmarks_process(tps_framelevel,fps_framelevel,fns_framelevel,low_thresholds,img_path)
    print("video level")
    img_path = "../result/videolevel_" + hash_name + ".pdf"
    precision_videolevel,recall_videolevel,F1_videolevel,iou_videolevel=benchmarks_process(tps_videolevel, fps_videolevel, fns_videolevel, low_thresholds, img_path)

    return [fps_framelevel_probability, tps_framelevel_probability,precision_framelevel,recall_framelevel,F1_framelevel,iou_framelevel],[fps_videolevel_probability,tps_videolevel_probability,precision_videolevel,recall_videolevel,F1_videolevel,iou_videolevel]

ahash_samples_path=[]
ahash_samples_deletion_path=[]
ahash_samples_compression_path=[]
ahash_samples_deletion_compression_path=[]
data_path="../data_deletion/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'hds_ahash_sample_advanced' in file :

            if "compression" in file:
                ahash_samples_deletion_compression_path.append(video_path)
            else:
                ahash_samples_deletion_path.append(video_path)

hds_ahash = np.load('../data/hds_ahash.npy', allow_pickle=True)
hds_ahash_compression = np.load('../data/hds_ahash_compression.npy', allow_pickle=True)

hds_ahash_samples=[]
hds_ahash_samples_compression=[]
hds_ahash_samples_deletion=[]
hds_ahash_samples_deletion_compression=[]
sample_names=[]
sample_names_deletion=[]

for path in ahash_samples_deletion_path:
    hds_ahash_samples_deletion.append(np.load(path, allow_pickle=True))
    (subpath, temp_file_name) = os.path.split(path)
    (output_file_name,extension)=os.path.splitext(temp_file_name)
    sample_names_deletion.append(output_file_name)
for path in ahash_samples_deletion_compression_path:
    hds_ahash_samples_deletion_compression.append(np.load(path, allow_pickle=True))
#hds_ahash_sample = np.load('../../data_deletion/dataset2/hds_ahash_sample.npy', allow_pickle=True)
#hds_ahash_sample_compression = np.load('../data_deletion/hds_ahash_sample_compression.npy', allow_pickle=True)

print(ahash_samples_path)
print(ahash_samples_compression_path)
print(ahash_samples_deletion_path)
print(ahash_samples_deletion_compression_path)
sample_nums = range(6)


ahash_framelevel,ahash_videolevel=data_processing(hds_ahash ,hds_ahash_compression,16, 16,'ahash')
fpr_ahash_framelevel,tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel=ahash_framelevel
fpr_ahash_videolevel,tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel=ahash_videolevel


np.save('../data_deletion/roc_videolevel_data_ahash.npy', np.array([fpr_ahash_videolevel, tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel],dtype=object))
np.save('../data_deletion/roc_framelevel_data_ahash.npy', np.array([fpr_ahash_framelevel, tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel],dtype=object))


fpr_framelevel = []
tpr_framelevel = []
precision_framelevel=[]
recall_framelevel=[]
F1_framelevel=[]
iou_framelevel=[]
fpr_framelevel_deletion = []
tpr_framelevel_deletion = []
precision_framelevel_deletion=[]
recall_framelevel_deletion=[]
F1_framelevel_deletion=[]
iou_framelevel_deletion=[]

fpr_videolevel = []
tpr_videolevel = []
precision_videolevel=[]
recall_videolevel=[]
F1_videolevel=[]
iou_videolevel=[]
fpr_videolevel_deletion = []
tpr_videolevel_deletion = []
precision_videolevel_deletion=[]
recall_videolevel_deletion=[]
F1_videolevel_deletion=[]
iou_videolevel_deletion=[]
for i in range(len(hds_ahash_samples_deletion)):


    print("processing",sample_names_deletion[i])
    framelevel_data_deletion,videolevel_data_deletion= data_processing(hds_ahash_samples_deletion[i],hds_ahash_samples_deletion_compression[i], 16, 16,
                                                                                         sample_names_deletion[i])
    fpr_framelevel_deletion_temp, tpr_framelevel_deletion_temp, precision_framelevel_deletion_temp, recall_framelevel_deletion_temp, F1_framelevel_deletion_temp, iou_framelevel_deletion_temp =framelevel_data_deletion
    fpr_videolevel_deletion_temp, tpr_videolevel_deletion_temp, precision_videolevel_deletion_temp, recall_videolevel_deletion_temp, F1_videolevel_deletion_temp, iou_videolevel_deletion_temp = videolevel_data_deletion
    if i>9:
        np.save('../data_deletion/framelevel_' + str(i) + '_deletion.npy',
                np.array([fpr_framelevel_deletion_temp, tpr_framelevel_deletion_temp, precision_framelevel_deletion_temp,
                          recall_framelevel_deletion_temp, F1_framelevel_deletion_temp, iou_framelevel_deletion_temp],
                         dtype=object))
        np.save('../data_deletion/videolevel_' + str(i) + '_deletion.npy',
                np.array([fpr_videolevel_deletion_temp, tpr_videolevel_deletion_temp, precision_videolevel_deletion_temp,
                          recall_videolevel_deletion_temp, F1_videolevel_deletion_temp, iou_videolevel_deletion_temp],
                         dtype=object))
    else:
        np.save('../data_deletion/framelevel_0' + str(i) + '_deletion.npy',
                np.array([fpr_framelevel_deletion_temp, tpr_framelevel_deletion_temp, precision_framelevel_deletion_temp,
                          recall_framelevel_deletion_temp, F1_framelevel_deletion_temp, iou_framelevel_deletion_temp],
                         dtype=object))
        np.save('../data_deletion/videolevel_0' + str(i) + '_deletion.npy',
                np.array([fpr_videolevel_deletion_temp, tpr_videolevel_deletion_temp, precision_videolevel_deletion_temp,
                          recall_videolevel_deletion_temp, F1_videolevel_deletion_temp, iou_videolevel_deletion_temp],
                         dtype=object))


    fpr_framelevel_deletion.append(fpr_framelevel_deletion_temp)
    tpr_framelevel_deletion.append(tpr_framelevel_deletion_temp)
    precision_framelevel_deletion.append(precision_framelevel_deletion_temp)
    recall_framelevel_deletion.append(recall_framelevel_deletion_temp)
    F1_framelevel_deletion.append(F1_framelevel_deletion_temp)
    iou_framelevel_deletion.append(iou_framelevel_deletion_temp)


    fpr_videolevel_deletion.append(fpr_videolevel_deletion_temp)
    tpr_videolevel_deletion.append(tpr_videolevel_deletion_temp)
    precision_videolevel_deletion.append(precision_videolevel_deletion_temp)
    recall_videolevel_deletion.append(recall_videolevel_deletion_temp)
    F1_videolevel_deletion.append(F1_videolevel_deletion_temp)
    iou_videolevel_deletion.append(iou_videolevel_deletion_temp)
