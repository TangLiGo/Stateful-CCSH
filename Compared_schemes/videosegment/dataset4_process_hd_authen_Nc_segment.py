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
segment_size=10
def VideoSegments(inputs,per):
	if np.shape(inputs)[0] % per != 0:
		lengh = np.shape(inputs)[0] / per
		for x in range(int(np.shape(inputs)[0]),int(lengh + 1)*per):
			inputs = np.append(inputs,inputs[np.shape(inputs)[0]-1])
	inputs = inputs.reshape((-1,per))
	tmpmean = inputs[0].mean()
	mean = []
	for tmp in inputs:
		mean.append(tmp.mean())

	return mean




def self_filter(inputs,high_threshold,low_threshold):
    #median_value=np.median(inputs)
    for index,temp in enumerate(inputs):

        if temp<low_threshold:
            inputs[index]=0
    return inputs

def forged_frame_statistic(forged_video_segment_info,segment_num,inputs):
    forged_frame_detected=[]
  #  print(forged_video_segment_info)
 #   print(len(inputs),inputs)
  #  print()
    true_positives=0
    false_positives=0
    false_negatives=0
    for index in range(len(inputs)):
        if inputs[index]>0:
            if index >= forged_video_segment_info[0] and index < forged_video_segment_info[1]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if index >= forged_video_segment_info[0] and index < forged_video_segment_info[1]:
                false_negatives += 1
            
    #print("true_positives,false_positives,false_negatives",true_positives,false_positives,false_negatives,segment_num)
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
def data_processing(nums_forgery,nums_compression,hash_len,high_threshold,hash_name,video_index):
    video_len = [840, 933, 858, 908, 359, 479, 736, 698, 666, 477,627,491,701,394,679,448]
    forged_info = [[500, 662], [551, 932], [495, 858], [473, 903], [317, 359], [1, 155], [511, 710], [596, 685], [383, 654],
                   [294, 373],[410,627],[328,491],[29,239],[250,348],[531,659],[1,149]]  # 954
 #   print(nums_forgery)
    forged_video_num = 0
    compressed_video_num = 0
    forged_frame_num = 0
    forged_video_segment_num=0
    window_size = 10
    forged_video_frame_num = 0  # The number of forged videos' frames
    video_segment_num = 0  # The number of original videos' frames
    for i in range(len(nums_forgery)):  # 第几组
        for j in range(video_index,video_index+1):  # 第几个
            forged_video_segment_num+=int(forged_info[j][1]/segment_size) - int(forged_info[j][0]/segment_size) + 1
            video_segment_num += int((forged_info[j][1] - forged_info[j][0] + 1)/segment_size)
    
    low_thresholds = np.arange(0, 7, 0.05)

    tps_segment = []  # The matrix to store the num of correctly detected forged videos       true positives & judged positives
    fps_segment = []  # The matrix to store the num of falsely judged forged videos          true negatives & judged positives
    fns_segment = []  # The matrix to store the num of falsely judged tampered videos        true positives & judged negatives
    tps_segment_probability = []
    fps_segment_probability = []

    for low_threshold in low_thresholds:
        tp_segment = 0
        fp_segment = 0
        fn_segment = 0

        for i in range(len(nums_forgery)):
            for j in range(video_index,video_index+1):
             #   if j == 3 or j == 14 or j == 15:
               #     continue
                num = np.array(nums_forgery[i][j])
                video_segments = VideoSegments(np.array(num).copy(), segment_size)
             #   print(num)

                result = self_filter(video_segments, high_threshold, low_threshold)
              #  print(video_segments)
          #      print(result)
                tp, fp, fn = forged_frame_statistic([int(forged_info[j][0]/segment_size),int(forged_info[j][1]/segment_size)], int(video_len[j]/segment_size),  result)
                tp_segment += tp
                fp_segment += fp
                fn_segment += fn
                
        
        tps_segment.append(tp_segment)
        fns_segment.append(fn_segment)
        fps_segment.append(fp_segment)
        tps_segment_probability.append(tp_segment / forged_video_segment_num)
        fps_segment_probability.append(fp_segment / (video_segment_num - forged_video_segment_num))

    print("frame level")
    img_path="result/segment_"+hash_name+".pdf"
    precision_segment,recall_segment,F1_segment,iou_segment=benchmarks_process(tps_segment,fps_segment,fns_segment,low_thresholds,img_path)

    return [fps_segment_probability, tps_segment_probability,precision_segment,recall_segment,F1_segment,iou_segment]



def getVideoData(video_index):
    fpr_segment = []
    tpr_segment = []
    precision_segment = []
    recall_segment = []
    F1_segment = []
    iou_segment = []

    fpr_videolevel = []
    tpr_videolevel = []
    precision_videolevel = []
    recall_videolevel = []
    F1_videolevel = []
    iou_videolevel = []

    for i in range(len(hds_ahash_samples)):

        print("processing ----------------------",i,len(hds_ahash_samples))
        segment_data = data_processing(hds_ahash_samples[i], hds_ahash_samples_compression[i], 16,
                                                           16,
                                                           'sample_names[i]',video_index)
        fpr_segment_temp, tpr_segment_temp, precision_segment_temp, recall_segment_temp, F1_segment_temp, iou_segment_temp = segment_data


        fpr_segment.append(fpr_segment_temp)
        tpr_segment.append(tpr_segment_temp)
        precision_segment.append(precision_segment_temp)
        recall_segment.append(recall_segment_temp)
        F1_segment.append(F1_segment_temp)
        iou_segment.append(iou_segment_temp)

    
    framedata=[fpr_segment,tpr_segment,precision_segment,recall_segment,F1_segment,iou_segment]
    np.save("data/Nc_video_segment_m1_"+str(video_index)+".npy",framedata)
    plt.figure()
    plt.plot(F1_segment)
    plt.savefig("figs/Nc_video_segment_m1_"+str(video_index)+".png")
    return F1_videolevel,F1_segment

def getHashData():
    ahash_samples_path = []
    ahash_samples_insertion_path = []
    ahash_samples_compression_path = []
    ahash_samples_insertion_compression_path = []
    data_path = "data/"
    for file in os.listdir(data_path):
        video_path = os.path.join(data_path, file)
        video_path = video_path.replace('\\', '/')
        if 'segment_m2' in file and 'hds_ahash_sample_advanced' in file:

            if "compression" in file:
                ahash_samples_compression_path.append(video_path)
            else:
                ahash_samples_path.append(video_path)

    hds_ahash_samples = []
    hds_ahash_samples_compression = []

    sample_names = []
    sample_names_insertion = []
    for path in ahash_samples_path:
        hds_ahash_samples.append(np.load(path, allow_pickle=True))
        (subpath, temp_file_name) = os.path.split(path)
        (output_file_name, extension) = os.path.splitext(temp_file_name)
        sample_names.append(output_file_name)
    for path in ahash_samples_compression_path:
        hds_ahash_samples_compression.append(np.load(path, allow_pickle=True))

    return hds_ahash_samples,hds_ahash_samples_compression
def readVideoData(video_index):

    framedata=np.load("data/Nc_video_segment_m1_" + str(video_index) + ".npy",allow_pickle=True )

    fpr_segment,tpr_segment,precision_segment,recall_segment,F1_segment,iou_segment=framedata

    N = [0,1,2,3,4,5]
    pl.figure()
    pl.bar(N, F1_segment, color='steelblue', width=2)
    pl.plot(N, F1_segment, color='darkorange')
    # plt.legend(fontsize=18)

    x_major_locator = MultipleLocator(10)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.ylim(0.5, 1)
    plt.xlabel('$N_c$', fontsize=18)
    plt.ylabel('F1', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("figs/Nc_video_m1_"+str(video_index)+".png")
    print(F1_segment)

    return F1_segment
hds_ahash_samples,hds_ahash_samples_compression=getHashData()

for i in range(16):
    print("Video aaaaaaaaaaaaaaa",i)
    d=getVideoData(i)
