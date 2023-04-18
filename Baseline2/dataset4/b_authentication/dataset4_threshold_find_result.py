import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import math
import os
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

   
    boundline = max(F1) * 0.95
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
    best_range=[low_thresholds[range_index[0]], low_thresholds[range_index[1]]]
    print("Processing",img_name)
    print("The best th", best_range)
    best_threshold_index = F1.index(max(F1))
    best_threshold = low_thresholds[best_threshold_index]
    precision_best=round(precision[best_threshold_index],4)
    recall_best=round(recall[best_threshold_index],4)
    F1_best=round(F1[best_threshold_index],4)
    iou_best=round(iou[best_threshold_index],4)
  #  print("best_threshold=", best_threshold)
  #  print("Precision={:.2%}".format(precision[best_threshold_index]))
  #  print("Recall={:.2%}".format(recall[best_threshold_index]))
  #  print("F1={:.2%}".format(F1[best_threshold_index]))
  #  print("IoU={:.2%}".format(iou[best_threshold_index]))

    pl.title("benchmarks")
    pl.figure()
    pl.plot(low_thresholds, precision, label="precision")
    pl.plot(low_thresholds, recall, label="recall")
    pl.plot(low_thresholds, F1, label="F1")
    pl.plot(low_thresholds, iou, label="iou")
    pl.vlines(low_thresholds[range_index[0]], 0, 1, color='black', linestyles='dashed')
    pl.vlines(low_thresholds[range_index[1]], 0, 1, color='black', linestyles='dashed')
    plt.legend(loc='upper center',bbox_to_anchor=(0.55,0.95),fontsize=15)
    plt.ylabel('Metrics', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    pl.savefig('../result/benchmarks_'+img_name+'.pdf')
    return precision_best,recall_best,F1_best,iou_best,best_range

advanced_framelevel_path=[]
advanced_videolevel_path=[]
advanced_framelevel_reverse_path=[]
advanced_videolevel_reverse_path=[]
data_path="../data/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'benchrmarks_' in file and 'ahash' not in file :
        
            if 'framelevel' in file:
                advanced_framelevel_path.append(video_path)
            elif 'videolevel' in file:
                advanced_videolevel_path.append(video_path)

print(advanced_framelevel_path)
print(advanced_videolevel_path)
print(advanced_framelevel_reverse_path)
print(advanced_videolevel_reverse_path)
ahash_framelevel = np.load('../data/benchrmarks_framelevel_data_ahash.npy', allow_pickle=True)
ahash_videolevel = np.load('../data/benchrmarks_videolevel_data_ahash.npy', allow_pickle=True)

advanced_framelevel=[]
advanced_videolevel=[]
advanced_framelevel_reverse=[]
advanced_videolevel_reverse=[]
for path in advanced_framelevel_path:
    advanced_framelevel.append(np.load(path, allow_pickle=True))
for path in advanced_videolevel_path:
    advanced_videolevel.append(np.load(path, allow_pickle=True))


sample_nums = range(11)
low_thresholds = np.arange(0, 16, 0.05)


tps_ahash_videolevel, fps_ahash_videolevel, fns_ahash_videolevel,[forged_video_num,real_video_num]=ahash_videolevel
tps_ahash_framelevel,fps_ahash_framelevel,fns_ahash_framelevel,[forged_frame_num,real_frame_num]=ahash_framelevel
precision_framelevel, recall_framelevel, F1_framelevel, iou_framelevel, range_framelevel = benchmarks_process(
    tps_ahash_framelevel,fps_ahash_framelevel,fns_ahash_framelevel, low_thresholds, 'fl_ahash')
precision_videolevel, recall_videolevel, F1_videolevel, iou_videolevel, range_videolevel = benchmarks_process(
    tps_ahash_videolevel,fps_ahash_videolevel,fns_ahash_videolevel, low_thresholds, 'vl_ahash')
range_framelevel_set=[]
range_videolevel_set=[]
range_framelevel_reverse_set=[]
range_videolevel_reverse_set=[]

for i in range(len(advanced_framelevel)):

    tps_framelevel,fps_framelevel,fns_framelevel,[forged_frame_num,real_frame_num]=advanced_framelevel[i]
    tps_videolevel,fps_videolevel,fns_videolevel,[forged_video_num,real_video_num]=advanced_videolevel[i]


    precision_framelevel, recall_framelevel, F1_framelevel, iou_framelevel, range_framelevel = benchmarks_process(
        tps_framelevel, fps_framelevel, fns_framelevel, low_thresholds, 'fl_sample_advanced_' + str(i) + '_deletion')
    range_framelevel_set.append(range_framelevel)
    precision_videolevel, recall_videolevel, F1_videolevel, iou_videolevel, range_videolevel = benchmarks_process(
        tps_videolevel, fps_videolevel, fns_videolevel, low_thresholds, 'vl_sample_advanced_' + str(i) + '_deletion')
    range_videolevel_set.append(range_videolevel)



