import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from matplotlib.pyplot import MultipleLocator
framelevel_path=[]
videolevel_path=[]
framelevel_reverse_path=[]
videolevel_reverse_path=[]
data_path="../data/dataset2/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'roc_' in file and 'ahash' not in file :
            if 'framelevel' in file:
                framelevel_path.append(video_path)
            elif 'videolevel' in file:
                videolevel_path.append(video_path)

print(videolevel_path)


framelevel=[]
videolevel=[]

for path in framelevel_path:
    framelevel.append(np.load(path, allow_pickle=True))
for path in videolevel_path:
    videolevel.append(np.load(path, allow_pickle=True))

sample_nums = range(5,105,5)

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



for i in range(len(framelevel)):

    fpr_framelevel_temp, tpr_framelevel_temp, precision_framelevel_temp, recall_framelevel_temp, F1_framelevel_temp, iou_framelevel_temp=framelevel[i]
    fpr_videolevel_temp, tpr_videolevel_temp, precision_videolevel_temp, recall_videolevel_temp, F1_videolevel_temp, iou_videolevel_temp=videolevel[i]

    precision_framelevel.append(precision_framelevel_temp)
    recall_framelevel.append(recall_framelevel_temp)
    F1_framelevel.append(F1_framelevel_temp)
    iou_framelevel.append(iou_framelevel_temp)

    precision_videolevel.append(precision_videolevel_temp)
    recall_videolevel.append(recall_videolevel_temp)
    F1_videolevel.append(F1_videolevel_temp)
    iou_videolevel.append(iou_videolevel_temp)




pl.figure()
pl.plot(sample_nums,F1_framelevel)
plt.legend()
plt.legend()
x_major_locator=MultipleLocator(5)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.7,0.85)
plt.xlabel('rN')
plt.ylabel('F1')
pl.savefig("../data/dataset2_result/d2_fl_f1_2.pdf")
benchmarks_videolevel=[]



pl.figure()
pl.plot(sample_nums,F1_videolevel)
plt.legend()
plt.legend()
x_major_locator=MultipleLocator(5)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.85,1)
plt.xlabel('rN')
plt.ylabel('F1')
pl.savefig("../data/dataset2_result/d2_vl_f1_2.pdf")


pl.figure()
pl.plot(sample_nums,F1_framelevel,label='Precision')
pl.plot(sample_nums,iou_framelevel,label='IoU')
plt.legend()
plt.legend()
x_major_locator=MultipleLocator(5)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.7,0.85)
plt.xlabel('rN')
plt.ylabel('F1')
pl.savefig("../data/dataset2_result/baseline1_d2_fl2.pdf")




pl.figure()
pl.plot(sample_nums,F1_videolevel,label='F1')
pl.plot(sample_nums,iou_videolevel,label='IoU')
plt.legend()
plt.legend()
x_major_locator=MultipleLocator(5)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.85,1)
plt.xlabel('rN')
plt.ylabel('F1')
pl.savefig("../data/dataset2_result/baseline1_d2_vl2.pdf")