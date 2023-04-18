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
data_path="../data/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'roc_' in file and 'ahash' not in file :
            if 'framelevel' in file:
                framelevel_path.append(video_path)
            elif 'videolevel' in file:
                videolevel_path.append(video_path)

print(videolevel_path)
ahash_framelevel=np.load("../data/roc_framelevel_data_ahash.npy", allow_pickle=True)
ahash_videolevel=np.load("../data/roc_videolevel_data_ahash.npy", allow_pickle=True)
fpr_framelevel_ahash, tpr_framelevel_ahash, precision_framelevel_ahash, recall_framelevel_ahash, F1_framelevel_ahash, iou_framelevel_ahash = \
ahash_framelevel
fpr_videolevel_ahash, tpr_videolevel_ahash, precision_videolevel_ahash, recall_videolevel_ahash, F1_videolevel_ahash, iou_videolevel_ahash = \
ahash_videolevel


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



for i in range(len(sample_nums)):

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



font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
pl.figure()
pl.plot(sample_nums,F1_framelevel,color='steelblue',marker='+')
#plt.legend(fontsize=12)

x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.7,0.85)
ax.set_xlabel('$N_r$',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/baseline1_d1_fl_marker.pdf")
benchmarks_videolevel=[]



pl.figure()
pl.plot(sample_nums,F1_videolevel,color='steelblue',marker='+')
#plt.legend(fontsize=12)

x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.85,1)
plt.xlabel('$N_r$',fontsize=12)
plt.ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/baseline1_d1_vl_marker.pdf")


pl.figure()
pl.bar(sample_nums,F1_framelevel,color='steelblue', width=2)
pl.plot(sample_nums,F1_framelevel,color='darkorange')
#plt.legend(fontsize=12)

x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.ylim(0.6,0.85)
ax.set_xlabel('$N_r$',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/baseline1_d1_fl_bar.pdf")




pl.figure()
pl.bar(sample_nums,F1_videolevel,color='steelblue', width=2)
pl.plot(sample_nums,F1_videolevel,color='darkorange')
#plt.legend(fontsize=12)

#x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
#ax=plt.gca()
#ax为两条坐标轴的实例
#ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.ylim(0.88,0.99)
plt.xlabel('$N_r$',fontsize=12)
plt.ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/baseline1_d1_vl_bar.pdf")


pl.figure()
pl.plot(sample_nums,F1_framelevel,color='darkorange')
#plt.legend(fontsize=12)

x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.7,0.85)
ax.set_xlabel('$N_r$',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/baseline1_d1_fl_ahash.pdf")




pl.figure()
pl.plot(sample_nums,[F1_videolevel_ahash for i in sample_nums],label='ahash',color='black')
pl.plot(sample_nums,F1_videolevel,label='baseline 1',color='steelblue')
#plt.legend(fontsize=12)

x_major_locator=MultipleLocator(10)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
#plt.ylim(0.85,1)
plt.xlabel('$N_r$',fontsize=12)
plt.ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/baseline1_d1_vl_ahash.pdf")