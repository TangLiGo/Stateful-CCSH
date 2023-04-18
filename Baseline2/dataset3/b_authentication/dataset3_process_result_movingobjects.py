import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from matplotlib.pyplot import MultipleLocator
advanced_framelevel_path=[]
advanced_videolevel_path=[]
advanced_framelevel_reverse_path=[]
advanced_videolevel_reverse_path=[]
data_path="dataset3_baseline2_moving/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'roc_' in file and 'ahash' not in file:
            if 'framelevel' in file:
                advanced_framelevel_path.append(video_path)
            elif 'videolevel' in file:
                advanced_videolevel_path.append(video_path)

print(advanced_videolevel_path)

ahash_framelevel = np.load('dataset3_baseline2_moving/roc_framelevel_data_ahash.npy', allow_pickle=True)
ahash_videolevel = np.load('dataset3_baseline2_moving/roc_videolevel_data_ahash.npy', allow_pickle=True)

advanced_framelevel=[]
advanced_videolevel=[]
advanced_framelevel_reverse=[]
advanced_videolevel_reverse=[]
for path in advanced_framelevel_path:
    advanced_framelevel.append(np.load(path, allow_pickle=True))
for path in advanced_videolevel_path:
    advanced_videolevel.append(np.load(path, allow_pickle=True))
for path in advanced_framelevel_reverse_path:
    advanced_framelevel_reverse.append(np.load(path, allow_pickle=True))
for path in advanced_videolevel_reverse_path:
    advanced_videolevel_reverse.append(np.load(path, allow_pickle=True))

sample_nums = range(6)



fpr_ahash_framelevel,tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel=ahash_framelevel
fpr_ahash_videolevel,tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel=ahash_videolevel

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



for i in range(len(advanced_framelevel)):

    fpr_framelevel_temp, tpr_framelevel_temp, precision_framelevel_temp, recall_framelevel_temp, F1_framelevel_temp, iou_framelevel_temp=advanced_framelevel[i]
    fpr_videolevel_temp, tpr_videolevel_temp, precision_videolevel_temp, recall_videolevel_temp, F1_videolevel_temp, iou_videolevel_temp=advanced_videolevel[i]

    fpr_framelevel.append(fpr_framelevel_temp)
    tpr_framelevel.append(tpr_framelevel_temp)
    fpr_videolevel.append(fpr_videolevel_temp)
    tpr_videolevel.append(tpr_videolevel_temp)

    precision_framelevel.append(precision_framelevel_temp)
    recall_framelevel.append(recall_framelevel_temp)
    F1_framelevel.append(F1_framelevel_temp)
    iou_framelevel.append(iou_framelevel_temp)

    precision_videolevel.append(precision_videolevel_temp)
    recall_videolevel.append(recall_videolevel_temp)
    F1_videolevel.append(F1_videolevel_temp)
    iou_videolevel.append(iou_videolevel_temp)

    pl.figure()
    pl.plot(fpr_ahash_framelevel,tpr_ahash_framelevel,label='ahash')
    pl.plot(fpr_framelevel[0], tpr_framelevel[0], label='baseline')
    pl.plot(fpr_framelevel_temp, tpr_framelevel_temp, label='CCSHG')
  #  pl.plot(fpr_framelevel_reverse_temp, tpr_framelevel_reverse_temp, label='CCSHG-insertion')
    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("dataset3_baseline2_moving_result/d1_roc_fl_"+str(i)+".pdf")

    pl.figure()
    pl.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash')
    pl.plot(fpr_videolevel[0],  tpr_videolevel[0], label='baseline')
    pl.plot(fpr_videolevel_temp, tpr_videolevel_temp, label='CCSHG')
  #  pl.plot(fpr_videolevel_reverse_temp, tpr_videolevel_reverse_temp, label='CCSHG-insertion')
    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("dataset3_baseline2_moving_result/d1_roc_vl_"+str(i)+".pdf")

pl.figure()
pl.plot(sample_nums[1:6],F1_framelevel[1:6], label='CCSH')

#pl.hlines((precision_ahash_framelevel+recall_ahash_framelevel+F1_ahash_framelevel+iou_ahash_framelevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_framelevel[0],1,5,color='red',linestyles='dashed',label="baseline")
#pl.vlines(5,0.6,0.82,color='black',linestyles='dashed')
plt.legend()
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数

plt.ylim(0.71,0.9)
#plt.ylim(0,1)
#plt.yscale('log')
plt.xlabel('cN')
plt.ylabel('F1')
pl.savefig("dataset3_baseline2_moving_result/d3_fl_single1.pdf")

pl.figure()
pl.plot(sample_nums[1:6],F1_videolevel[1:6], label='CCSH ')

#pl.hlines((precision_ahash_videolevel+recall_ahash_videolevel+F1_ahash_videolevel+iou_ahash_videolevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_videolevel[0],1,5,color='red',linestyles='dashed',label="baseline")

plt.legend()
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.ylim((0.85, 1))
#plt.ylim(0,1)
#plt.yscale('log')
plt.xlabel('cN')
plt.ylabel('F1')
pl.savefig("dataset3_baseline2_moving_result/d3_vl_single1.pdf")

pl.figure()
pl.plot(sample_nums[:6],F1_framelevel[:6], label='CCSH ')

#pl.hlines((precision_ahash_framelevel+recall_ahash_framelevel+F1_ahash_framelevel+iou_ahash_framelevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_framelevel[0],0,5,color='red',linestyles='dashed',label="baseline")
#pl.vlines(5,0.6,0.82,color='black',linestyles='dashed')
plt.legend()
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数

plt.ylim(0.71,0.9)
#plt.ylim(0,1)
#plt.yscale('log')
plt.xlabel('cN')
plt.ylabel('F1')
pl.savefig("dataset3_baseline2_moving_result/d3_fl_single2.pdf")

pl.figure()
pl.plot(sample_nums[:6],F1_videolevel[:6], label='CCSH')

#pl.hlines((precision_ahash_videolevel+recall_ahash_videolevel+F1_ahash_videolevel+iou_ahash_videolevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_videolevel[0],0,5,color='red',linestyles='dashed',label="baseline")

plt.legend()
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.ylim((0.9, 1))
#plt.ylim(0,1)
#plt.yscale('log')
plt.xlabel('cN')
plt.ylabel('F1')
pl.savefig("dataset3_baseline2_moving_result/d3_vl_single2.pdf")