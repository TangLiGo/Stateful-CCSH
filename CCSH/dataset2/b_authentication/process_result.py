import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from matplotlib.pyplot import MultipleLocator

def drawBenchmarks(len_quota,benchmark_framelevel,benchmark_videolevel,ben_name):
    x_data=range(1, len_quota)

    pl.figure()
    pl.plot(x_data, benchmark_framelevel[1:len_quota], label='CCSHG')
    # pl.plot(sample_nums[1:],F1_framelevel_insertion[1:], label='CCSHG - Object Insertion')
    # pl.hlines((precision_ahash_framelevel+recall_ahash_framelevel+F1_ahash_framelevel+iou_ahash_framelevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
    pl.hlines(benchmark_framelevel[0], 1, len_quota, color='red',
              linestyles='dashed', label="baseline")
    # pl.vlines(5,0.6,0.82,color='black',linestyles='dashed')
    plt.legend()
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数

    #plt.ylim(0.6, 0.82)
    # plt.ylim(0,1)
    # plt.yscale('log')
    plt.xlabel('cN')
    plt.ylabel('F1')
    pl.savefig("../result/d2_fl_"+ben_name+"_double.pdf")


    pl.figure()
    pl.plot(x_data, benchmark_videolevel[1:len_quota], label='CCSHG')
    # pl.plot(sample_nums[1:],F1_videolevel_insertion[1:], label='CCSHG - Object Insertion')
    # pl.hlines((precision_ahash_videolevel+recall_ahash_videolevel+F1_ahash_videolevel+iou_ahash_videolevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
    pl.hlines(benchmark_videolevel[0] , 1, len_quota, color='red',
              linestyles='dashed', label="baseline")
   # pl.vlines(5, 0.85, 1, color='black', linestyles='dashed')
    plt.legend()
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.ylim((0.85, 1))
    # plt.ylim(0,1)
    # plt.yscale('log')
    plt.xlabel('cN')
    plt.ylabel('F1')
    pl.savefig("../result/d2_vl_"+ben_name+"_double.pdf")
advanced_framelevel_path=[]
advanced_videolevel_path=[]
advanced_framelevel_insertion_path=[]
advanced_videolevel_insertion_path=[]
data_path="../data/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'ahash' not in file :
        if 'framelevel' in file:
            advanced_framelevel_insertion_path.append(video_path)
        elif 'videolevel' in file:
            advanced_videolevel_insertion_path.append(video_path)


print(advanced_framelevel_insertion_path)

ahash_framelevel = np.load('../data/roc_framelevel_data_ahash.npy', allow_pickle=True)
ahash_videolevel = np.load('../data/roc_videolevel_data_ahash.npy', allow_pickle=True)


advanced_framelevel_insertion=[]
advanced_videolevel_insertion=[]

for path in advanced_framelevel_insertion_path:
    advanced_framelevel_insertion.append(np.load(path, allow_pickle=True))
for path in advanced_videolevel_insertion_path:
    advanced_videolevel_insertion.append(np.load(path, allow_pickle=True))
len_quota=6
sample_nums = range(len_quota)



fpr_ahash_framelevel,tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel=ahash_framelevel
fpr_ahash_videolevel,tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel=ahash_videolevel


fpr_framelevel_insertion = []
tpr_framelevel_insertion = []
precision_framelevel_insertion=[]
recall_framelevel_insertion=[]
F1_framelevel_insertion=[]
iou_framelevel_insertion=[]


fpr_videolevel_insertion = []
tpr_videolevel_insertion = []
precision_videolevel_insertion=[]
recall_videolevel_insertion=[]
F1_videolevel_insertion=[]
iou_videolevel_insertion=[]


for i in range(len_quota):

    fpr_framelevel_insertion_temp, tpr_framelevel_insertion_temp, precision_framelevel_insertion_temp, recall_framelevel_insertion_temp, F1_framelevel_insertion_temp, iou_framelevel_insertion_temp =advanced_framelevel_insertion[i]
    fpr_videolevel_insertion_temp, tpr_videolevel_insertion_temp, precision_videolevel_insertion_temp, recall_videolevel_insertion_temp, F1_videolevel_insertion_temp, iou_videolevel_insertion_temp =advanced_videolevel_insertion[i]
  
    fpr_framelevel_insertion.append(fpr_framelevel_insertion_temp)
    tpr_framelevel_insertion.append(tpr_framelevel_insertion_temp)
    fpr_videolevel_insertion.append(fpr_videolevel_insertion_temp)
    tpr_videolevel_insertion.append(tpr_videolevel_insertion_temp)

  

    precision_framelevel_insertion.append(precision_framelevel_insertion_temp)
    recall_framelevel_insertion.append(recall_framelevel_insertion_temp)
    F1_framelevel_insertion.append(F1_framelevel_insertion_temp)
    iou_framelevel_insertion.append(iou_framelevel_insertion_temp)

 
    precision_videolevel_insertion.append(precision_videolevel_insertion_temp)
    recall_videolevel_insertion.append(recall_videolevel_insertion_temp)
    F1_videolevel_insertion.append(F1_videolevel_insertion_temp)
    iou_videolevel_insertion.append(iou_videolevel_insertion_temp)
    pl.figure()
    pl.plot(fpr_ahash_framelevel,tpr_ahash_framelevel,label='ahash')
    pl.plot(fpr_framelevel_insertion[0],  tpr_framelevel_insertion[0], label='baseline')

    pl.plot(fpr_framelevel_insertion_temp, tpr_framelevel_insertion_temp, label='CCSHG-insertion')
    plt.legend()
    plt.ylim((0.5, 1))
    plt.xlim((0, 0.27))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("../result/d2_roc_fl_"+str(i)+".pdf")

    pl.figure()
    pl.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash')
    pl.plot( fpr_videolevel_insertion[0], tpr_videolevel_insertion[0], label='baseline')
 
    pl.plot(fpr_videolevel_insertion_temp, tpr_videolevel_insertion_temp, label='CCSHG-insertion')
    plt.legend()
    plt.ylim((0.8, 1))
    plt.xlim((0, 0.4))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("../result/d2_roc_vl_"+str(i)+".pdf")


pl.figure()

pl.plot(sample_nums[0:6],F1_framelevel_insertion[0:6], label='CCSH')
#pl.hlines((precision_ahash_framelevel+recall_ahash_framelevel+F1_ahash_framelevel+iou_ahash_framelevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_framelevel_insertion[0],0,5,color='red',linestyles='dashed',label="baseline")
#pl.vlines(5,0.6,0.82,color='black',linestyles='dashed')
plt.legend()
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数

plt.ylim(0.71,0.835)
#plt.ylim(0,1)
#plt.yscale('log')
plt.xlabel('cN')
plt.ylabel('F1')
pl.savefig("../result/d2_fl.pdf")



pl.figure()

pl.plot(sample_nums[:6],F1_videolevel_insertion[0:6], label='CCSH')
#pl.hlines((precision_ahash_videolevel+recall_ahash_videolevel+F1_ahash_videolevel+iou_ahash_videolevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_videolevel_insertion[0],0,5,color='red',linestyles='dashed',label="baseline")

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
pl.savefig("../result/d2_vl.pdf")

drawBenchmarks(len_quota,precision_framelevel_insertion,precision_videolevel_insertion,"precision")
drawBenchmarks(len_quota,recall_framelevel_insertion,recall_videolevel_insertion,"recall")
drawBenchmarks(len_quota,F1_framelevel_insertion,F1_videolevel_insertion,"F1")
drawBenchmarks(len_quota,iou_framelevel_insertion,iou_videolevel_insertion,"iou")