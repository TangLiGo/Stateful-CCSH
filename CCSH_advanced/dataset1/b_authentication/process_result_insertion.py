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
    pl.plot(x_data, benchmark_framelevel[1:len_quota], label='CCSH')
    # pl.plot(sample_nums[1:],F1_framelevel_insertion[1:], label='CCSH - Object Insertion')
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
    pl.savefig("../result/d1_fl_"+ben_name+"_double.pdf")


    pl.figure()
    pl.plot(x_data, benchmark_videolevel[1:len_quota], label='CCSH')
    # pl.plot(sample_nums[1:],F1_videolevel_insertion[1:], label='CCSH - Object Insertion')
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
    pl.savefig("../result/d1_vl_"+ben_name+"_double.pdf")
advanced_framelevel_path=[]
advanced_videolevel_path=[]
advanced_framelevel_insertion_path=[]
advanced_videolevel_insertion_path=[]
data_path="../data/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'roc_' in file and 'ahash' not in file :
        if 'insertion' in file:
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
    pl.plot(fpr_ahash_framelevel,tpr_ahash_framelevel,label='ahash',color='steelblue')
    pl.plot( fpr_framelevel_insertion[0], tpr_framelevel_insertion[0], label='baseline',color='darkorange')
   # pl.plot(fpr_framelevel_temp, tpr_framelevel_temp, label='CCSH')
    pl.plot(fpr_framelevel_insertion_temp, tpr_framelevel_insertion_temp, label='CCSH',color='seagreen')
    plt.legend(fontsize=12)
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    pl.savefig("../result/d1_roc_fl_"+str(i)+"_insertion.pdf")

    pl.figure()
    pl.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash',color='steelblue')
    pl.plot(fpr_videolevel_insertion[0], tpr_videolevel_insertion[0], label='baseline',color='darkorange')
  #  pl.plot(fpr_videolevel_temp, tpr_videolevel_temp, label='CCSH')
    pl.plot(fpr_videolevel_insertion_temp, tpr_videolevel_insertion_temp, label='CCSH',color='seagreen')
    plt.legend(fontsize=12)
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    pl.savefig("../result/d1_roc_vl_"+str(i)+"_insertion.pdf")




pl.figure()
pl.plot(sample_nums[:6],F1_framelevel_insertion[:6], label='CCSH',color='seagreen')

#pl.hlines((precision_ahash_framelevel+recall_ahash_framelevel+F1_ahash_framelevel+iou_ahash_framelevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_framelevel_insertion[0],0,5,color='darkorange',linestyles='dashed',label="baseline")
#pl.vlines(5,0.6,0.82,color='black',linestyles='dashed')
plt.legend(fontsize=12)
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数

plt.ylim(0.71,0.9)
#plt.ylim(0,1)
#plt.yscale('log')
ax.set_xlabel('$N_c$',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/d1_fl_single_insertion.pdf")


pl.figure()
pl.plot(sample_nums[:6],F1_videolevel_insertion[:6], label='CCSH',color='seagreen')

pl.hlines(F1_videolevel_insertion[0],0,5,color='darkorange',linestyles='dashed',label="baseline")

plt.legend(fontsize=12)
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
plt.ylim((0.9, 1))
#plt.ylim(0,1)
#plt.yscale('log')
ax.set_xlabel('$N_c$',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/d1_vl_single_insertion.pdf")