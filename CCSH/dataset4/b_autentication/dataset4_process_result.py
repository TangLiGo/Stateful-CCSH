import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import os
from matplotlib.pyplot import MultipleLocator

def drawBenchmarks(len_quota,benchmark_framelevel,benchmark_videolevel,ben_name):
    x_data=range(1, len_quota)

    pl.figure()
    pl.plot(x_data, benchmark_framelevel[1:len_quota], label='CCSH')
    # pl.plot(sample_nums[1:],benchmarks_framelevel_insertion[1:], label='CCSH - Object Insertion')
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
    plt.ylabel('Detection Performance')
    pl.savefig("../result/d4_fl_"+ben_name+"_single.pdf")


    pl.figure()
    pl.plot(x_data, benchmark_videolevel[1:len_quota], label='CCSH')
    # pl.plot(sample_nums[1:],benchmarks_videolevel_insertion[1:], label='CCSH - Object Insertion')
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
    plt.ylabel('Detection Performance')
    pl.savefig("../result/d4_vl_"+ben_name+"_single.pdf")
advanced_framelevel_path=[]
advanced_videolevel_path=[]
advanced_framelevel_insertion_path=[]
advanced_videolevel_insertion_path=[]
data_path="../data/"
for file in os.listdir(data_path):
    video_path = os.path.join(data_path, file)
    video_path = video_path.replace('\\', '/')
    if 'roc_' in file and 'ahash' not in file :

            if 'framelevel' in file:
                advanced_framelevel_path.append(video_path)
            elif 'videolevel' in file:
                advanced_videolevel_path.append(video_path)

print(advanced_videolevel_path)

ahash_framelevel = np.load('../data/roc_framelevel_data_ahash.npy', allow_pickle=True)
ahash_videolevel = np.load('../data/roc_videolevel_data_ahash.npy', allow_pickle=True)

advanced_framelevel=[]
advanced_videolevel=[]
advanced_framelevel_insertion=[]
advanced_videolevel_insertion=[]
for path in advanced_framelevel_path:
    advanced_framelevel.append(np.load(path, allow_pickle=True))
for path in advanced_videolevel_path:
    advanced_videolevel.append(np.load(path, allow_pickle=True))

len_quota=6
sample_nums = range(len_quota)



fpr_ahash_framelevel,tpr_ahash_framelevel,precision_ahash_framelevel,recall_ahash_framelevel,F1_ahash_framelevel,iou_ahash_framelevel=ahash_framelevel
fpr_ahash_videolevel,tpr_ahash_videolevel,precision_ahash_videolevel,recall_ahash_videolevel,F1_ahash_videolevel,iou_ahash_videolevel=ahash_videolevel

fpr_framelevel = []
tpr_framelevel = []
precision_framelevel=[]
recall_framelevel=[]
F1_framelevel=[]
iou_framelevel=[]
fpr_framelevel_insertion = []
tpr_framelevel_insertion = []
precision_framelevel_insertion=[]
recall_framelevel_insertion=[]
F1_framelevel_insertion=[]
iou_framelevel_insertion=[]

fpr_videolevel = []
tpr_videolevel = []
precision_videolevel=[]
recall_videolevel=[]
F1_videolevel=[]
iou_videolevel=[]
fpr_videolevel_insertion = []
tpr_videolevel_insertion = []
precision_videolevel_insertion=[]
recall_videolevel_insertion=[]
F1_videolevel_insertion=[]
iou_videolevel_insertion=[]


for i in range(len_quota):

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
    pl.plot(fpr_ahash_framelevel,tpr_ahash_framelevel,label='ahash',color='steelblue')
    pl.plot(fpr_framelevel[0],  tpr_framelevel[0],  label='baseline',color='darkorange')
    pl.plot(fpr_framelevel_temp, tpr_framelevel_temp, label='CCSH',color='seagreen')

    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    pl.savefig("../result/d4_roc_fl_"+str(i)+".pdf")

    pl.figure()
    pl.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash',color='steelblue')
    pl.plot(fpr_videolevel[0],  tpr_videolevel[0],  label='baseline',color='darkorange')
    pl.plot(fpr_videolevel_temp, tpr_videolevel_temp, label='CCSH',color='seagreen')

    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pl.savefig("../result/d4_roc_vl_"+str(i)+".pdf")

    pl.figure()
    fig,ax=pl.subplots(1,1)
    ax.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash',color='steelblue')
    ax.plot(fpr_videolevel[0],  tpr_videolevel[0],  label='baseline',color='darkorange')
    ax.plot(fpr_videolevel_temp, tpr_videolevel_temp, label='CCSH',color='seagreen')
    ax.legend(fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    axins = ax.inset_axes((0.6, 0.2, 0.3, 0.3))
    axins.plot(fpr_videolevel_temp,
               tpr_videolevel_temp,color='seagreen')
    axins.set_xlim(0,0.04)
    axins.set_ylim(0.96, 1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    pl.savefig("../result/d4_roc_vl_"+str(i)+"_focus.pdf")

    pl.figure()
    fig,ax=pl.subplots(1,1)
    ax.plot(fpr_ahash_videolevel,tpr_ahash_videolevel,label='ahash',color='steelblue')
    ax.plot(fpr_videolevel[0],  tpr_videolevel[0],  label='baseline',color='darkorange')
    ax.plot(fpr_videolevel_temp, tpr_videolevel_temp, label='CCSH',color='seagreen')
    ax.legend(fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    axins = ax.inset_axes((0.6, 0.2, 0.3, 0.3))
    axins.plot(fpr_videolevel_temp,
               tpr_videolevel_temp,color='seagreen')
    axins.plot(fpr_videolevel[0],
               tpr_videolevel[0],color='darkorange')
    axins.set_xlim(0,0.1)
    axins.set_ylim(0.9, 1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
    pl.savefig("../result/d4_roc_vl_"+str(i)+"_focus2.pdf")



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
pl.savefig("../result/d4_fl_single1.pdf")

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
pl.savefig("../result/d4_vl_single1.pdf")

pl.figure()
pl.plot(sample_nums[:6],F1_framelevel[:6], label='CCSH',color='seagreen')

#pl.hlines((precision_ahash_framelevel+recall_ahash_framelevel+F1_ahash_framelevel+iou_ahash_framelevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_framelevel[0],0,5,color='darkorange',linestyles='dashed',label="baseline")
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
ax.set_xlabel('rN',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/d4_fl_single2.pdf")

pl.figure()
pl.plot(sample_nums[:6],F1_videolevel[:6], label='CCSH',color='seagreen')

#pl.hlines((precision_ahash_videolevel+recall_ahash_videolevel+F1_ahash_videolevel+iou_ahash_videolevel)/4,0,10,color='blue',linestyles='dashed',label="ahash")
pl.hlines(F1_videolevel[0],0,5,color='darkorange',linestyles='dashed',label="baseline")

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
ax.set_xlabel('rN',fontsize=12)
ax.set_ylabel('F1',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
pl.savefig("../result/d4_vl_single2.pdf")






