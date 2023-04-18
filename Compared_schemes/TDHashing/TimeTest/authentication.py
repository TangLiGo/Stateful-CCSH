import hashlib
import numpy as np
from moviepy.editor import VideoFileClip
from skimage.transform import resize
import imagehash
from PIL import Image
import cv2
import distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import skimage
import math
import timeit
from time import process_time
import hashlib
import numpy as np
from moviepy.editor import VideoFileClip
from skimage.transform import resize
import imagehash
from PIL import Image
import cv2
import distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import skimage
import math
import timeit
from time import process_time
import sys
import random
from PIL import Image, ImageFilter
from skimage import io, color,filters
import numpy as np
import tensorly as tl
import binascii
import PIL
from imagehash import ImageHash
from tensorly.decomposition import tucker
from PIL import Image
#from thash.tucker_hash import tucker_hash
from itertools import combinations
from scipy.signal import convolve2d
from skimage.color import rgb2lab
import pywt
'''
算术平均滤波法
'''
import os
from matplotlib.pyplot import MultipleLocator
from time import process_time
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


def data_processing(hds,low_threshold):
    window_size=10
    high_threshold=16
    filter_result = SlidingAverage(hds, window_size)
    result = self_filter(filter_result, high_threshold, low_threshold)

    return


path_CCSH='../CCSH/dataset4/data/hds_ahash_sample_advanced_02_deletion.npy'
path_baseline2='../Baseline2/dataset4/data/hds_ahash_sample_advanced_02_deletion.npy'
path_baseline1='../Baseline1/dataset4/data/hds_ahash_sample_advanced_sample_020.npy'
hds_CCSH=np.load(path_CCSH, allow_pickle=True)
hds_baseline2=np.load(path_baseline2, allow_pickle=True)
hds_baseline1=np.load(path_baseline1, allow_pickle=True)
low_threshold=1.15
repeatTime=1000
frame_len=len(hds_CCSH[0][15])
hds=[hds_CCSH,hds_baseline2,hds_baseline1]
time_costs=[]
time_costs_video=[]
for hd in hds:
    time_start = process_time()
    for i in range(repeatTime):
        data_processing(hd[0][15],low_threshold)
    time_end = process_time()
    time_costs.append((time_end - time_start) / (repeatTime * frame_len))
    time_costs_video.append((time_end - time_start) / (repeatTime ))
print(time_costs)
print(time_costs_video)
print(frame_len)

#video 4: 908 len [1.4282764317180617e-06, 1.411068281938326e-06, 1.411068281938326e-06]

