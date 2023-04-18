
import numpy as np
import imagehash
from PIL import Image
import cv2
import distance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import timeit
from time import process_time
def bits_to_hexhash(bits):
    return '{0:0={width}x}'.format(int(''.join([str(x) for x in bits]), 2), width = len(bits) // 4)
def hash_compare(method,video_path1,video_path2):
    #print(video_path1,video_path2)
    (path, temp_file_name) = os.path.split(video_path1)
    (output_file_name1,extension)=os.path.splitext(temp_file_name)
    (path, temp_file_name) = os.path.split(video_path2)
    (output_file_name2,extension)=os.path.splitext(temp_file_name)
    img_name=output_file_name1+'_'+output_file_name2+'_.png'


    clip = cv2.VideoCapture(video_path1)
    clip_forged = cv2.VideoCapture(video_path2)
    imgs = []
    imgs_forged = []
    while True:
        success, frame = clip.read()
        if not success:
            break
        imgs.append(frame)
    while True:
        success, frame = clip_forged.read()
        if not success:
            break
        imgs_forged.append(frame)
    print(video_path1,video_path2)
    print(len(imgs),len(imgs_forged))
    frame_hash_values = []
    frame_hash_values_forged = []
    index=0
    frame_hash_values_hex = []
    frame_hash_values_forged_hex=[]
    for f in imgs:
        im=Image.fromarray(f)
        cur_frame_hash = method(im)

        frame_hash_values.append((cur_frame_hash))

    for f in imgs_forged:
        im=Image.fromarray(f)
        cur_frame_hash = method(im)
        frame_hash_values_forged.append((cur_frame_hash))

    forged_frame_indices = []

    for i in range(min(len(frame_hash_values),len(frame_hash_values_forged))):
        #print(frame_hash_values[i])
        hd = frame_hash_values[i]-frame_hash_values_forged[i]
        forged_frame_indices.append(hd)
    plt.figure()

    plt.plot(forged_frame_indices)
    plt.savefig("../result/ahash_"+img_name)
    #print(forged_threshold_indices)
  #  if "10_forged" in output_file_name1 or "10_original" in output_file_name1:
     #   plt.figure()

    #    plt.plot(forged_frame_indices)
     #   plt.savefig("D:/TangLi/results/dataset1/a_comparision/"+img_name)
     #   print("frame_hash_values",frame_hash_values_hex)
    #    print("frame_hash_values_forged",frame_hash_values_forged_hex)
    #    print(os.path.splitext(img_name)[0], forged_frame_indices)

    return forged_frame_indices

def compare_hd():
    data_path0 = 'C:/Users/tangli/Desktop/dataset/dataset1/h264_lossless'
    data_path1 = 'C:/Users/tangli/Desktop/dataset/dataset1/h264_lossy_q10'
    data_path2 = 'C:/Users/tangli/Desktop/dataset/dataset1/h264_lossy_q20'
    data_path3 = 'C:/Users/tangli/Desktop/dataset/dataset1/h264_lossy_q30'
    data_path=[data_path0,data_path1,data_path2,data_path3]
    original_videos=[[],[],[],[]]
    forged_videos=[[],[],[],[]]

    for i in range(len(data_path)):
        path=data_path[i]
        for file in os.listdir(path):
            video_path = os.path.join(path, file)
            video_path = video_path.replace('\\', '/')
            if 'original' in file:
                original_videos[i].append(video_path)
            elif 'forged' in file:
                forged_videos[i].append(video_path)
    #print("info", original_videos, forged_videos)
    hds_forgery=[]
    for i in range(len(original_videos)):
        for j in range(len(forged_videos)):
            sub_hds_forgery = []
            for k in range(len(original_videos[i])):

                sub_hds_forgery.append(hash_compare(imagehash.average_hash, original_videos[i][k], forged_videos[j][k]))
            hds_forgery.append(sub_hds_forgery)

    hds_compression=[]
    for i in range(len(original_videos)-1):
        for j in range(i+1,len(original_videos)):
            sub_hds_compression = []
            for k in range(len(original_videos[i])):

                sub_hds_compression.append(hash_compare(imagehash.average_hash, original_videos[i][k], original_videos[j][k]))
            hds_compression.append(sub_hds_compression)
    for i in range(len(forged_videos)-1):
        for j in range(i+1,len(forged_videos)):
            sub_hds_compression = []
            for k in range(len(forged_videos[i])):

                sub_hds_compression.append(hash_compare(imagehash.average_hash, forged_videos[i][k], forged_videos[j][k]))
            hds_compression.append(sub_hds_compression)
    print("The hd info for forgery detection:",hds_forgery)
    print("The hd info for compression",hds_compression)


    return hds_forgery,hds_compression

time_start=process_time()
hds_forgery_mul=[]
hds_compression_mul=[]
repeatTime=1
for i in range(repeatTime):
    temp_forgery,temp_compression=compare_hd()
    hds_forgery_mul.extend(temp_forgery)
    hds_compression_mul.extend(temp_compression)

hds_forgery_arr=np.array(hds_forgery_mul,dtype=object)
np.save('../../data/dataset1/hds_ahash_test.npy',hds_forgery_arr)
hds_compression_arr=np.array(hds_compression_mul,dtype=object)
np.save('../../data/dataset1/hds_ahash_test_compression.npy',hds_compression_arr)
time_end=process_time()
print("finish",time_end-time_start)#defaluter_timer()68.4676604s




