import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import matplotlib.pyplot as plt
import math
from PIL import Image
from itertools import combinations
from skimage import data, img_as_float, color, io

plt.style.use('seaborn')


def make_dataset(folder='./data/Pasadena-Houses/', img_format='jpg'):
    images = list()
    for _path in glob.glob(f'{folder}*.{img_format}'):
        images.append(Image.open(_path))
    return images


def bootstrap_prob(array, threshold, iteration=1000):
    probs = np.empty((iteration,))
    for i in range(iteration):
        sample = np.random.choice(array, size=array.shape)
        probs[i] = np.sum(sample < threshold) / array.shape[0]
    return np.mean(probs), np.percentile(probs, 2.5), np.percentile(probs, 97.5)


def hash_collision_test(hash_dict, dataset):
    labels = list()
    hash_array = dict()
    colors = sns.color_palette('Dark2', len(hash_dict))
    gen_colors = (i for i in colors)
    fig, (ax, ax_table) = plt.subplots(
        nrows=2, figsize = (10,15), gridspec_kw=dict(height_ratios=[3,1])
    )
    
    for hash_name, hash_func in hash_dict.items():
        img_comb = combinations(map(hash_func, dataset), 2)
        hash_len = len(hash_func(dataset[0]).hash.flatten())
        diff = list(map(lambda x: (x[0] - x[1])/hash_len, img_comb))
        hash_array[hash_name] = np.array(diff) 
        sns.kdeplot(
            hash_array[hash_name], ax=ax, 
            label=hash_name, shade=True, color=next(gen_colors)
        )
        labels.append(hash_name)
    
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    cellText = list()
    for hash_name, hash_array in hash_array.items():
        row = list()
        for thr in thresholds:
            mean, pr_1, pr_2 = bootstrap_prob(hash_array, thr)
            row.append(f'{mean:.2E}')
        cellText.append(row)
    
    ax.legend()
    ax.set_title('Hamming distance distrtibution')
    ax.set_xlabel('Normalized hamming distance')

    ax_table.axis('off')
    ax_table.set_title('Probability of collision for different thresholds')
    ax_table = ax_table.table(
        cellText=cellText, rowLabels=labels,
        colLabels=thresholds, loc='upper center',
        rowColours=colors
    )
  #  return fig


def image_rotate(img, degree):
    size = img.size
    img_mod = img.rotate(degree)
    t1 = size[0] / 2 * (1 - math.cos((45+degree)*math.pi / 180)/ math.cos(45*math.pi / 180))
    t2 = size[1] / 2 * (1 - math.cos((45+degree)*math.pi / 180)/ math.cos(45*math.pi / 180))
    box = (t1, t2, size[0] - t1, size[1] - t2)
    return img_mod.crop(box)

def image_crop(img, scale):
    size = img.size
    t1 = size[0] - size[0] / (scale / 100 + 1)
    t2 = size[1] - size[1] / (scale / 100 + 1)
    box = (t1, t2, size[0]-t1, size[1]-t2)
    return img.crop(box)

def image_modification_test(hash_dict, dataset, mod, parameter):
    labels = list()
    hash_array = dict()
    colors = sns.color_palette('Dark2', len(hash_dict))
    gen_colors = (i for i in colors)
    fig, (ax, ax_table) = plt.subplots(
        nrows=2, figsize=(10, 13), gridspec_kw=dict(height_ratios=[4, 1])
    )
    for hash_name, hash_func in hash_dict.items():
        hash_len = len(hash_func(dataset[0]).hash.flatten())
        dataset_mod = map(lambda x: mod(x, parameter), dataset)
        diff = list(map(lambda x: (hash_func(x[0]) - hash_func(x[1])) / hash_len, zip(dataset, dataset_mod)))
        hash_array[hash_name] = np.array(diff) 
        sns.kdeplot(
            hash_array[hash_name], ax=ax, 
            label=hash_name, shade=True, color=next(gen_colors)
            )
        labels.append(hash_name)
    
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    cellText = list()
    for hash_name, hash_array in hash_array.items():
        row = list() 
        for thr in thresholds:
            mean, pr_1, pr_2 = bootstrap_prob(hash_array, thr)
            row.append(f'{mean:.2E}')
        cellText.append(row)
    
    ax.legend()
    ax.set_title('Hamming distance distrtibution')
    ax.set_xlabel('Normalized hamming distance')

    ax_table.axis('off')
    ax_table.set_title('Probability of collision for different thresholds')
    ax_table = ax_table.table(
        cellText=cellText, rowLabels=labels,
        colLabels=thresholds, loc='upper center',
        rowColours=colors
    )
   # return fig


def image_rotate_test(hash_dict, dataset, degree):
    image_modification_test(hash_dict, dataset, image_rotate, degree)

def image_crop_test(hash_dict, dataset, scale):
    image_modification_test(hash_dict, dataset, image_crop, scale)