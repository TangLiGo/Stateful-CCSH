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
from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as pyplot
import imageio
a = 300
b = 500
x = 20
y = 20
w = 40
h = 80


def Gener_mat(a, b, x, y, w, h):  # 生成图片矩阵
    img_mat = np.zeros((a, b), dtype=np.int)
    for i in range(0, a):
        for j in range(0, b):
            img_mat[i][j] = 0
    for i in range(x, x + w):
        for j in range(y, y + h):
            img_mat[i][j] = 1
    return img_mat


def out_img(data):  # 输出图片
    new_im = Image.fromarray(data)  # 调用Image库，数组归一化
    # new_im.show()
    pyplot.imshow(data)  # 显示新图片
    imageio.imwrite('new_img.jpg', new_im)  # 保存图片到本地

print(np.sum(np.array([210, 329, 313, 319, 583, 262, 412, 274, 554, 239])))
img_mat = Gener_mat(a, b, x, y, w, h)
out_img(img_mat)
