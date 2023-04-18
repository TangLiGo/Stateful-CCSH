import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rescale, resize
from skimage.color import rgb2lab
from skimage.io import imread


def preprocessing_image_array(img_array, size=(256, 256)):
    img_array = resize(img_array, size, anti_aliasing=True)
    L = rgb2lab(img_array)[:, :, 0]
    GaussKernel = (
        np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ]
        )
        / 256
    )
    blured = convolve2d(
        L, GaussKernel, boundary="fill", mode="same", fillvalue=np.mean(L)
    )
    return blured


def io_image_to_array(path):
    img_array = imread(path)
    return img_array
