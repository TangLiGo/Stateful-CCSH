import numpy as np
import tensorly as tl
import binascii
import PIL
from imagehash import ImageHash
from tensorly.decomposition import tucker
from thash.image_preprocessing import preprocessing_image_array, io_image_to_array


def make_tensor_decomposition(
    l_matrix,
    u_block_size=2,
    q_block_size=32,
    i_core=1,
    j_core=1,
    k_core=1,
    random_state=1234,
):
    """ 
    
    Make transformations and Tucker decomposition of L component  
  
    Parameters
    ----------
    l_matrix : ndarray
        L image component
    u_block_size : int
        size of U blocks
    q_block_size : int
        size of QxQ blocks
    i_core : int
        first dimension of core tensor
    j_core : int
        second dimension of core tensor
    k_core : int
        third dimension of core tensor
    random_state : int
  
    Returns
    -------
    core : ndarray
    factors : ndarray list
  
    """
    l_matrix_shape = l_matrix.shape[0]
    u_matrix = (
        l_matrix.reshape((l_matrix_shape, l_matrix_shape))
        .reshape(
            l_matrix_shape // u_block_size,
            u_block_size,
            l_matrix_shape // u_block_size,
            u_block_size,
        )
        .mean(axis=(1, 3))
    )
    n = (l_matrix_shape // u_block_size // q_block_size) ** 2
    block_matrix = (
        u_matrix.reshape(
            u_matrix.shape[0] // q_block_size, q_block_size, -1, q_block_size
        )
        .swapaxes(1, 2)
        .reshape(-1, q_block_size, q_block_size)
    )
    x = tl.tensor(block_matrix, dtype="float")
    return tucker(x, rank=[i_core, j_core, k_core], random_state=random_state)


def make_hash(a_factor, b_factor, c_factor):
    a_p = a_factor.mean(axis=1)
    a_h = (a_p >= a_p.mean()).astype(int)
    b_p = b_factor.mean(axis=1)
    b_h = (b_p >= b_p.mean()).astype(int)
    c_p = c_factor.mean(axis=1)
    c_h = (c_p >= c_p.mean()).astype(int)
    return np.hstack((a_h, b_h, c_h))


def tucker_hash(img, **kwargs):

    l_matrix = preprocessing_image_array(img)
    _, factors = make_tensor_decomposition(l_matrix, **kwargs)
    a_factor, b_factor, c_factor = factors[0], factors[1], factors[2]
    print("a,b,c size", a_factor.shape, b_factor.shape, c_factor.shape)
    return make_hash(a_factor, b_factor, c_factor).reshape((-1, 8))


def binary_array_to_hex(arr):
    arr = arr.reshape((-1, 8))
    return binascii.hexlify(np.packbits(arr, axis=1)).decode("utf8")
