import pytest
from PIL import Image
from thash.tucker_hash import tucker_hash
from itertools import combinations


def test_image_house():
    img = Image.open('test_data/calhouse_0081.jpg')
    print(str(tucker_hash(img)) )
    assert str(tucker_hash(img)) == '68ce0ff807fe0003ffff' 


def test_initial_state_ind():
    img = Image.open('test_data/calhouse_0081.jpg')
    arr_img = (tucker_hash(img) for _ in range(200))
    arr_img = map(lambda x: abs(x[0] - x[1]), combinations(arr_img, 2))
    print("arr_img",arr_img)
    assert sum(arr_img) == 0

test_image_house()
test_initial_state_ind()