```python
from imagehash import dhash, average_hash, whash, phash
from thash.tucker_hash import tucker_hash
from utils import hash_collision_test, image_rotate_test, image_crop_test, make_dataset
import matplotlib.pyplot as plt
```


```python
hashes = {
    'dhash': dhash,
    'average_hash': average_hash,
    'tucker_hash': tucker_hash,
    'whash': whash,
    'phash': phash
}
```


```python
dataset = make_dataset()
```


```python
hash_collision_test(hashes, dataset)
```


![svg](hash_test_files/hash_test_3_0.svg)



```python
image_rotate_test(hashes, dataset, 1)
```


![svg](hash_test_files/hash_test_4_0.svg)



```python
image_rotate_test(hashes, dataset, 3)
```


![svg](hash_test_files/hash_test_5_0.svg)



```python
image_rotate_test(hashes, dataset, 5)
```


![svg](hash_test_files/hash_test_6_0.svg)



```python
image_rotate_test(hashes, dataset, 10)
```


![svg](hash_test_files/hash_test_7_0.svg)



```python
image_crop_test(hashes, dataset, 10)
```


![svg](hash_test_files/hash_test_8_0.svg)



```python
image_crop_test(hashes, dataset, 20)
```


![svg](hash_test_files/hash_test_9_0.svg)



```python
image_crop_test(hashes, dataset, 5)
```


![svg](hash_test_files/hash_test_10_0.svg)

