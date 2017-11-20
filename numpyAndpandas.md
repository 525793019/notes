# <b id=title_numpy>[Numpy](#numpy)</b> && <b id=title_pandas> [Pandas](#pandas)</b>

### <b id=numpy>[Numpy](#title_numpy)</b>

* <big id=list_array>[numpy.array](#array)</big>
* numpy.arange
* numpy.linspace
---

* **安装**
```
$ sudo pip3 install numpy
```
* **引入**
```py
>>> import numpy as np
```
#### <b id=array>[numpy.array](#numpy)</b><small>(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)</small>
* Create an array.
```py
>>> np.array([1, 2, 3])
array([1, 2, 3])

# Upcasting:
>>> np.array([1, 2, 3.0])
array([ 1.,  2.,  3.])

# More than one dimension:
>>> np.array([[1, 2], [3, 4]])
array([[1, 2],
       [3, 4]])

# Minimum dimensions 2:
>>> np.array([1, 2, 3], ndmin=2)
array([[1, 2, 3]])

# Type provided:
>>> np.array([1, 2, 3], dtype=complex)
array([ 1.+0.j,  2.+0.j,  3.+0.j])

# Data-type consisting of more than one element:
>>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
>>> x['a']
array([1, 3])

# Creating an array from sub-classes:
>>> np.array(np.mat('1 2; 3 4'))
array([[1, 2],
       [3, 4]])
>>> np.array(np.mat('1 2; 3 4'), subok=True)
matrix([[1, 2],
        [3, 4]])
```
#### numpy.arange<small>([start, ]stop, [step, ]dtype=None)</small>
* Return evenly spaced values within a given interval.
* [start, stop)
* One dimension array
* For integer arguments the function is equivalent to the Python built-in range function, but returns an ndarray rather than a list.
```py
>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])
```
#### numpy.linspace<small>(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]</small>
* Return evenly spaced numbers over a specified interval.
* [start, stop].
* The endpoint of the interval can optionally be excluded.
```py
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([ 2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
```
* **array的属性**
```py
>>> a.ndim              # 维度
2
>>> a.shape             # 形状
(3,3)
>>> a.size              # 元素个数
9
```
* **元素类型**
```py
>>> b = np.array([1,2])
>>> b.dtype
dtype('int64')
>>> b = np.array([1,2],dtype=np.int32)      # 创建时指定元素类型
>>> b.dtype
dtype('int32')
```
* **方法**
```py
# reshape
print(c)                    #Out: [0 2 4 8 10]
print(c.reshape((2,3)))     #Out: [[0 2 4][6 8 10]]
```
```py
# 0矩阵
In [20]: e = np.zeros((3,4))

In [21]: e
Out[21]:
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
# 1矩阵
In [22]: f = np.ones((4,3))

In [23]: f
Out[23]:
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])

In [24]: type(f)
Out[24]: numpy.ndarray
# 默认dtype为float
In [25]: f.dtype
Out[25]: dtype('float64')
```
### <b id=pandas>[Pandas](#title_pandas)</b>

* **安装**
```python
sudo pip3 install pandas
```
