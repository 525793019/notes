# <b id=title_numpy>[Numpy](#numpy)</b> && <b id=title_pandas> [Pandas](#pandas)</b>

### Install
```
$ sudo pip3 install numpy
$ sudo pip3 install pandas
```

### <b id=numpy>[Numpy](#title_numpy)</b>

* [array](#array)
* [arange](#arange)
* [linspace](#linspace)
* [zeros](#zeros)
* [ones](#ones)
* <font id=attributes>Array attributes</font>
    * [ndim](#ndim)
    * [shape](#shape)
    * [size](#size)
    * [dtype](#dtype)
* <font id=methods>Array methods</font>
    * [reshape](#reshape)
---

* **引入**
```py
>>> import numpy as np
```
#### <big id=array>[numpy.array](#numpy)</big><small>(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)</small>
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
#### <big id=arange>[numpy.arange](#numpy)</big><small>([start, ]stop, [step, ]dtype=None)</small>
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
#### <big id=linspace>[numpy.linspace](#numpy)</big><small>(start, stop, num=50, endpoint=True, retstep=False, dtype=None)</small>
* Return evenly spaced numbers over a specified interval.
* [start, stop].
* The endpoint of the interval can optionally be excluded.
* [[source]](http://github.com/numpy/numpy/blob/v1.13.0/numpy/core/function_base.py#L25-L143)
```py
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([ 2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
```
#### <big id=zeros>[numpy.zeros](#numpy)</big><small>(shape, dtype=float, order='C')</small>
* Return a new array of given shape and type, filled with zeros.
```py
>>> np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])
>>> np.zeros((5,), dtype=np.int)
array([0, 0, 0, 0, 0])
>>> np.zeros((2, 1))
array([[ 0.],
       [ 0.]])
>>> s = (2,2)
>>> np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])
>>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
array([(0, 0), (0, 0)],
      dtype=[('x', '<i4'), ('y', '<i4')])
```
#### <big id=ones>[numpy.ones](#numpy)</big><small>(shape, dtype=float, order='C')</small>
* Return a new array of given shape and type, filled with ones.
```py
>>> np.ones(5)
array([ 1.,  1.,  1.,  1.,  1.])
>>> np.ones((5,), dtype=np.int)
array([1, 1, 1, 1, 1])
>>> np.ones((2, 1))
array([[ 1.],
       [ 1.]])
>>> s = (2,2)
>>> np.ones(s)
array([[ 1.,  1.],
       [ 1.,  1.]])
```
#### <big>Array attributes</big>
#### <b id=ndim>[array.ndim](#attributes)</b>
* Number of array dimensions.
```py
>>> x = np.array([1, 2, 3])
>>> x.ndim
1
>>> y = np.zeros((2, 3, 4))
>>> y.ndim
3
```
#### <b id=shape>[array.shape](#attributes)</b>
* Tuple of array dimensions.
```py
>>> x = np.array([1, 2, 3, 4])
>>> x.shape
(4,)
>>> y = np.zeros((2, 3, 4))
>>> y.shape
(2, 3, 4)
>>> y.shape = (3, 8)
>>> y
array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
>>> y.shape = (3, 6)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: total size of new array must be unchanged
```
#### <b id=size>[array.size](#attributes)</b>
* Number of elements in the array.
```py
>>> x = np.zeros((3, 5, 2), dtype=np.complex128)
>>> x.size
30
>>> np.prod(x.shape)
30
```
#### <b id=dtype>[array.type](#attributes)</b>
* Data-type of the array’s elements.
```py
>>> b = np.array([1,2])
>>> b.dtype
dtype('int64')
>>> b = np.array([1,2],dtype=np.int32)      # 创建时指定元素类型
>>> b.dtype
dtype('int32')
>>> type(b.dtype)
<type 'numpy.dtype'>
```
#### <big>Array methods</big>
#### <big id=reshape>[array.reshape](#methods)</big><small>(shape, order='C')</small>
* Returns an array containing the same data with a new shape.
```py
>>> a = np.arange(6).reshape((3, 2))
>>> a
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> np.reshape(a, (2, 3)) # C-like index ordering
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
array([[0, 4, 3],
       [2, 1, 5]])
>>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
array([[0, 4, 3],
       [2, 1, 5]])
```
### <b id=pandas>[Pandas](#title_pandas)</b>

