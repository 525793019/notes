# <b id=title_numpy>[Numpy](#numpy)</b> && <b id=title_pandas> [Pandas](#pandas)</b>

### Install

```
$ sudo pip3 install numpy
$ sudo pip3 install pandas
```

### Import

```python
>>> import numpy as np
>>> import pandas as pd
```

### <b id=numpy>[Numpy](#title_numpy)</b>

* [array](#array)
* [arange](#arange)
* [linspace](#linspace)
* [zeros](#zeros)
* [ones](#ones)
* [random.random](#random)
* [sum](#sum)
* [max](#max)
* [min](#min)
* [argsort](#argsort)
* [argmax](#argmax)
* [argmin](#argmin)
* [mean](#mean)
* [median](#median)
* [cumsum](#cumsum)
* [diff](#diff)
* [nonzero](#nonzero)
* [clip](#clip)
* [ravel](#ravel)
* [stack](#stack)
* [concatenate](#concatenate)
* [vstack](#vstack)
* [hstack](#hstack)
* [array_split](#array_split)
* [split](#split)
* [vsplit](#vsplit)
* [hsplit](#hsplit)
* [newaxis](#newaxis)
* <b id=attributes>Array attributes</b>
    * [ndim](#ndim)
    * [shape](#shape)
    * [size](#size)
    * [dtype](#dtype)
    * [flat](#flat)
* <b id=methods>Array methods</b>
    * [reshape](#reshape)
    * [dot](#dot)
    * [sort](#sort)
    * [transpose](#transpose)
    * [T](#T)
    * [flatten](#flatten)
    * [copy](#copy)
* <b id=compare>Compare</b>
    * [ravel vs flatten](#c1)
    * [stack vs concatenate](#c2)
    * [shape vs newaxis](#c3)
    * [array_split vs split](#c4)
---

#### <big id=array>[numpy.array](#numpy)</big><small>(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)</small>
* Create an array.

```python
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
#### <big id=random>[numpy.random.random](#numpy)</big><small>(size=None)</small>
* Return random floats in the half-open interval [0.0, 1.0).
```py
>>> np.random.random_sample()
0.47108547995356098
>>> type(np.random.random_sample())
<type 'float'>
>>> np.random.random_sample((5,))
array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

Three-by-two array of random numbers from [-5, 0):

>>> 5 * np.random.random_sample((3, 2)) - 5
array([[-3.99149989, -0.52338984],
       [-2.99091858, -0.79479508],
       [-1.23204345, -1.75224494]])
```
#### <big id=sum>[numpy.sum](#numpy)</big><small>(a, axis=None, dtype=None, out=None, keepdims=\<class numpy._globals._NoValue\>)</small>
* Sum of array elements over a given axis.
```py
>>> np.sum([])
0.0
>>> np.sum([0.5, 1.5])
2.0
>>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
1
>>> np.sum([[0, 1], [0, 5]])
6
>>> np.sum([[0, 1], [0, 5]], axis=0)
array([0, 6])
>>> np.sum([[0, 1], [0, 5]], axis=1)
array([1, 5])

If the accumulator is too small, overflow occurs:

>>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
-128
```
#### <big id=max>[numpy.max](#numpy)</big><small>(a, axis=None, out=None, keepdims=\<class numpy._globals._NoValue\>)</small>
* Return the maximum of an array or maximum along an axis.
```py
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.amax(a)           # Maximum of the flattened array
3
>>> np.amax(a, axis=0)   # Maxima along the first axis
array([2, 3])
>>> np.amax(a, axis=1)   # Maxima along the second axis
array([1, 3])
>>> b = np.arange(5, dtype=np.float)
>>> b[2] = np.NaN
>>> np.amax(b)
nan
>>> np.nanmax(b)
4.0
```

#### <big id=min>[numpy.min](#numpy)</big><small>(a, axis=None, out=None, keepdims=\<class numpy._globals._NoValue\>)</small>
* Return the minimum of an array or minimum along an axis.
```py
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.amin(a)           # Minimum of the flattened array
0
>>> np.amin(a, axis=0)   # Minima along the first axis
array([0, 1])
>>> np.amin(a, axis=1)   # Minima along the second axis
array([0, 2])
>>> b = np.arange(5, dtype=np.float)
>>> b[2] = np.NaN
>>> np.amin(b)
nan
>>> np.nanmin(b)
0.0
```
#### <big id=argsort>[numpy.argsort](#numpy)</big><small>(a, axis=-1, kind='quicksort', order=None)</small>
* Returns the indices that would sort an array.
```py
One dimensional array:

>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])

Two-dimensional array:

>>> x = np.array([[0, 3], [2, 2]])
>>> x
array([[0, 3],
       [2, 2]])

>>> np.argsort(x, axis=0)
array([[0, 1],
       [1, 0]])

>>> np.argsort(x, axis=1)
array([[0, 1],
       [0, 1]])

Sorting with keys:

>>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
>>> x
array([(1, 0), (0, 1)],
      dtype=[('x', '<i4'), ('y', '<i4')])

>>> np.argsort(x, order=('x','y'))
array([1, 0])

>>> np.argsort(x, order=('y','x'))
array([0, 1])
```
#### <big id=argmax>[numpy.argmax](#numpy)</big><small>(a, axis=None, out=None)</small>
* Returns the indices of the maximum values along an axis.
```py
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])

>>> b = np.arange(6)
>>> b[1] = 5
>>> b
array([0, 5, 2, 3, 4, 5])
>>> np.argmax(b) # Only the first occurrence is returned.
1
```
#### <big id=argmin>[numpy.argmin](#numpy)</big><small>(a, axis=None, out=None)</small>
* Returns the indices of the minimum values along an axis.
```py
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argmin(a)
0
>>> np.argmin(a, axis=0)
array([0, 0, 0])
>>> np.argmin(a, axis=1)
array([0, 0])

>>> b = np.arange(6)
>>> b[4] = 0
>>> b
array([0, 1, 2, 3, 0, 5])
>>> np.argmin(b) # Only the first occurrence is returned.
0
```
#### <big id=mean>[numpy.mean](#numpy)</big><small>(a, axis=None, dtype=None, out=None, keepdims=\<class numpy._globals._NoValue\>)</small>
* Compute the arithmetic mean along the specified axis.
```py
>>> a = np.array([[1, 2], [3, 4]])
>>> np.mean(a)
2.5
>>> np.mean(a, axis=0)
array([ 2.,  3.])
>>> np.mean(a, axis=1)
array([ 1.5,  3.5])

In single precision, mean can be inaccurate:

>>> a = np.zeros((2, 512*512), dtype=np.float32)
>>> a[0, :] = 1.0
>>> a[1, :] = 0.1
>>> np.mean(a)
0.54999924

Computing the mean in float64 is more accurate:

>>> np.mean(a, dtype=np.float64)
0.55000000074505806
```
#### <big id=median>[numpy.median](#numpy)</big><small>(a, axis=None, out=None, overwrite_input=False, keepdims=False)</small>
* Compute the median along the specified axis.
* Returns the median of the array elements.
```py
>>> a = np.array([[10, 7, 4], [3, 2, 1]])
>>> a
array([[10,  7,  4],
       [ 3,  2,  1]])
>>> np.median(a)
3.5
>>> np.median(a, axis=0)
array([ 6.5,  4.5,  2.5])
>>> np.median(a, axis=1)
array([ 7.,  2.])
>>> m = np.median(a, axis=0)
>>> out = np.zeros_like(m)
>>> np.median(a, axis=0, out=m)
array([ 6.5,  4.5,  2.5])
>>> m
array([ 6.5,  4.5,  2.5])
>>> b = a.copy()
>>> np.median(b, axis=1, overwrite_input=True)
array([ 7.,  2.])
>>> assert not np.all(a==b)
>>> b = a.copy()
>>> np.median(b, axis=None, overwrite_input=True)
3.5
>>> assert not np.all(a==b)
```
#### <big id=cumsum>[numpy.cumsum](#numpy)</big><small>(a, axis=None, dtype=None, out=None)</small>
* Return the cumulative sum of the elements along a given axis.
```py
>>> a = np.array([[1,2,3], [4,5,6]])
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.cumsum(a)
array([ 1,  3,  6, 10, 15, 21])
>>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
array([  1.,   3.,   6.,  10.,  15.,  21.])

>>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
array([[1, 2, 3],
       [5, 7, 9]])
>>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
array([[ 1,  3,  6],
       [ 4,  9, 15]])
```
#### <big id=diff>[numpy.diff](#numpy)</big><small>(a, n=1, axis=-1)</small>
* Calculate the n-th discrete difference along given axis.
* The first difference is given by out[n] = a[n+1] - a[n] along the given axis, higher differences are calculated by using diff recursively.
```py
>>> x = np.array([1, 2, 4, 7, 0])
>>> np.diff(x)
array([ 1,  2,  3, -7])
>>> np.diff(x, n=2)
array([  1,   1, -10])

>>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
>>> np.diff(x)
array([[2, 3, 4],
       [5, 1, 2]])
>>> np.diff(x, axis=0)
array([[-1,  2,  0, -2]])
```
#### <big id=nonzero>[numpy.nonzero](#numpy)</big><small>(a)</small>
* Return the indices of the elements that are non-zero.
```py
>>> x = np.array([[1,0,0], [0,2,0], [1,1,0]])
>>> x
array([[1, 0, 0],
       [0, 2, 0],
       [1, 1, 0]])
>>> np.nonzero(x)
(array([0, 1, 2, 2], dtype=int64), array([0, 1, 0, 1], dtype=int64))

>>> x[np.nonzero(x)]
array([ 1.,  1.,  1.])
>>> np.transpose(np.nonzero(x))
array([[0, 0],
       [1, 1],
       [2, 2]])

A common use for nonzero is to find the indices of an array, where a condition is True. Given an array a, the condition a > 3 is a boolean array and since False is interpreted as 0, np.nonzero(a > 3) yields the indices of the a where the condition is true.

>>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> a > 3
array([[False, False, False],
       [ True,  True,  True],
       [ True,  True,  True]], dtype=bool)
>>> np.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

The nonzero method of the boolean array can also be called.

>>> (a > 3).nonzero()
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
```
#### <big id=clip>[numpy.clip](#numpy)</big><small>(a, a_min, a_max, out=None)</small>
* Clip (limit) the values in an array.
* Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
```py
>>> a = np.arange(10)
>>> np.clip(a, 1, 8)
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, 3, 6, out=a)
array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
```
#### <big id=ravel>[numpy.ravel](#numpy)</big><small>(a, order='C')</small>
* Return a contiguous flattened array.
```py
It is equivalent to reshape(-1, order=order).

>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> print(np.ravel(x))
[1 2 3 4 5 6]
>>> print(x.reshape(-1))
[1 2 3 4 5 6]
>>> print(np.ravel(x, order='F'))
[1 4 2 5 3 6]

When order is ‘A’, it will preserve the array’s ‘C’ or ‘F’ ordering:

>>> print(np.ravel(x.T))
[1 4 2 5 3 6]
>>> print(np.ravel(x.T, order='A'))
[1 2 3 4 5 6]

When order is ‘K’, it will preserve orderings that are neither ‘C’ nor ‘F’, but won’t reverse axes:

>>> a = np.arange(3)[::-1]; a
array([2, 1, 0])
>>> a.ravel(order='C')
array([2, 1, 0])
>>> a.ravel(order='K')
array([2, 1, 0])

>>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
array([[[ 0,  2,  4],
        [ 1,  3,  5]],
       [[ 6,  8, 10],
        [ 7,  9, 11]]])
>>> a.ravel(order='C')
array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
>>> a.ravel(order='K')
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```
#### <big id=stack>[numpy.stack](#numpy)</big><small>(arrays, axis=0)</small>
* Join a sequence of arrays along a **new axis**.
* The axis parameter specifies the index of the new axis in the dimensions of the result. For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
* New in version 1.10.0.
```py
>>> arrays = [np.random.randn(3, 4) for _ in range(10)]
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)
>>> np.stack(arrays, axis=1).shape
(3, 10, 4)
>>> np.stack(arrays, axis=2).shape
(3, 4, 10)
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b))
array([[1, 2, 3],
       [2, 3, 4]])
>>> np.stack((a, b), axis=-1)
array([[1, 2],
       [2, 3],
       [3, 4]])
```
#### <big id=concatenate>[numpy.concatenate](#numpy)</big><small>(arrays,axis=0)</small>
* Join a sequence of arrays along an **existing axis**.
```py
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])

This function will not preserve masking of MaskedArray inputs.

>>> a = np.ma.arange(3)
>>> a[1] = np.ma.masked
>>> b = np.arange(2, 5)
>>> a
masked_array(data = [0 -- 2],
             mask = [False  True False],
       fill_value = 999999)
>>> b
array([2, 3, 4])
>>> np.concatenate([a, b])
masked_array(data = [0 1 2 2 3 4],
             mask = False,
       fill_value = 999999)
>>> np.ma.concatenate([a, b])
masked_array(data = [0 -- 2 2 3 4],
             mask = [False  True False False False False],
       fill_value = 999999)
```
#### <big id=vstack>[numpy.vstack](#numpy)</big><small>(tup)</small>
* Stack arrays in sequence vertically (row wise).
* Take a sequence of arrays and stack them vertically to make a single array. Rebuild arrays divided by vsplit.
* This function continues to be supported for backward compatibility, but you should prefer np.concatenate or np.stack. The np.stack function was added in NumPy 1.10.
```py
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])

>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[2], [3], [4]])
>>> np.vstack((a,b))
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
```
#### <big id=hstack>[numpy.hstack](#numpy)</big><small>(tup)</small>
* Stack arrays in sequence horizontally (column wise).
* Take a sequence of arrays and stack them horizontally to make a single array. Rebuild arrays divided by hsplit.
* This function continues to be supported for backward compatibility, but you should prefer np.concatenate or np.stack. The np.stack function was added in NumPy 1.10.
```py
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.hstack((a,b))
array([1, 2, 3, 2, 3, 4])
>>> a = np.array([[1],[2],[3]])
>>> b = np.array([[2],[3],[4]])
>>> np.hstack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
```
#### <big id=array_split>[numpy.array_split](#numpy)</big><small>(ary, indices_or_sections, axis=0)</small>
* Split an array into multiple sub-arrays.
* Please refer to the split documentation. The only difference between these functions is that array_split allows indices_or_sections to be an integer that does not equally divide the axis.
```py
>>> x = np.arange(8.0)
>>> np.array_split(x, 3)
    [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]
```
#### <big id=split>[numpy.split](#numpy)</big><small>(ary, indices_or_sections, axis=0)</small>
* Split an array into multiple sub-arrays.
```py
>>> x = np.arange(9.0)
>>> np.split(x, 3)
[array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]

>>> x = np.arange(8.0)
>>> np.split(x, [3, 5, 6, 10])
[array([ 0.,  1.,  2.]),
 array([ 3.,  4.]),
 array([ 5.]),
 array([ 6.,  7.]),
 array([], dtype=float64)]
```
#### <big id=vsplit>[numpy.vsplit](#numpy)</big><small>(ary, indices_or_sections)</small>
* Split an array into multiple sub-arrays vertically (row-wise).
* Please refer to the split documentation. vsplit is equivalent to split with axis=0 (default), the array is always split along the first axis regardless of the array dimension.
```py
>>> x = np.arange(16.0).reshape(4, 4)
>>> x
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.],
       [ 12.,  13.,  14.,  15.]])
>>> np.vsplit(x, 2)
[array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.]]),
 array([[  8.,   9.,  10.,  11.],
       [ 12.,  13.,  14.,  15.]])]
>>> np.vsplit(x, np.array([3, 6]))
[array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]]),
 array([[ 12.,  13.,  14.,  15.]]),
 array([], dtype=float64)]
 
With a higher dimensional array the split is still along the first axis.

>>> x = np.arange(8.0).reshape(2, 2, 2)
>>> x
array([[[ 0.,  1.],
        [ 2.,  3.]],
       [[ 4.,  5.],
        [ 6.,  7.]]])
>>> np.vsplit(x, 2)
[array([[[ 0.,  1.],
        [ 2.,  3.]]]),
 array([[[ 4.,  5.],
        [ 6.,  7.]]])]
```
#### <big id=hsplit>[numpy.hsplit](#numpy)</big><small>(ary, indices_or_sections)</small>
* Split an array into multiple sub-arrays horizontally (column-wise).
* Please refer to the split documentation. hsplit is equivalent to split with axis=1, the array is always split along the second axis regardless of the array dimension.
```py
>>> x = np.arange(16.0).reshape(4, 4)
>>> x
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.],
       [ 12.,  13.,  14.,  15.]])
>>> np.hsplit(x, 2)
[array([[  0.,   1.],
       [  4.,   5.],
       [  8.,   9.],
       [ 12.,  13.]]),
 array([[  2.,   3.],
       [  6.,   7.],
       [ 10.,  11.],
       [ 14.,  15.]])]
>>> np.hsplit(x, np.array([3, 6]))
[array([[  0.,   1.,   2.],
       [  4.,   5.,   6.],
       [  8.,   9.,  10.],
       [ 12.,  13.,  14.]]),
 array([[  3.],
       [  7.],
       [ 11.],
       [ 15.]]),
 array([], dtype=float64)]

With a higher dimensional array the split is still along the second axis.

>>> x = np.arange(8.0).reshape(2, 2, 2)
>>> x
array([[[ 0.,  1.],
        [ 2.,  3.]],
       [[ 4.,  5.],
        [ 6.,  7.]]])
>>> np.hsplit(x, 2)
[array([[[ 0.,  1.]],
       [[ 4.,  5.]]]),
 array([[[ 2.,  3.]],
       [[ 6.,  7.]]])]
```
#### <big id=newaxis>[numpy.newaxis](#numpy)</big>
* The newaxis object can be used in all slicing operations to create an axis of length one. newaxis is an alias for ‘None’, and ‘None’ can be used in place of this with the same result.
```py
>>> np.arange(4).shape
(4,)
>>> np.arange(4)[np.newaxis,:].shape
(1, 4)
>>> np.arange(4)[None,:].shape
(1, 4)
```
#### <big id=hstack>[numpy.hstack](#numpy)</big><small></small>
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
#### <b id=flat>[array.flat](#attributes)</b>
* A 1-D iterator over the array.
```py
>>> x = np.arange(1, 7).reshape(2, 3)
>>> x
array([[1, 2, 3],
       [4, 5, 6]])
>>> x.flat[3]
4
>>> x.T
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> x.T.flat[3]
5
>>> type(x.flat)
<type 'numpy.flatiter'>

An assignment example:

>>> x.flat = 3; x
array([[3, 3, 3],
       [3, 3, 3]])
>>> x.flat[[1,4]] = 1; x
array([[3, 1, 3],
       [3, 1, 3]])
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
#### <big id=dot>[array.dot](#methods)</big><small>(b, out=None)</small>
* Dot product of two arrays.
```py
>>> a = np.eye(2)
>>> b = np.ones((2, 2)) * 2
>>> a.dot(b)
array([[ 2.,  2.],
       [ 2.,  2.]])

This array method can be conveniently chained:

>>> a.dot(b).dot(b)
array([[ 8.,  8.],
       [ 8.,  8.]])
```
#### <big id=sort>[array.sort](#methods)</big><small>(axis=-1, kind='quicksort', order=None)</small>
* Sort an array, in-place.
```py
>>> a = np.array([[1,4], [3,1]])
>>> a.sort(axis=1)
>>> a
array([[1, 4],
       [1, 3]])
>>> a.sort(axis=0)
>>> a
array([[1, 3],
       [1, 4]])

Use the order keyword to specify a field to use when sorting a structured array:

>>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
>>> a.sort(order='y')
>>> a
array([('c', 1), ('a', 2)],
      dtype=[('x', '|S1'), ('y', '<i4')])
```
#### <big id=transpose>[array.transpose](#methods)</big><small>(*axes)</small>
* Returns a view of the array with axes transposed.
```py
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])
```
#### <big id=T>[array.T](#methods)</big>
* Same as self.transpose(), except that self is returned if self.ndim < 2.
```py
>>> x = np.array([[1.,2.],[3.,4.]])
>>> x
array([[ 1.,  2.],
       [ 3.,  4.]])
>>> x.T
array([[ 1.,  3.],
       [ 2.,  4.]])
>>> x = np.array([1.,2.,3.,4.])
>>> x
array([ 1.,  2.,  3.,  4.])
>>> x.T
array([ 1.,  2.,  3.,  4.])
```
#### <big id=flatten>[array.flatten](#methods)</big><small>(order='C')</small>
* Return a copy of the array collapsed into one dimension.
```py
>>> a = np.array([[1,2], [3,4]])
>>> a.flatten()
array([1, 2, 3, 4])
>>> a.flatten('F')
array([1, 3, 2, 4])
```
#### <big id=copy>[array.copy](#methods)</big><small>(order='C')</small>
* Return a copy of the array.
```py
>>> x = np.array([[1,2,3],[4,5,6]], order='F')
>>> y = x.copy()
>>> x.fill(0)
>>> x
array([[0, 0, 0],
       [0, 0, 0]])
>>> y
array([[1, 2, 3],
       [4, 5, 6]])
>>> y.flags['C_CONTIGUOUS']
True
```
### <b id=pandas>[Pandas](#title_pandas)</b>

