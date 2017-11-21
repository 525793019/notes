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
### <b id=pandas>[Pandas](#title_pandas)</b>

