## 0. Import library numpy
In Python, we initialize:
```python
import numpy as np
```
## 1. Initialize 1d array
### 1.1. Initialize vector
In numpy, vector is as same as 1d array.
```python
x = np.array([0, 1, 2, 3])
print x # array([0, 1, 2, 3])
```
### 1.2. Data type of array
```python
a1 = np.array([0, 1])
print type(a1[0]) # <type 'numpy.int64'>
a2 = np.array([1.0, 2.0])
print type([a2[0]]) # <type 'numpy.float64'>
a3 = np.array([0, 1], dtype=np.float64)
print type(a3[0]) # <type 'numpy.float64'>
```
Default, all members of interger array will have ```numpy.int64``` type, and real array will have ```numpy.float64``` type, but, we can type cast by using ```dtype```
### 1.3. Initialize special 1d array
#### 1.3.1. Array with 0 or 1 value
Vector zeros is a special vector which usually initialize. To initialize, we use function ```numpy.zeros```
```python
np.zeros(5) # array([0., 0., 0., 0., 0.])
```
Similar, with array has all 1 value, we use finction ```numpy.ones```
```python
np.ones(3) # array([1., 1., 1.])
```
Another, numpy has two special function ```numpy.zeros_like``` and ```numpy.ones_like``` to create zero value or one value array which has same dimension with variable.
```python
x = np.array([1, 2, 3])
print np.zeros_like(x) # array([0, 0, 0])
print np.ones_like(x) # array([1, 1, 1])
```
#### 1.3.2. Arithmetic progression
To create arithmetic progression in array, we use function ```numpy.arange```
```python
print np.arange(5) # array([0, 1, 2, 3, 4])
print np.arange(3, 6) # array([3, 4, 5])
print np.arange(1, 6, 2) # array([1, 3, 5])
```
### 1.4. Access to 1d array
#### 1.4.1. Shape of array
Shape of numpy array can be determined by ```numpy.array.shape```
```python
x = np.array([1, 2, 3, 4])
print x.shape # (4,)
```
The answer return **tuple** type. To get number in 1d array, we use:
```python
d = x.shape[0]
print d # 4
```
#### 1.4.2. Index
Each member in 1d array correlates with one index and is like index in ```python```, that start is 0.
#### 1.4.3. Access an element from vector
```python
x = np.array([1, 2, 3])
print x[0] # 1
```
#### 1.4.4. Reverse index
We have 1d array has ```n``` elements. To read the last element and don't know value of ```n```, we can use ```-1```
```python
x = np.array([1, 2, 3, 4])
print x[-1] # 4
```
***Note:*** If index ```i``` is out of array's size, when use ```x[i]```,will return error ```index ... is out of bounds for axis 0 with size ...```
#### 1.4.5. Change value an element of array
```python
x = np.array([1, 2, 3, 4])
x[0] = 2
print x # array([2, 2, 3, 4])
```
### 1.5. Access elements from 1d array
#### 1.5.1. Read
```python
x = np.array([1, 2, 3, 4, 5, 6, 7])
ids = [1, 2, 5]
print x[ids] # array([2, 3, 6])
print x[:5] # array([1, 2, 3, 4, 5])
print x[4:] # array([5, 6, 7])
print x[1:5] # array([2, 3, 4, 5])
```
#### 1.5.2. Write
```python
x = np.array([1, 2, 3, 4, 5, 6])

x[[1, 2, 3]] = 0
print x # array([1, 0, 0, 0, 5, 6])

x[4:] = np.array([9, 2])
print x # array([1, 0, 0, 0, 9, 2])

x[::-1] # reverse an array
print x # array([2, 9, 0, 0, 0, 1])
```
### 1.6. Compute with 1d array 
#### 1.6.1. Compute 1d array with scalar number
To compute 1d array with scalar number, we use operator addtion ```+```, subtraction ```-```, multiplication ```*```, division ```/``` and exponential ```**```
```python
x = np.array([1, 2, 3, 4, 5])
print x + 3 # array([4, 5, 6, 7, 8])
print x * 2 # array([2, 4, 6, 8, 10])
print x ** 2 # array([1, 4, 9, 16, 25])
```
In Python, we understand that the division a number with a array as same as the division a number with each members in array.
#### 1.6.2. Compute with two array
To compute two array, they must be same size.The answer is array which has same size with them. Operators ```+, -, *, /, **``` will compute in element-wise.
```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 1, 0, -1])
print x + y # array([3, 3, 3, 3])
print x * y # array([2, 2, 0, -4])
```
#### 1.6.3. Math functions
Some math functions in nupu such as: ```np.abs, np.log, np.exp, np.sin, np.cos, np.tan```.<br>
Function ```np.sum(x)``` will return sum of all member value in x array.
### 1.8. Inner product of two vector
```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 4, 5])
print x.dot(y) # 39
print np.dot(x, y) # 39
```
### 1.9. min, max, argmin, argmax of array
To find min and max value in 1d array, we use function ```np.min``` or ```np.max```. To find index where get min or max value in array, we use function ```np.argmin``` or ```np.argmax```
```python
x = np.array([1, 3, 5, 2, 0, 4])
print np.min(x) # 0
print np.max(x) # 5

print x.min() # 0
print x.max() # 5

print np.argmin(x) # 4
print np.argmax(x) # 2
```