## 0. Import library
```python
import numpy as np
```
## 2. Multi dimention array
In numpy, we use 2d numpy array to show a matrix. Example about matrix in numpy:
```python
array([[1, 2],
       [3, 4]])
```
### 2.1. Initialize a matrix
Default, type of matrix is defined by type of matrix
To type cast, we use ```dtype```.
```python
A = np.array([1, 2, 3], [4, 5, 6], dtpye=np.float64)
print type(A[0][0]) # <type 'numpy.float64'>
```
### 2.2. Unit matrix and diagonal matrix
#### 2.2.1. Unit matrix
To create an unit ```n``` dimention matrix, we use function ```numpy.eyes()```
```python
I = np.eye(3)
print I # array([[1., 0., 0.],
        #        [0., 1., 0.],
        #        [0., 0., 1.]])
```
Function ```numpy.eye()``` also create any extra diagonal matrices:
```python
print np.eye(3, k=1) # array([[0., 1., 0.],
                     #        [0., 0., 1.],
                     #        [1., 0., 0.]])

print np.eye(3, k=-1) # array([[0., 0., 1.],
                      #        [1., 0., 0.],
                      #        [0., 1., 0.]])
```
#### 2.2.2. Diagonal matrix
To create a diagonal matrix or extract the diagonal of a matrix, we use function ```numpy.diag```
```python
print np.diag([1, 2, 3]) # array([[1., 0., 0.],
                         #        [0., 2., 0.],
                         #        [0., 0., 3.]])

print np.diag(np.diag([1, 2, 3])) # array([1, 2, 3])
```
* If input is 1d array, function returns 2d array that show matrix has diagonal is members of 1d array.
* If input is 2d array, function returns 1d array has value in column ```i``` and row ```i```.<br>
Extra diagonal of a matrix can get by this function.
```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.diag(A, k=1) # array([2, 6])
```
### 2.3. Shape of matrix
```python
A = np.array([[1, 2], [2, 3], [3, 4]])
print A.shape # (3, 2)
```
The answer is ```tuple```.
### 2.4. Access to matrix
#### 2.4.1. Access to each member
* As same as list
* As same as Matlab
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
print A[1][2] # 6
print A[1, 2] # 6
```
#### 2.4.2. Access to row, column
To access to row of matrix, we use ```A[i]``` or ```A[i,:]``` or ```A[i][:]```. To access to column, we use ```A[:,j]```
```python
A = np.array([[[1, 2, 3], [4, 5, 6]]])
print A[0, :] # array([1, 2, 3])
print A[:, 1] # array([2, 5])
```
### 2.5. Access to members of matrix
#### 2.5.1 A row or a column
```python
A = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
print A[0, 2:] # array([3, 4])
print A[:3, 1] # array([5, 6, 7])
```
#### 2.5.2. Rows or columns
```python
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print A[[1, 2]][[:, [2, 3]]] # array([[7, 8],
                             #        [11, 12]])
```
#### 2.5.3. Pair of coordinates
```python
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print A[[1, 2],[0, 1]] # array([5, 10])
```
Command above returns 1d array that includes ```A[1, 0]``` and ```A[2, 1]``.
### 2.6. Sum, min, max and average function
Convention ```axis=0``` is mean columns and ```axis=1``` is mean rows.
```python
A = np.array([[1.0, 2, 3], [4, 5, 6]])
print np.sum(A, axis=0) # array([5., 7., 9.])
print np.min(A, axis=0) # array([1., 2., 3.])
print np.mean(A, axis=0) # array([2.5, 3.5, 4.5])
```
Similar with ```axis=1```. <br>
If we don't use ```axis```, answer will compute with matrix
```python
A = np.array([[1.0, 2, 3], [4, 5, 6]])
print np.sum(A) # 21.
print np.max(A) # 6.
print np.mean(A) # 3.5
```
```keepdims = True```
Sometimes, we want to have the answer when ```axis=0``` is row, and when ```axis=1``` is column. So, Numpy provide ```keepdims=True``` (default is ```False```)

```python
A = np.array([[1.0, 2, 3], [4, 5, 6]])
print np.sum(A, axis=0, keepdims=True) # array([[5., 7., 9.]])
print np.sum(A, axis=1, keepdims=True) # array([[ 6.],
                                       #        [15.]])
```
### 2.7. The operator affect all members of matrix
#### 2.7.1. Compute a matrix with a scalar
```python
A = np.array([[1, 4], [6, 5]])
print A + 2 # array([[3, 7],
            #        [9, 8]])
print A * 2 # array([[2, 8],
            #        [12, 10]])
```
#### 2.7.2. numpy.abs, numpy.sin, numpy.exp, ...
```python
A = np.array([[1, -2], [-5, 3]])
print np.abs(A) # array([[1, 2],
                #        [5, 3]])
```
### 2.8. The operators with two matrices
The operators with two same size matrices (```+, -, *, /, **```) be done with pairs of elements. The answer is a matrix that same size with two computed matrix.
```python
A = np.array([[1, 2], [3, 0]])
B = np.array([[4, 1], [5, 2]])
print A*B # array([[4, 2],
          #        [15, 0]])
```
### 2.9. Transpose matrix, reshape matrix
#### 2.9.1. Transpose matrix
There are two ways to get transpose matrix: use ```.T  ``` properties or use ```numpy.transpose``` function:
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
print A.T  # array([[1, 4],
           #        [2, 5],
           #        [3, 6]])
print np.transpose(A) # array([[1, 4],
                      #        [2, 5],
                      #        [3, 6]])
```
#### 2.9.2. Reshape
When work with matrix, we usually transform matrix's size (reshape). Reshape matrix is mean arrange elements of matrix into another matrix that has same elements. In Numpy, we use ```numpy.reshape``` method, or ```numpy.reshape``` function.
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
print np.reshape(A, (3, 2)) # array([[1, 2],
                            #        [3, 4],
                            #        [5, 6]])
print A.reshape(3, 2) # array([[1, 2],
                      #        [3, 4],
                      #        [5, 6]])
print A.reshape(6) # array([1, 2, 3, 4, 5, 6])
print A.reshape(3, 1, 2) # array([[[1 2]]
                         #
                         #        [[3 4]]
                         #
                         #        [[5 6]]])
```
More about [numpy.reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html#numpy.reshape)
### 2.10. The operators between matrix and vector
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])
print A + b # array([[2, 4, 6],
            #        [5, 7, 9]])
```
### 2.11. Multiply between two matrices, multiply between matrix and vector
Using ```np.dot``` or ```.dot```. The answer is follow by fomular in Linear Algebra.
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[0, 1], [1, 2], [1, 0]])
b = np.array([1, 2, 3])
print np.dot(A, b)
```