---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: SageMath 9.5
    language: sage
    name: sagemath
---

# Tutorial for Linear Algebra 

Linear algebra underpins a lot of Sage’s algorithms, so it is fast, robust and comprehensive. We’ve already seen some basic linear algebra, including matrices, determinants, and the `.rref()` method for row-reduced echelon form in the Programming Tutorial, so the content here continues from there to some extent.


## Matrices and Vectors
We can make a matrix easily by passing a list of the rows. Don’t forget to use tab-completion to see routines that are possible.

```sage vscode={"languageId": "sage"}
A = matrix([[1,2,3],[4,5,6]]); A
```

But there are lots of other ways to make matrices. Each of these shows what is assumed with different input; can you figure out how Sage interprets them before you read the documentation which the command `matrix?` provides?

It’s a good idea to get in the habit of telling Sage what ring to make the matrix over. Otherwise, Sage guesses based on the elements, so you may not have a matrix over a field! Here, we tell Sage to make the ring over the rationals.

```sage vscode={"languageId": "sage"}
B = matrix(QQ, 3, 2, [1,2,3,4,5,6]); B
```

```sage vscode={"languageId": "sage"}
C = matrix(CC, 3, [1,2,3,4,5,6]); C
```

```sage vscode={"languageId": "sage"}
D = matrix(CC, 20, range(400)); D
```

Don’t forget that when viewing this in the notebook, you can click to the left of the matrix in order to cycle between “wrapped”, “unwrapped” and “hidden” modes of output.

```sage vscode={"languageId": "sage"}
print(D.str())
```

```sage vscode={"languageId": "sage"}
E = diagonal_matrix( [0..40,step=4] ); E
```

```sage vscode={"languageId": "sage"}
column_matrix(QQ,[[1,2,3],[4,5,6],[7,8,9]])
```

You can also combine matrices in different ways.

```sage vscode={"languageId": "sage"}
F1=matrix(QQ,2,2,[0,1,1,0])
F2=matrix(QQ,2,2,[1,2,3,4])
F3=matrix(QQ,1,2,[3,1])
block_matrix(2,2,[F1,F2,0,F3])
```

```sage vscode={"languageId": "sage"}
F1.augment(F2)
```

```sage vscode={"languageId": "sage"}
F1.stack(F2)
```

```sage vscode={"languageId": "sage"}
block_diagonal_matrix([F1,F2])
```

Vectors are rows or columns, whatever you please, and Sage interprets them as appropriate in multiplication contexts.

```sage vscode={"languageId": "sage"}
row = vector( (3, -1, 4))
row
```

```sage vscode={"languageId": "sage"}

col = vector( QQ, [4, 5] )
col
```

```sage vscode={"languageId": "sage"}
F = matrix(QQ, 3, 2, range(6)); F
```

```sage vscode={"languageId": "sage"}
F*col
```

```sage vscode={"languageId": "sage"}
row*F
```

Although our “vectors” (especially over rings other than fields) might be considered as elements of an appropriate free module, they basically behave as vectors for our purposes.

```sage vscode={"languageId": "sage"}
ring_vec = vector(SR, [2, 12, -4, 9])
ring_vec
```

```sage vscode={"languageId": "sage"}
type( ring_vec )
```

```sage vscode={"languageId": "sage"}

field_vec = vector( QQ, (2, 3, 14) )
field_vec
```

```sage vscode={"languageId": "sage"}
type( field_vec )
```

## Left-Handed or Right-handed?

Sage “prefers” rows to columns. For example, the kernel method for a matrix $A$
 computes the left kernel – the vector space of all vectors $v$
 for which $v \cdot A = 0$
 – and prints out the vectors as the rows of a matrix.

```sage vscode={"languageId": "sage"}
G = matrix(QQ, 2, 3, [[1,2,3],[2,4,6]])
G.kernel()
```

```sage vscode={"languageId": "sage"}
G.left_kernel()
```

```sage vscode={"languageId": "sage"}
G.right_kernel()
```

## Vector Spaces
Since Sage knows the kernel is a vector space, you can compute things that make sense for a vector space.

```sage vscode={"languageId": "sage"}
V=G.right_kernel()
V
```

```sage vscode={"languageId": "sage"}
V.dimension()
```

```sage vscode={"languageId": "sage"}
V.coordinate_vector([1,4,-3])
```

Here we get the basis matrix (note that the basis vectors are the rows of the matrix):



```sage vscode={"languageId": "sage"}
V.basis_matrix()
```

```sage vscode={"languageId": "sage"}
V.basis_matrix()
```

Kernels are vector spaces and bases are “echelonized” (canonicalized).

This is why the ring for the matrix is important. Compare the kernels above with the kernel using a matrix which is only defined over the integers.

```sage vscode={"languageId": "sage"}
G = matrix(ZZ,2, 3, [[1,2,3],[2,4,6]])
G.kernel()
```

## Computations
Here are some more computations with matrices and vectors.

As you might expect, random matrices are random.

```sage vscode={"languageId": "sage"}
set_random_seed(42)  # Set the seed for reproducibility
H = random_matrix(QQ, 5, 5, num_bound = 10, den_bound = 4)
H 

```

```sage vscode={"languageId": "sage"}

H.det() 

```

```sage vscode={"languageId": "sage"}
H.eigenvalues() 
```

According to the [Numerical analysis quickstart](https://doc.sagemath.org/html/en/prep/Quickstarts/NumAnalysis.html), the question marks indicate that the actual number is inside the interval found by incrementing and decrementing the last digit of the printed number. So 9.1? is a number between 9.0 and 9.2. Sage knows exactly what number this is (since it’s a root of a polynomial), but uses interval notation to print an approximation for ease of use.




The eigenvectors_right command prints out a list of `(eigenvalue, [list of eigenvectors], algebraic multiplicity)` tuples for each eigenvalue.

```sage vscode={"languageId": "sage"}
H.eigenvectors_right() 
```

It may be more convenient to use the `eigenmatrix_right` command, which gives a diagonal matrix of eigenvalues and a column matrix of eigenvectors.

```sage vscode={"languageId": "sage"}
D,P=H.eigenmatrix_right()
P*D-H*P
```

## Matrix Solving
We can easily solve linear equations using the backslash, like in Matlab.

```sage vscode={"languageId": "sage"}
A = random_matrix(QQ, 3, algorithm='unimodular')
v = vector([2,3,1])
A,v  
x=A\v; x  
A*x  
```
