# Lecture Notes on Using SageMath to Find Laplace Transformations

## Introduction to Laplace Transforms
The Laplace transform is a powerful integral transform used to convert differential equations into algebraic equations. Given a function $f(t)$, its Laplace transform is defined as:
$$
L[f(t)] = F(s) = \int_{0}^{\infty} e^{-st} f(t) dt
$$
where $s$ is a complex number.

## Computing Laplace Transforms in SageMath
SageMath provides built-in functionality to compute the Laplace transform of a function. To use it, we first declare the variables and function.

### Example 1: Basic Laplace Transform
Let's compute the Laplace transform of $f(t) = e^{at}$:
```python
t,s,a = var('t s a')
f = exp(a*t)
L = laplace(f, t, s)
print(L)
```
#### Output:
$$
\frac{1}{s-a}, \quad \text{for } s > a
$$

### Example 2: Laplace Transform of Trigonometric Functions
Let's compute the Laplace transform of $f(t) = \sin(bt)$:
```python
t, s, b = var('t s b')
f = sin(b*t)
L = laplace(f, t, s)
print(L)
```
#### Output:
$$
\frac{b}{s^2 + b^2}
$$

### Example 3: Laplace Transform of a Piecewise Function
Consider the Heaviside step function $u(t - c)$, defined as:
$$
 u(t - c) = \begin{cases}
 0, & t < c \\
 1, & t \geq c
 \end{cases}
$$
We compute its Laplace transform using SageMath:
```python
var('t s c')
assume(c > 0)  # Assuming c is positive
f = heaviside(t-c)
L = laplace(f, t, s)
print(L)
```
#### Output:
$$
\frac{e^{-cs}}{s}
$$

## Inverse Laplace Transform in SageMath
SageMath also allows us to compute inverse Laplace transforms. For example:

### Example 4: Inverse Laplace Transform
Find the inverse Laplace transform of $F(s) = \frac{1}{s^2 + 1}$:
```python
s, t = var('s t')
F = 1/(s^2 + 1)
f_t = inverse_laplace(F, s, t)
print(f_t)
```
#### Output:
$$
\sin(t)
$$

## Solving Differential Equations Using Laplace Transforms
Laplace transforms can be used to solve linear differential equations with initial conditions.

### Example 5: Solving a Second-Order Differential Equation
Solve $y'' - 3y' - 4y = \sin(x)$ with initial conditions $y(0) = 1, y'(0) = -1$:
```python
def L(y,x):
    return diff(y, x, x) - 3*diff(y, x) - 4*y 
```

```python
var('x s')
y = function('y')(x)
deq = L(y, x) == sin(x)
L_deq = laplace(deq, x, s)
print(L_deq)
```
This results in an algebraic equation in $s$, which can be solved for $Y(s)$. The inverse Laplace transform then gives $y(x)$.


```python
# Solve L_deq for Y(s)

Y, a, b = var("Y, a, b")

substitutions = {
    laplace(y, x, s): Y,
    y(x=0): 1,
    diff(y, x).subs(x==0): -1
}

L_algebraic = L_deq.subs(substitutions)

print(L_algebraic)
```

```python
algebraic_solutions = solve(L_algebraic, Y, solution_dict=True)
print(algebraic_solutions)
```

```python
y_x = inverse_laplace(algebraic_solutions[0][Y], s, x)
print(y_x)
```

```python
L(y_x, x)
```

```python
y_x(x = 0)
```

```python
diff(y_x, x).subs(x=0)
```

```python
F = algebraic_solutions[0][Y]
F.partial_fraction_decomposition()
```

```python

```
