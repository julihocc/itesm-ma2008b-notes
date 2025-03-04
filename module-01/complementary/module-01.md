# Module 1 | Preliminaries

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Pseudo-Random Number Generators](#2-pseudo-random-number-generators)  
   2.1 [Key Concepts in PRNGs](#21-key-concepts-in-prngs)  
   2.2 [Linear Congruential Generator (LCG)](#22-linear-congruential-generator-lcg)  
       - 2.2.1 [Theory](#221-theory)  
       - 2.2.2 [Python Example](#222-python-example)  
   2.3 [Mersenne Twister (MT19937)](#23-mersenne-twister-mt19937)  
       - 2.3.1 [Overview](#231-overview)  
       - 2.3.2 [Python Examples](#232-python-examples)  
   2.4 [Sobol Sequences (Quasi-Random)](#24-sobol-sequences-quasi-random)  
       - 2.4.1 [What Are Sobol Sequences?](#241-what-are-sobol-sequences)  
       - 2.4.2 [Python Example (SciPy)](#242-python-example-scipy)  
   2.5 [Kolmogorov-Smirnov (KS) Test](#25-kolmogorov-smirnov-ks-test)  
       - 2.5.1 [Purpose](#251-purpose)  
       - 2.5.2 [Python Example](#252-python-example)  
   2.6 [Summary and Comparisons](#26-summary-and-comparisons)  
3. [Taylor and Maclaurin Series](#3-taylor-and-maclaurin-series)  
   3.1 [Taylor Series](#31-taylor-series)  
       - 3.1.1 [Example: Approximating $e^x$](#311-example-approximating-ex)  
   3.2 [Maclaurin Series](#32-maclaurin-series)  
       - 3.2.1 [Example: Approximating $\sin(x)$](#321-example-approximating-sinx)  
   3.3 [Applications](#33-applications)  
4. [Concluding Remarks](#4-concluding-remarks)

---

## 1. Introduction

Monte Carlo methods, random number generation, and **Taylor/Maclaurin expansions** are foundational in **non-linear optimization** and **numerical analysis**. This document covers:

- An overview of **pseudo-random** and **quasi-random** number generators, including the **Kolmogorov–Smirnov (KS)** test.  
- A review of **Taylor** and **Maclaurin** polynomials for function approximation, error analysis, and iterative methods.

---

## 2. Pseudo-Random Number Generators

### 2.1 Key Concepts in PRNGs

- **Period**: A PRNG eventually repeats its sequence; a longer period reduces unwanted repetition.  
- **Statistical Quality**: Good PRNGs pass tests like Kolmogorov–Smirnov to confirm uniformity.  
- **Cryptographic Security**: Some PRNGs (like cryptographic PRNGs) are designed for unpredictability—LCG or Mersenne Twister are *not* secure.  

Quasi-random (low-discrepancy) sequences (e.g., **Sobol**) provide more uniform coverage in multiple dimensions, often speeding up Monte Carlo convergence.

---

### 2.2 Linear Congruential Generator (LCG)

#### 2.2.1 Theory

A **Linear Congruential Generator** uses:

$$
X_{n+1} = (a X_n + c) \bmod m
$$

- $a$: multiplier  
- $c$: increment  
- $m$: modulus  
- $X_0$: seed  

We map $\{X_n\}$ into $[0,1)$ via $\frac{X_n}{m}$.

**Strengths**  
- Simple & fast

**Limitations**  
- Max period $\leq m$  
- Not secure  
- Tends to produce points on hyperplanes in higher dimensions

#### 2.2.2 Python Example

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearCongruentialGenerator:
    def __init__(self, seed=1, a=1664525, c=1013904223, m=2**32):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self):
        """Generate the next random number."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m  # Normalize to [0,1)

    def generate(self, n):
        """Generate n random numbers."""
        return [self.next() for _ in range(n)]

# Usage
lcg = LinearCongruentialGenerator(seed=42)
random_numbers = lcg.generate(1000)

plt.hist(random_numbers, bins=20, density=True, alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of LCG-Generated Numbers")
plt.show()
```

---

### 2.3 Mersenne Twister (MT19937)

#### 2.3.1 Overview

- Period: $2^{19937} - 1$  
- Excellent statistical quality  
- Very fast (bitwise operations)  
- **Not** cryptographically secure

#### 2.3.2 Python Examples

1. **Using NumPy**:

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)
random_numbers_mt = rng.random(100000)

plt.hist(random_numbers_mt, bins=50, density=True, alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Mersenne Twister (NumPy)")
plt.show()
```

2. **Using Python’s built-in `random`**:

```python
import random
import matplotlib.pyplot as plt

random.seed(42)
random_numbers_builtin = [random.random() for _ in range(100000)]

plt.hist(random_numbers_builtin, bins=50, density=True, alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Mersenne Twister (Python built-in)")
plt.show()
```

---

### 2.4 Sobol Sequences (Quasi-Random)

#### 2.4.1 What Are Sobol Sequences?

A **Sobol** sequence is a *low-discrepancy* (quasi-random) sequence offering more uniform coverage of \([0,1)^d\). This improves efficiency for high-dimensional tasks like integration or optimization.

#### 2.4.2 Python Example (SciPy)

```python
from scipy.stats.qmc import Sobol
import numpy as np
import matplotlib.pyplot as plt

sobol = Sobol(d=2, scramble=False)
quasi_random_points = sobol.random(n=100)

plt.scatter(quasi_random_points[:, 0], quasi_random_points[:, 1], alpha=0.7)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sobol Sequence in 2D")
plt.show()
```

---

### 2.5 Kolmogorov-Smirnov (KS) Test

#### 2.5.1 Purpose

- **One-sample KS**: Sample vs. theoretical distribution (e.g., uniform).  
- **Two-sample KS**: Sample vs. sample.

#### 2.5.2 Python Example

```python
import numpy as np
from scipy.stats import kstest, ks_2samp

# One-sample KS
sample = np.random.normal(loc=0, scale=1, size=100)
ks_stat, p_value = kstest(sample, 'norm')

# Two-sample KS
sample1 = np.random.normal(0, 1, 100)
sample2 = np.random.uniform(-1, 1, 100)
ks_stat_2, p_value_2 = ks_2samp(sample1, sample2)
```

---

### 2.6 Summary and Comparisons

| **Generator/Sequence** | **Period**               | **Speed**        | **Statistical Quality**         | **Security**               | **Usage**                                   |
|------------------------|--------------------------|------------------|---------------------------------|----------------------------|---------------------------------------------|
| **LCG**                | $\le m$               | ✅ Fast          | Fair (depends on parameters)    | ❌ Not secure             | Simple demos, teaching examples             |
| **Mersenne Twister**   | $2^{19937}-1$         | ✅✅ Very Fast   | Excellent for simulations       | ❌ Not secure             | Default in NumPy, general random needs      |
| **Sobol** (Quasi-Rand) | No standard “period”    | ✅✅ Efficient   | Very uniform in multi-dims      | N/A                       | Integration, optimization, variance reduction|
| **Crypto PRNG**        | Extremely large/unfeas. | ❌ Slower        | Strong unpredictability         | ✅ Secure                 | Security, cryptography                      |

---

## 3. Taylor and Maclaurin Series

**Taylor** and **Maclaurin** expansions approximate functions, analyze errors, and support methods like **Newton’s method**.

### 3.1 Taylor Series

$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \dots
$$

#### 3.1.1 Example: Approximating $e^x$

```python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def taylor_expansion_e_x(x, n_terms):
    return sum([(x**n) / factorial(n) for n in range(n_terms)])

x_vals = np.linspace(-2, 2, 100)
y_true = np.exp(x_vals)
y_approx = [taylor_expansion_e_x(x, 5) for x in x_vals]

plt.plot(x_vals, y_true, label="Actual e^x")
plt.plot(x_vals, y_approx, '--', label="Taylor Approx (n=5)")
plt.legend()
plt.xlabel("x")
plt.ylabel("e^x")
plt.title("Taylor Series Approx. of e^x")
plt.show()
```

---

### 3.2 Maclaurin Series

A **Maclaurin series** is a Taylor series centered at 0.  
For example, $\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$.

#### 3.2.1 Example: Approximating $\sin(x)$

```python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def maclaurin_sin(x, n_terms):
    return sum([((-1)**n * x**(2*n+1)) / factorial(2*n+1) for n in range(n_terms)])

x_vals = np.linspace(-np.pi, np.pi, 100)
y_true = np.sin(x_vals)
y_approx = [maclaurin_sin(x, 5) for x in x_vals]

plt.plot(x_vals, y_true, label="Actual sin(x)")
plt.plot(x_vals, y_approx, '--', label="Maclaurin Approx (n=5)")
plt.legend()
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Maclaurin Series Approx. of sin(x)")
plt.show()
```

---

### 3.3 Applications

1. **Error Analysis**: Higher-order terms show truncation error.  
2. **Inverse Transform Sampling**: Approximate inverse CDFs via Taylor expansions.  
3. **Newton’s Method** (root-finding, optimization):

```python
def newton_method(f, f_prime, x0, tol=1e-5, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x)/f_prime(x)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x

# Example: Solve x^2 - 2 = 0
root_approx = newton_method(lambda x: x**2 - 2, lambda x: 2*x, 1)
print("Approx root:", root_approx)
```

---

## 4. Concluding Remarks

**Pseudo-random number generators** (LCG, Mersenne Twister, Sobol) and **Taylor/Maclaurin expansions** are fundamental in **numerical analysis** and **non-linear optimization**:

- PRNGs drive Monte Carlo simulations, sampling, and randomized algorithms.  
- Taylor/Maclaurin expansions underpin function approximations, error analysis, and iterative solvers (e.g., Newton’s method).

Familiarity with both areas is essential for advanced topics in **scientific computing**, **statistical analysis**, and **optimization**.