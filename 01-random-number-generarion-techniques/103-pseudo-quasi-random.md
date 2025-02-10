# **Numerical Analysis for Non-Linear Optimization | Module 1**

## **Pseudo and Quasi-Random Numbers**

### **1. Introduction**

Random number generation is a critical component of numerical methods, especially in **Monte Carlo simulations**, **optimization algorithms**, and **stochastic modeling**. This module explores **pseudo-random** and **quasi-random** numbers, their properties, generation methods, and applications in numerical analysis.

---

### **2. Pseudo-Random Numbers (PRNs)**

Pseudo-random numbers are generated deterministically using algorithms that mimic randomness.

#### **2.1 Characteristics of PRNs**
- **Deterministic**: The sequence is fully determined by an initial seed.
- **Reproducible**: The same seed generates the same sequence.
- **Statistically random**: The numbers appear random but exhibit periodicity.
- **Uniform distribution**: Common generators produce values in \([0,1)\).

#### **2.2 Common PRNG Algorithms**
- **Linear Congruential Generator (LCG)**:
  \[
  X_{n+1} = (a X_n + c) \mod m
  \]
- **Mersenne Twister**: Default PRNG in NumPy, long period (~\(2^{19937}-1\)).
- **Xoshiro / SplitMix**: Modern alternatives to Mersenne Twister.
- **PCG (Permuted Congruential Generator)**: High statistical quality and efficiency.

#### **2.3 Implementation of a PRNG in Python**
```python
import numpy as np

# Initialize PRNG with seed for reproducibility
rng = np.random.default_rng(42)
random_numbers = rng.random(10)  # Generate 10 uniform random numbers
print(random_numbers)
```

---

### **3. Quasi-Random Numbers (QRNs)**

Quasi-random numbers are **low-discrepancy sequences** that cover the sample space more uniformly than PRNs.

#### **3.1 Characteristics of QRNs**
- **Low discrepancy**: Less clustering, ensuring better space coverage.
- **Deterministic**: Sequence is fixed but lacks periodicity.
- **Improves numerical integration**: More efficient in high-dimensional sampling.

#### **3.2 Common QRN Sequences**
- **Sobol sequence**: Best for high-dimensional integration.
- **Halton sequence**: Suitable for lower dimensions.
- **Faure sequence**: Similar to Halton but better uniformity properties.

#### **3.3 Implementation of a QRN Generator**
```python
from scipy.stats.qmc import Sobol

# Generate a Sobol sequence of 10 points in 2D
sobol = Sobol(d=2, scramble=False)
quasi_random_points = sobol.random(n=10)
print(quasi_random_points)
```

---

### **4. Applications of PRNs and QRNs**

#### **4.1 Monte Carlo Integration with PRNs**
```python
def monte_carlo_integral_prn(f, a, b, n):
    rng = np.random.default_rng(42)
    x = rng.uniform(a, b, n)  # Generate pseudo-random numbers
    return (b - a) * np.mean(f(x))

def f(x):
    return np.exp(-x**2)

print("Monte Carlo Integral (PRN):", monte_carlo_integral_prn(f, 0, 1, 10000))
```

#### **4.2 Monte Carlo Integration with QRNs**
```python
from scipy.stats.qmc import Halton

def monte_carlo_integral_qrn(f, a, b, n):
    halton = Halton(d=1, scramble=False)
    x = halton.random(n) * (b - a) + a  # Scale to [a, b]
    return (b - a) * np.mean(f(x))

print("Monte Carlo Integral (QRN):", monte_carlo_integral_qrn(f, 0, 1, 10000))
```

#### **4.3 Quasi-Random Sampling for High-Dimensional Optimization**
```python
from scipy.stats.qmc import Sobol

def sobol_sampling(dim, num_samples):
    sampler = Sobol(d=dim, scramble=False)
    return sampler.random(n=num_samples)

samples = sobol_sampling(3, 10)  # 10 samples in 3D space
print(samples)
```

---

### **5. Comparison of PRNs and QRNs**

| Feature               | Pseudo-Random Numbers | Quasi-Random Numbers |
|----------------------|---------------------|---------------------|
| **Generation**       | Algorithm-based     | Deterministic sequence |
| **Periodicity**      | Yes                 | No |
| **Uniformity**      | Moderate            | High |
| **Efficiency in Integration** | Lower | Higher |
| **Usage**            | General-purpose simulations | Numerical integration, optimization |

---

### **6. Conclusion**

Both **pseudo-random** and **quasi-random** numbers are essential tools in numerical computing. While **PRNs** are ideal for general simulations and cryptographic applications, **QRNs** provide superior efficiency in **integration and high-dimensional optimization**. Choosing the right type of randomness depends on the application context.

---

### **7. Exercises**

#### **Basic Implementations**
1. Generate 1000 pseudo-random numbers using **LCG** and plot their histogram.
2. Generate and plot **Halton and Sobol sequences** in 2D.

#### **Advanced Applications**
1. Implement a Monte Carlo integration for \( \int_0^1 \sin(x)dx \) using both **PRNs and QRNs** and compare their convergence.
2. Use **QRNs for optimizing a function** in 5 dimensions and compare performance with **random search**.

