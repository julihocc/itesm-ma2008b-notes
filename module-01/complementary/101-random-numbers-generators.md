# Numerical Analysis for Non-Linear Optimization | Module 1

## **Random Number Generators: Theory, Implementation, and Applications**

### **1. Introduction**

Random number generators (RNGs) are fundamental in **scientific computing**, used in **Monte Carlo simulations**, **cryptography**, **statistical analysis**, and **machine learning**. This module explores the theory behind RNGs, their implementation in Python, statistical validation techniques, and real-world applications.

### **2. Understanding Randomness: Algorithmic vs. Physical**

#### **2.1 Pseudo-Random Number Generators (PRNGs)**

PRNGs are deterministic algorithms that generate sequences of numbers appearing random but are reproducible given the same seed.

##### **Key Characteristics:**

- **Deterministic**: Same seed produces the same sequence.
- **Statistical randomness**: Should pass randomness tests.
- **Periodicity**: PRNGs eventually repeat sequences.
- **Uniform distribution**: Most PRNGs generate numbers in [0,1).

##### **Common PRNG Algorithms:**

- **Mersenne Twister** (default in NumPy, but not ideal for cryptography).
- **PCG (Permuted Congruential Generator)**.
- **Xoshiro and SplitMix** (better alternatives to Mersenne Twister).

#### **2.2 Hardware Random Number Generators (HRNGs)**

HRNGs generate numbers using physical phenomena such as **thermal noise** or **quantum effects**.

##### **Entropy Sources for HRNGs:**

- **Electrical noise** in circuits.
- **Photonic processes** (quantum randomness).
- **Radioactive decay** (unpredictable at quantum level).

HRNGs are essential in **cryptographic security** where true randomness is required.

### **3. Implementing Random Number Generators in Python**

#### **3.1 Using NumPy's Modern PRNG API**

```python
from numpy.random import default_rng
rng = default_rng(42)  # PRNG with new API
rand_nums = rng.random(10)  # Generate 10 uniform numbers
```

#### **3.2 Generating Secure Random Numbers with `secrets`**

```python
import secrets
import string
def generate_password(length=12):
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(chars) for _ in range(length))
print("Secure Password:", generate_password())
```

### **4. Statistical Analysis & Testing Randomness**

To validate the quality of RNGs, statistical tests such as **Kolmogorov-Smirnov (KS) test**, **Chi-square test**, and **autocorrelation analysis** can be performed.

#### **4.1 Kolmogorov-Smirnov Test for Uniformity**

```python
from scipy.stats import kstest
samples = rng.random(1000)
ks_stat, p_value = kstest(samples, 'uniform')
print(f"KS Test Statistic: {ks_stat}, P-value: {p_value}")
```

#### **4.2 Visualization: Histogram and QQ-Plot**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
samples = rng.normal(0, 1, 1000)
plt.figure(figsize=(12, 5))
sns.histplot(samples, bins=30, kde=True)
stats.probplot(samples, dist="norm", plot=plt)
plt.show()
```

### **5. Monte Carlo Simulations & Efficiency Enhancements**

Monte Carlo methods rely on randomness for approximating deterministic problems.

#### **5.1 Estimating π Using Monte Carlo**

```python
from numba import njit, prange
import numpy as np
@njit(parallel=True)
def monte_carlo_pi(n):
    count = 0
    for i in prange(n):
        x, y = np.random.random(), np.random.random()
        if x**2 + y**2 <= 1:
            count += 1
    return (count / n) * 4
print("Estimated Pi:", monte_carlo_pi(1000000))
```

#### **5.2 Monte Carlo Convergence Analysis**

```python
n_values = np.logspace(2, 6, num=20, dtype=int)
pi_estimates = [monte_carlo_pi(n) for n in n_values]
plt.plot(n_values, pi_estimates, marker='o', linestyle='dashed')
plt.xscale("log")
plt.axhline(y=np.pi, color="red", linestyle="--", label="Actual Pi")
plt.xlabel("Number of Samples")
plt.ylabel("Estimated Pi")
plt.legend()
plt.title("Monte Carlo Convergence")
plt.show()
```

### **6. Real-World Applications of Random Number Generators**

#### **6.1 Financial Modeling: Stock Price Simulation Using Brownian Motion**

```python
T, N, S0, mu, sigma = 1, 1000, 100, 0.05, 0.2
dt = T/N
t = np.linspace(0, T, N)
brownian_motion = np.cumsum(np.random.randn(N) * np.sqrt(dt))
stock_prices = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * brownian_motion)
plt.plot(t, stock_prices)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price Simulation (Geometric Brownian Motion)")
plt.show()
```

### **7. Conclusion**

Random number generators are vital in numerous domains, from simulations and cryptography to finance and AI. Understanding the differences between PRNGs and HRNGs ensures appropriate application selection. Moreover, leveraging statistical tests and efficiency techniques like Monte Carlo convergence significantly enhances computational accuracy and reliability.

### **8. Exercises**

#### **Basic Random Number Generation**

1. Generate a 10x10 array of uniform random numbers and compute its mean and standard deviation.
2. Generate 20 random integers between 1 and 10 and count their frequencies.
3. Generate 100 Gaussian-distributed samples, plot a histogram, and compute statistical properties.
4. Generate random numbers with a fixed seed and verify reproducibility.

#### **Statistical Distributions**

1. Generate and plot samples from exponential, binomial, and chi-squared distributions.
2. Perform KS and Chi-square tests to analyze the quality of generated random numbers.

#### **Advanced Topics**

1. Estimate Pi using Monte Carlo methods.
2. Implement and analyze a comparison between PRNGs and HRNGs.
3. Simulate a 2D random walk and visualize the trajectory.
4. Implement a Linear Congruential Generator (LCG) and evaluate its output.
5. Use Simulated Annealing for optimization and analyze the impact of different RNGs.
