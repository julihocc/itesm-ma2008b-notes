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
# Import the NumPy random module
from numpy.random import default_rng
rng = default_rng(42)  # Initialize a PRNG with a fixed seed for reproducibility  
rand_nums = rng.random(10)  # Generate an array of 10 uniform random numbers in [0,1) range  
print(rand_nums)
```

#### **3.2 Generating Secure Random Numbers with****`secrets`**

```python
# Import the secrets module for generating secure random numbers
import secrets # Import the secrets module for generating secure random numbers
import string # Import the string module for character selection
def generate_password(length=12):  # Function to generate a secure password
    chars = string.ascii_letters + string.digits + string.punctuation  # Define possible password characters
    return ''.join(secrets.choice(chars) for _ in range(length))
print("Secure Password:", generate_password())  # Generate and display a secure password
```

### **4. Statistical Analysis & Testing Randomness**

To validate the quality of RNGs, statistical tests such as **Kolmogorov-Smirnov (KS) test**, **Chi-square test**, and **autocorrelation analysis** can be performed.

#### **4.1 Kolmogorov-Smirnov Test for Uniformity**

```python
# Import the Kolmogorov-Smirnov test from scipy.stats
from scipy.stats import kstest
samples = rng.random(1000)  # Generate 1000 uniform random samples
ks_stat, p_value = kstest(samples, 'uniform')  # Perform KS test for uniformity
print(f"KS Test Statistic: {ks_stat}, P-value: {p_value}")
```

#### **4.2 Visualization: Histogram and QQ-Plot**

```python
# Import required libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
samples = rng.normal(0, 1, 1000)  # Generate 1000 samples from a standard normal distribution
plt.figure(figsize=(12, 5))  # Create a figure with defined size
sns.histplot(samples, bins=30, kde=True)
stats.probplot(samples, dist="norm", plot=plt)
plt.show()  # Display the plot  # Display the plot  # Display the plot
```

### **5. Monte Carlo Simulations & Efficiency Enhancements**

Monte Carlo methods rely on randomness for approximating deterministic problems.

#### **5.1 Estimating π Using Monte Carlo**

```python
# Import necessary libraries for Monte Carlo simulation
from numba import njit, prange
import numpy as np
@njit(parallel=True)  # Optimize function with parallel processing using Numba
def monte_carlo_pi(n):  # Function to estimate Pi using the Monte Carlo method
    count = 0  # Initialize counter for points inside the circle
    for i in prange(n):  # Loop through n random points
        x, y = np.random.random(), np.random.random()  # Generate random (x, y) points in unit square
        if x**2 + y**2 <= 1:  # Check if point falls within unit circle
            count += 1
    return (count / n) * 4  # Calculate Pi approximation
print("Estimated Pi:", monte_carlo_pi(1000000))  # Estimate and print Pi value
```

#### **5.2 Monte Carlo Convergence Analysis**

```python
# Generate logarithmically spaced values for sample sizes
n_values = np.logspace(2, 6, num=20, dtype=int)
pi_estimates = [monte_carlo_pi(n) for n in n_values]  # Estimate Pi for different sample sizes
plt.plot(n_values, pi_estimates, marker='o', linestyle='dashed')  # Plot Pi estimates
plt.xscale("log")  # Use logarithmic scale for x-axis
plt.axhline(y=np.pi, color="red", linestyle="--", label="Actual Pi")  # Reference line for actual Pi value
plt.xlabel("Number of Samples")  # Label x-axis
plt.ylabel("Estimated Pi")  # Label y-axis
plt.legend()  # Show legend
plt.title("Monte Carlo Convergence")  # Title for the plot
plt.show()  # Display the plot
```

### **6. Real-World Applications of Random Number Generators**

#### **6.1 Financial Modeling: Stock Price Simulation Using Brownian Motion**

```python
# Define parameters for stock price simulation
T, N, S0, mu, sigma = 1, 1000, 100, 0.05, 0.2
dt = T/N  # Time step size
t = np.linspace(0, T, N)  # Create time vector
brownian_motion = np.cumsum(np.random.randn(N) * np.sqrt(dt))  # Generate Brownian motion
stock_prices = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * brownian_motion)  # Compute stock price path
plt.plot(t, stock_prices)  # Plot simulated stock prices
plt.xlabel("Time")  # Label x-axis
plt.ylabel("Stock Price")  # Label y-axis
plt.title("Stock Price Simulation (Geometric Brownian Motion)")  # Set plot title
plt.show()  # Display the plot
```

### **7. Conclusion**

Random number generators are vital in numerous domains, from simulations and cryptography to finance and AI. Understanding the differences between PRNGs and HRNGs ensures appropriate application selection. Moreover, leveraging statistical tests and efficiency techniques like Monte Carlo convergence significantly enhances computational accuracy and reliability.

### **8. Exercises**

#### **Basic Random Number Generation**

1. Generate a 10x10 array of uniform random numbers and compute its mean and standard deviation.
2. Generate 20 random integers between 1 and 10 and count their frequencies.
3. Generate random numbers with a fixed seed and verify reproducibility.

#### **Statistical Distributions**

1. Generate and plot samples from exponential and binomial distributions.
2. Perform KS and Chi-square tests to analyze the quality of generated random numbers.

#### **Advanced Topics**

1. Implement and analyze a comparison between PRNGs and HRNGs.
2. Simulate a 2D random walk and visualize the trajectory.
3. Implement a Linear Congruential Generator (LCG) and evaluate its output.
4. Use Simulated Annealing for optimization and analyze the impact of different RNGs.
