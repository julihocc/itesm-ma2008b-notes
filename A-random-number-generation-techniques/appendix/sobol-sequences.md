# **Sobol Sequences and Their Application in Quasi-Random Sampling**

## **1. Introduction**

Sobol sequences are a type of **quasi-random sequence** used in **low-discrepancy sampling**. Unlike purely random numbers, Sobol sequences **cover the space more uniformly**, making them useful in **Monte Carlo simulations**, **numerical integration**, and **optimization**.

Quasi-random sequences such as Sobol are particularly effective in **high-dimensional** problems where standard random sampling may lead to clustering or inefficient space coverage.

This lecture introduces the Sobol sequence, its mathematical formulation, implementation in Python, and applications.

---

## **2. Mathematical Definition of Sobol Sequences**

A **Sobol sequence** is a deterministic sequence of points that fills a space **more evenly than purely random sampling**. It is constructed using **direction numbers** and **bitwise operations**, ensuring that each new sample improves the coverage of the space.

For a **Sobol sequence in ****\(d\)****-dimensions**, the sequence of points \(\mathbf{x}_i = (x_{i,1}, x_{i,2}, \dots, x_{i,d})\) is generated recursively using a binary representation of the index \(i\).

Each component of \(\mathbf{x}_i\) is computed as:

$$
x_{i,j} = \sum_{k=1}^{b} v_k^{(j)} b_k
$$

where:

- \(b_k\) are the bits of the index \(i\) in binary representation,
- \(v_k^{(j)}\) are **direction numbers** that control the distribution,
- \(b\) is the bit-length of \(i\).

The sequence ensures **low discrepancy**, meaning that each additional sample improves the uniformity of the coverage.

---

## **3. Python Implementation**

### **3.1. Generating a Sobol Sequence**

In Python, the `scipy.stats.qmc` module provides an implementation of the **Sobol sequence**:

```python
from scipy.stats.qmc import Sobol

# Generate a Sobol sequence of 10 points in 2D
sobol = Sobol(d=2, scramble=False)
quasi_random_points = sobol.random(n=10)
print(quasi_random_points)
```

### **3.2. Explanation of the Code**

1. **Importing Sobol from SciPy**:\
   The **Quasi-Monte Carlo (QMC)** module in `scipy.stats.qmc` provides **low-discrepancy sequences**, including Sobol.

2. **Initializing the Sobol Generator**:

   ```python
   sobol = Sobol(d=2, scramble=False)
   ```

   - \(d=2\) specifies a **2-dimensional** Sobol sequence.
   - `scramble=False` uses the **original Sobol sequence** without additional randomness.

3. **Generating Points**:

   ```python
   quasi_random_points = sobol.random(n=10)
   ```

   - Generates **10 quasi-random points** in **2D**.

4. **Output Example**:

   ```
   [[0.     0.    ]
    [0.5    0.5   ]
    [0.25   0.75  ]
    [0.75   0.25  ]
    [0.125  0.625 ]
    [0.625  0.125 ]
    [0.375  0.875 ]
    [0.875  0.375 ]
    [0.0625 0.8125]
    [0.5625 0.3125]]
   ```

   - Each row represents a **quasi-random point** in the \([0,1)^2\) unit square.
   - The sequence **avoids clustering** and **progressively fills the space**.

---

## **4. Visualization of Sobol Sequence**

To better understand how Sobol points are distributed, we can visualize them:

```python
import matplotlib.pyplot as plt

# Generate 100 Sobol points in 2D
sobol = Sobol(d=2, scramble=False)
quasi_random_points = sobol.random(n=100)

# Scatter plot
plt.scatter(quasi_random_points[:, 0], quasi_random_points[:, 1], alpha=0.7)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sobol Sequence in 2D")
plt.show()
```

- This **scatter plot** demonstrates how Sobol points **cover the space more evenly** than purely random sampling.

---

## **5. Comparison: Sobol vs. Random Sampling**

To highlight the advantage of **low-discrepancy sequences**, we compare Sobol sampling with purely random sampling:

```python
import numpy as np

# Generate random points for comparison
random_points = np.random.rand(100, 2)

# Plot comparison
plt.scatter(random_points[:, 0], random_points[:, 1], alpha=0.7, color="red", label="Random")
plt.scatter(quasi_random_points[:, 0], quasi_random_points[:, 1], alpha=0.7, color="blue", label="Sobol")
plt.legend()
plt.title("Sobol Sequence vs. Random Sampling")
plt.show()
```

### **Observations**

- **Red points (Random Sampling)**: Shows **clustering** and **uneven gaps**.
- **Blue points (Sobol Sampling)**: More **evenly spaced** across the area.

---

## **6. Applications of Sobol Sequences**

Sobol sequences are widely used in applications that require **efficient space coverage** and **low variance**.

### **6.1. Monte Carlo Integration**

In Monte Carlo integration, Sobol sequences provide **faster convergence** than purely random sampling:

$$
I \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)
$$

where \(x_i\) are **Sobol-distributed points**.

### **6.2. Global Optimization**

Sobol sequences improve the **exploration of high-dimensional spaces** in **Bayesian optimization** and **machine learning hyperparameter tuning**.

### **6.3. Financial Simulations**

Used in **risk analysis, derivative pricing, and portfolio optimization**, where Sobol sampling reduces simulation variance.

### **6.4. Computer Graphics**

In **ray tracing**, Sobol sequences **minimize aliasing** and **increase rendering efficiency**.

---

## **7. Scrambled Sobol Sequences**

A **scrambled Sobol sequence** introduces controlled randomness while preserving low discrepancy. This improves performance in **stochastic simulations**.

```python
sobol_scrambled = Sobol(d=2, scramble=True)
scrambled_points = sobol_scrambled.random(n=10)
print(scrambled_points)
```

- Scrambling adds **randomness** but maintains **better uniformity** than purely random numbers.

---

## **8. Summary**

- **Sobol sequences** generate **quasi-random** numbers with **low discrepancy**.
- They **cover space more evenly** than purely random sampling.
- Useful in **Monte Carlo methods, finance, global optimization, and graphics**.
- **SciPy's ****`qmc.Sobol`**** provides an efficient implementation**.

By leveraging Sobol sequences, practitioners can enhance the efficiency and accuracy of various computational tasks requiring **low-discrepancy sampling**.