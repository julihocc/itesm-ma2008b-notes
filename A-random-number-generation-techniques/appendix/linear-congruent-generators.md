# **Linear Congruential Generator (LCG)**

## **1. Introduction**
The **Linear Congruential Generator (LCG)** is one of the earliest and most widely used methods for generating **pseudo-random numbers**. It is a type of **recursive formula** that produces a sequence of numbers that appear random but are **deterministic**.

An LCG is defined by the recurrence relation:

\[
X_{n+1} = (a X_n + c) \mod m
\]

where:
- \( X_n \) is the sequence of pseudo-random numbers,
- \( a \) is the **multiplier** (\( a > 0 \)),
- \( c \) is the **increment** (\( c \geq 0 \)),
- \( m \) is the **modulus** (\( m > 0 \)),
- \( X_0 \) is the **seed** (initial value).

The sequence \( X_n \) generates integers in the range \( 0 \leq X_n < m \). The pseudo-random numbers are obtained by normalizing:

\[
U_n = \frac{X_n}{m}, \quad 0 \leq U_n < 1
\]

where \( U_n \) approximates a **uniform distribution** on \( [0,1) \).

---

## **2. Characteristics of LCG**
### **2.1. Periodicity**
Since the sequence is **deterministic** and operates modulo \( m \), it is guaranteed to **repeat** after at most \( m \) iterations. The **maximum possible period** is \( m \), which occurs when the generator satisfies the following conditions (from Hull-Dobell theorem):

1. \( c \) is **relatively prime** to \( m \) (i.e., \( \gcd(c, m) = 1 \)),
2. \( a-1 \) is a multiple of every **prime divisor** of \( m \),
3. \( a-1 \) is a multiple of **4** if \( m \) is a multiple of \( 4 \).

When \( c = 0 \), the generator is called a **Multiplicative Congruential Generator (MCG)**, but it has a smaller period.

### **2.2. Choice of Parameters**
- **Good choices** of \( a, c, m \) ensure a **long period** and **good randomness properties**.
- **Common parameters** used in practice:
  - **RANDU (old IBM LCG)**
    \[
    a = 65539, \quad c = 0, \quad m = 2^{31}
    \]
    (bad randomness properties)
  - **Minimal standard LCG (Park & Miller, 1988)**
    \[
    a = 16807, \quad c = 0, \quad m = 2^{31} - 1
    \]
  - **Numerical Recipes LCG**
    \[
    a = 1664525, \quad c = 1013904223, \quad m = 2^{32}
    \]

---

## **3. Python Implementation**
The following function implements an **LCG** to generate a sequence of pseudo-random numbers.

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

# Example: Generate and visualize 1000 random numbers
lcg = LinearCongruentialGenerator(seed=42)
random_numbers = lcg.generate(1000)

# Histogram to check uniformity
plt.hist(random_numbers, bins=20, density=True, alpha=0.7, color='blue')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of LCG-generated Numbers")
plt.show()
```

### **3.1. Explanation of Implementation**
- The class **`LinearCongruentialGenerator`** initializes with parameters \( a, c, m \) and a **seed**.
- The **`next()`** function computes the next random number using the **LCG formula**.
- The **`generate(n)`** function returns a list of \( n \) pseudo-random numbers.
- The generated numbers are plotted as a **histogram** to check **uniformity**.

---

## **4. Properties and Issues**
### **4.1. Advantages**
✅ **Simple and Efficient**: Requires only multiplication, addition, and modulus operations.  
✅ **Deterministic**: The same seed produces the **same sequence** of numbers (useful for reproducibility).  
✅ **Fast**: Works well for applications requiring large quantities of random numbers.

### **4.2. Limitations**
❌ **Short Period**: The maximum period is at most \( m \), which is much smaller than modern **cryptographic generators**.  
❌ **Poor High-Dimensional Randomness**: Successive values often fall into **hyperplanes** in higher dimensions (Marsaglia’s lattice test).  
❌ **Not Cryptographically Secure**: Since LCG is **predictable**, it should **not be used for security applications**.

---

## **5. Comparison with Other PRNGs**
| **Generator** | **Speed** | **Period** | **Security** | **Common Use** |
|--------------|----------|------------|-------------|----------------|
| **LCG** | ✅ Fast | Short (\( \leq m \)) | ❌ Weak | Basic simulations |
| **Mersenne Twister** | ✅ Fast | \( 2^{19937} - 1 \) | ❌ Weak | Default in NumPy, SciPy |
| **Xoshiro256** | ✅✅ Very Fast | Long | ❌ Weak | Games, Simulations |
| **Cryptographic PRNGs** | ❌ Slower | Very Long | ✅ Secure | Cryptography, Security |

- **For general simulations**, use **Mersenne Twister (MT19937)** (default in NumPy).
- **For cryptographic security**, use **cryptographically secure generators** like **`secrets`** or **ChaCha20**.

---

## **6. Conclusion**
- The **Linear Congruential Generator (LCG)** is a **simple** and **efficient** method for generating pseudo-random numbers.
- **Good parameter selection** improves randomness and **maximizes the period**.
- LCG is **fast**, but has **poor randomness properties** in high dimensions and is **not suitable for cryptography**.
- **Modern alternatives**, such as **Mersenne Twister** or **Xoshiro**, are preferred for most applications.

Would you like an **extension on testing randomness (e.g., Chi-square test, spectral test)?** 🚀