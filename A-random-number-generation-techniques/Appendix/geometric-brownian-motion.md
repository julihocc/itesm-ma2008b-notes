# Module 1: Random Number Generators and Applications

# Appendix: Geometric Brownian Motion (GBM) for Stock Price Modeling

### **Geometric Brownian Motion (GBM) for Stock Price Modeling**

The given formula:

\[
S_t = S_0 \cdot \exp\left( (\mu - \frac{1}{2} \sigma^2) t + \sigma W_t \right)
\]

describes the evolution of a stock price \( S_t \) over time under the **Geometric Brownian Motion (GBM)** model. This model is fundamental in financial mathematics and is widely used in **option pricing**, such as in the **Black-Scholes model**.

---

### **1. Stochastic Differential Equation (SDE) Representation**

GBM is defined by the following **stochastic differential equation (SDE)**:

\[
dS_t = \mu S_t dt + \sigma S_t dW_t,
\]

where:
- \( S_t \) is the stock price at time \( t \),
- \( \mu \) is the **drift coefficient** (expected return),
- \( \sigma \) is the **volatility** (standard deviation of returns),
- \( W_t \) is a **standard Brownian motion** (Wiener process) with \( W_0 = 0 \) and increments \( dW_t \sim \mathcal{N}(0, dt) \).

Applying **Itô's Lemma** to the logarithm of \( S_t \), we obtain the closed-form solution:

\[
S_t = S_0 \cdot \exp\left( (\mu - \frac{1}{2} \sigma^2) t + \sigma W_t \right).
\]

---

### **2. Interpretation of the Components**

- \( S_0 \) is the initial stock price at \( t = 0 \).
- The term \( (\mu - \frac{1}{2} \sigma^2) t \) represents the **deterministic drift**, which accounts for the expected growth of the stock price.
- The term \( \sigma W_t \) represents the **stochastic component**, incorporating market randomness via a Wiener process.
- The factor \( -\frac{1}{2} \sigma^2 \) is a **correction term** (from Itô's Lemma) ensuring that the expected value of \( S_t \) follows the deterministic drift.

---

### **3. Expected Value and Variance**
Taking the expectation:

\[
\mathbb{E}[S_t] = S_0 e^{\mu t}.
\]

This shows that the expected stock price follows an **exponential growth** at rate \( \mu \), without considering volatility.

The variance is given by:

\[
\text{Var}(S_t) = S_0^2 e^{2\mu t} \left(e^{\sigma^2 t} - 1\right).
\]

Higher volatility \( \sigma \) increases the uncertainty in future stock prices.

---

### **4. Log-normal Distribution of Stock Prices**
Taking the natural logarithm:

\[
\ln S_t = \ln S_0 + (\mu - \frac{1}{2} \sigma^2) t + \sigma W_t.
\]

Since \( W_t \sim \mathcal{N}(0, t) \), we conclude:

\[
\ln S_t \sim \mathcal{N} \left(\ln S_0 + (\mu - \frac{1}{2} \sigma^2) t, \sigma^2 t \right).
\]

Thus, \( S_t \) follows a **log-normal distribution**, meaning that the **logarithm of the stock price is normally distributed**.

---

### **5. Python Simulation**
We can simulate stock prices using the GBM model with discretized Brownian motion:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100   # Initial stock price
mu = 0.1   # Expected return (10% annual growth)
sigma = 0.2  # Volatility (20% annual standard deviation)
T = 1     # Time horizon (1 year)
N = 252   # Number of time steps (daily prices for 1 year)
dt = T / N  # Time step size

# Time grid
t = np.linspace(0, T, N)

# Simulated Brownian motion
brownian_motion = np.cumsum(np.sqrt(dt) * np.random.randn(N))

# Stock price simulation using GBM formula
S_t = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * brownian_motion)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, S_t, label="Simulated Stock Price")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.title("Geometric Brownian Motion Simulation")
plt.legend()
plt.show()
```

---

### **6. Conclusion**
- The **GBM model** captures both **deterministic growth** and **random market fluctuations**.
- It assumes **continuous compounding of returns**, making it well-suited for **option pricing models**.
- Real-world markets **deviate from GBM** due to **jumps, mean reversion, and heavy tails**, but it remains a widely used model in finance.
