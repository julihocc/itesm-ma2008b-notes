# **Monte Carlo Simulation for Estimating Value at Risk (VaR)**

## **1. Introduction**

This lecture presents a **Monte Carlo simulation** method for estimating the **Value at Risk (VaR)** of a financial asset or portfolio under the **Geometric Brownian Motion (GBM) model**. VaR is a risk measure that quantifies the **potential loss** of an investment over a specified time horizon with a given confidence level.

For a given confidence level $ \alpha $, the **Value at Risk** is defined as:

$$
\text{VaR}_\alpha = \inf \{ x \in \mathbb{R} \mid P(L > x) \leq 1 - \alpha \}
$$

where:

- $ L $ represents **losses** over a given time horizon $ T $,
- $ \alpha $ is the **confidence level** (e.g., 95% or 99%),
- $ \text{VaR}_\alpha $ represents the **worst expected loss** over $ T $ with probability $ 1 - \alpha $.

In other words, **VaR estimates the maximum expected loss at a given confidence level**.

---

## **2. Geometric Brownian Motion (GBM) for Asset Prices**

The underlying asset price follows the **GBM model**:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

where:

- $ S_t $ is the asset price at time $ t $,
- $ \mu $ is the **expected return**,
- $ \sigma $ is the **volatility** (standard deviation of returns),
- $ W_t $ is a **standard Wiener process**.

The **closed-form solution** for the asset price at time $ T $ is:

$$
S_T = S_0 e^{( \mu - \frac{1}{2} \sigma^2) T + \sigma \sqrt{T} Z}
$$

where:

- $ S_0 $ is the **initial asset value**,
- $ Z \sim \mathcal{N}(0,1) $ is a standard normal random variable.

Losses are then defined as:

$$
L = S_0 - S_T
$$

VaR is computed as a **percentile** of the simulated loss distribution.

---

## **3. Monte Carlo Simulation Approach**

The **Monte Carlo method** estimates the loss distribution by generating many possible future asset prices and computing the corresponding losses. The **percentile function** is then used to determine the VaR threshold.

The approach can be summarized as follows:

1. Simulate future asset values $ S_T $ using the **GBM model**.
2. Compute the corresponding losses $ L = S_0 - S_T $.
3. Estimate $ \text{VaR}_\alpha $ as the $ (1 - \alpha) $-quantile of the loss distribution.

---

## **4. Python Implementation**

The function `monte_carlo_var` implements this approach:

```python
import numpy as np
import scipy.stats as stats

def monte_carlo_var(initial_value, mu, sigma, T, alpha, num_simulations):
    dt = T  # Single-step simulation
    
    # Generate standard normal samples
    Z = np.random.normal(0, 1, num_simulations)
    
    # Simulate asset price at T using GBM
    ST = initial_value * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Compute losses
    losses = initial_value - ST
    
    # Compute Value at Risk (VaR) as the (1-alpha) percentile of losses
    var_estimate = np.percentile(losses, 100 * alpha)
    
    return var_estimate

# Parameters
initial_value = 1_000_000  # Portfolio initial value ($1M)
mu = 0.07  # Expected return (7% annually)
sigma = 0.2  # Volatility (20% annually)
T = 1  # Time horizon (1 year)
alpha = 0.95  # 95% confidence level
num_simulations = 100000  # Number of Monte Carlo simulations

# Compute VaR
var_value = monte_carlo_var(initial_value, mu, sigma, T, alpha, num_simulations)

print(f"Estimated 95% Value at Risk (VaR): ${var_value:,.2f}")
```

---

## **5. Step-by-Step Breakdown**

| **Step** | **Mathematical Formula** | **Code Implementation** |
|----------|-------------------------|--------------------------|
| **Generate random normal samples** | $ Z \sim \mathcal{N}(0,1) $ | `Z = np.random.normal(0, 1, num_simulations)` |
| **Simulate asset prices at $ T $** | $ S_T = S_0 e^{( \mu - \frac{1}{2} \sigma^2) T + \sigma \sqrt{T} Z} $ | `ST = initial_value * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)` |
| **Compute portfolio losses** | $ L = S_0 - S_T $ | `losses = initial_value - ST` |
| **Compute VaR as the $ (1 - \alpha) $-percentile** | $ \text{VaR}_\alpha = \text{Percentile}(L, 100(1-\alpha)) $ | `np.percentile(losses, 100 * (1 - alpha))` |

---

## **6. Interpretation of Results**

The output provides an estimated **Value at Risk (VaR)** at the specified confidence level.

Example output:

```
Estimated 95% Value at Risk (VaR): $73,214.56
```

- This means that with **95% confidence**, the maximum expected loss over **one year** does not exceed **$73,214.56**.
- There is a **5% probability** that losses exceed this threshold.

---

## **7. Advantages and Limitations**

### **Advantages**

✅ **Flexible**: Can be applied to **any distribution** of asset returns.  
✅ **Captures Non-Linearity**: Works well with **non-normal** asset return distributions.  
✅ **Handles Complex Portfolios**: Can be extended to **multi-asset portfolios**.

### **Limitations**

❌ **Computationally Expensive**: Requires **large simulations** for accuracy.  
❌ **Monte Carlo Variance**: Results **fluctuate** with different random seeds.  
❌ **Assumes GBM**: Ignores **fat tails** and **market crashes** (e.g., Black Swan events).  

---

## **8. Alternative VaR Methods**

Besides Monte Carlo, VaR can also be estimated using:

1. **Historical VaR**: Uses past market data instead of simulations.
2. **Parametric VaR (Variance-Covariance Method)**: Assumes normally distributed returns and estimates VaR as:

   $$
   \text{VaR}_\alpha = S_0 \cdot ( \mu - z_{\alpha} \sigma )
   $$

   where $ z_{\alpha} $ is the quantile of a standard normal distribution.

Monte Carlo **outperforms** these methods for non-normal distributions but at a higher computational cost.

---

## **9. Conclusion**

- The **Monte Carlo method** simulates **thousands of potential outcomes** to estimate the **risk of extreme losses**.
- The **percentile function** determines **Value at Risk (VaR)** at a given confidence level.
- **This method is widely used in risk management** by banks, hedge funds, and regulators (e.g., Basel III framework).