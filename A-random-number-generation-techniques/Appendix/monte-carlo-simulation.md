# **Monte Carlo Simulation for Pricing a European Call Option**

## **1. Introduction**
This lecture presents a Monte Carlo simulation approach for estimating the price of a European call option under the **Black-Scholes framework**. The method incorporates **variance reduction** through **antithetic variates** to improve accuracy.

A European call option provides the right (but not the obligation) to purchase an asset at a fixed strike price \( K \) at maturity \( T \). The theoretical price is given by the expected discounted payoff under the **risk-neutral measure**:

\[
C = e^{-rT} \mathbb{E}[\max(S_T - K, 0)]
\]

where:
- \( S_T \) is the stock price at maturity,
- \( K \) is the strike price,
- \( r \) is the risk-free rate,
- \( T \) is the time to maturity.

This expectation is estimated using **Monte Carlo simulation**.

---

## **2. Geometric Brownian Motion (GBM) Model for Stock Prices**
The stock price follows a **Geometric Brownian Motion (GBM)**:

\[
dS_t = \mu S_t dt + \sigma S_t dW_t
\]

where:
- \( \mu \) is the **drift** (expected return),
- \( \sigma \) is the **volatility** (standard deviation of returns),
- \( W_t \) is a **standard Wiener process**.

The closed-form solution for \( S_T \) is:

\[
S_T = S_0 e^{(r - \frac{1}{2} \sigma^2) T + \sigma \sqrt{T} Z}
\]

where:
- \( S_0 \) is the initial stock price,
- \( Z \sim \mathcal{N}(0,1) \) is a standard normal random variable.

---

## **3. Monte Carlo Simulation Approach**
The Monte Carlo method estimates the expectation:

\[
C \approx e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)
\]

where \( N \) is the number of simulated paths.

### **3.1 Variance Reduction via Antithetic Variates**
To improve efficiency, we use **antithetic variates**:
1. Generate **half** of the random normal samples \( U \sim \mathcal{N}(0,1) \).
2. Use their **negatives** \( V = -U \) to generate a second set.
3. This technique **reduces variance** while keeping the expected value unchanged.

---

## **4. Python Implementation**
The function `monte_carlo_european_call` performs the following steps:

```python
import numpy as np

def monte_carlo_european_call(S0, K, T, r, sigma, num_simulations):
    dt = T  # Single-step simulation
    
    # Generate standard normal samples and their antithetic counterparts
    U = np.random.normal(0, 1, num_simulations // 2)
    V = -U
    Z = np.concatenate((U, V))  # Combine for variance reduction
    
    # Simulate stock prices at maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Compute call option payoff
    payoff = np.maximum(ST - K, 0)
    
    # Discounted expected payoff
    discounted_payoff = np.exp(-r * T) * payoff
    
    # Estimate option price and standard error
    return np.mean(discounted_payoff), np.std(discounted_payoff) / np.sqrt(num_simulations)

# Parameters
S0, K, T, r, sigma, num_simulations = 100, 100, 1, 0.05, 0.2, 100000

call_price, error = monte_carlo_european_call(S0, K, T, r, sigma, num_simulations)

print(f"European Call Option Price: {call_price:.4f} ± {error:.4f}")
```

---

## **5. Step-by-Step Breakdown**
| **Step** | **Mathematical Formula** | **Code Implementation** |
|----------|-----------------|---------------------|
| **Generate random normal variables** \( U \sim \mathcal{N}(0,1) \) | \( U, V = -U \) | `U = np.random.normal(0,1, num_simulations//2)` |
| **Simulate terminal stock prices** | \( S_T = S_0 e^{(r - \frac{1}{2} \sigma^2) T + \sigma \sqrt{T} Z} \) | `ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)` |
| **Compute call option payoff** | \( \max(S_T - K, 0) \) | `payoff = np.maximum(ST - K, 0)` |
| **Discount payoff** | \( e^{-rT} \cdot \max(S_T - K, 0) \) | `discounted_payoff = np.exp(-r * T) * payoff` |
| **Estimate option price** | \( C = \frac{1}{N} \sum_{i=1}^{N} e^{-rT} \max(S_T^{(i)} - K, 0) \) | `np.mean(discounted_payoff)` |
| **Estimate standard error** | \( \frac{\sigma_{\text{payoff}}}{\sqrt{N}} \) | `np.std(discounted_payoff) / np.sqrt(num_simulations)` |

---

## **6. Results and Interpretation**
The output consists of:
1. **Estimated Call Option Price**: The Monte Carlo estimate of the theoretical price.
2. **Confidence Interval**: Given by \( \pm \text{standard error} \).

Example output:
```
European Call Option Price: 10.4503 ± 0.0321
```

- The estimated **call option price** is **10.45**.
- The **standard error** is **0.032**, indicating the confidence level of the estimate.

---

## **7. Advantages and Limitations**
### **Advantages**
✅ **Flexible**: Works with complex option payoffs (e.g., exotic options).  
✅ **Handles Any Distribution**: Unlike Black-Scholes, it accommodates non-lognormal stock returns.  
✅ **Improved Efficiency**: Antithetic variates reduce variance without extra simulations.

### **Limitations**
❌ **Slow for High Accuracy**: Requires many simulations for precise estimates.  
❌ **Path Dependency**: Inefficient for options requiring full price paths (e.g., American options).  
❌ **Dependent on Variance Reduction**: Without techniques like **antithetic variates**, results may be noisy.

---

## **8. Conclusion**
- Monte Carlo methods provide a **numerical approach** to estimating option prices.
- The **GBM model** ensures risk-neutral valuation.
- **Antithetic variates** enhance accuracy while reducing computational cost.
- This method is **widely used in financial engineering** for pricing derivatives where closed-form solutions are unavailable.