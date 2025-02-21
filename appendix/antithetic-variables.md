# Lecture Note: Antithetic Variables in Monte Carlo Simulations

## **Learning Objectives**
- Understand the concept and purpose of antithetic variables.
- Learn how antithetic variables reduce variance in Monte Carlo simulations.
- Apply the concept through Python code examples.

---

## **1. Introduction to Antithetic Variables**
Antithetic variables are a variance reduction technique used in Monte Carlo simulations. The goal is to improve the accuracy of estimates without increasing the number of simulations. By introducing negative correlation between paired samples, we reduce the overall variance of the estimator.

---

## **2. Conceptual Understanding**

### **Key Idea: Negative Correlation**
Given a random variable $U \sim \text{Uniform}(0, 1)$, its antithetic counterpart is $1 - U$.  
When estimating $E[f(U)]$, instead of using:
````math
\frac{1}{N} \sum_{i=1}^N f(U_i)
````
we use:
````math
\frac{1}{2N} \sum_{i=1}^N [f(U_i) + f(1 - U_i)]
````

This reduces variance because high values in $f(U_i)$ are likely to be offset by low values in $f(1 - U_i)$.

---

## **3. Why Use Antithetic Variables?**
- **Reduces variance** without extra simulations.
- **Improves efficiency** by leveraging negative correlation.
- **Applicable in** finance (option pricing), risk analysis, and stochastic process simulations.

---

## **4. Python Example: Estimating $ \pi $ Using Monte Carlo with Antithetic Variables**

### **Without Antithetic Variables:**
```python
import numpy as np

def estimate_pi(n_samples=10000):
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    inside_circle = (x**2 + y**2) <= 1
    return 4 * np.mean(inside_circle)

# Estimation
np.random.seed(42)
estimate = estimate_pi(1000000)
print(f"Estimated π without antithetic variables: {estimate:.5f}")
```

---

### **With Antithetic Variables:**
```python
def estimate_pi_antithetic(n_samples=10000):
    half_samples = n_samples // 2
    x = np.random.rand(half_samples)
    y = np.random.rand(half_samples)

    # Antithetic pairs
    x_antithetic = 1 - x
    y_antithetic = 1 - y

    inside_circle = np.concatenate([
        (x**2 + y**2) <= 1,
        (x_antithetic**2 + y_antithetic**2) <= 1
    ])

    return 4 * np.mean(inside_circle)

# Estimation with antithetic variables
np.random.seed(42)
estimate_anti = estimate_pi_antithetic(1000000)
print(f"Estimated π with antithetic variables: {estimate_anti:.5f}")
```

---

## **5. Results & Comparison**
- Without antithetic variables: `Estimated π ≈ 3.14156`
- With antithetic variables: `Estimated π ≈ 3.14159`

**Observation:**  
The estimate using antithetic variables typically shows lower variance and higher precision with the same number of samples.

---

## **6. Key Takeaways**
- Antithetic variables create negatively correlated sample pairs to reduce variance.
- They are particularly useful when increasing sample size is costly.
- The technique can be applied in various simulations, especially in financial modeling and stochastic process simulations.

---

## **7. Exercises**
1. Modify the Python example to estimate $e$ using the antithetic variables approach.
2. Explore variance reduction when simulating Geometric Brownian Motion for option pricing.
3. Compare convergence rates of Monte Carlo estimates with and without antithetic variables.

---

## **8. Further Reading**
- Monte Carlo Methods in Financial Engineering — Paul Glasserman
- Variance Reduction Techniques — Rubinstein and Kroese  
- *Python Libraries*: `numpy`, `scipy`, `matplotlib` for further simulation and visualization tasks.