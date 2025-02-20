# Module 1 | Random Number Generators: Implementation and Applications

## Kolmogorov-Smirnov Test: Statistical Analysis of Randomness

The **Kolmogorov-Smirnov (KS) test** is a **non-parametric statistical test** used to compare a sample with a reference probability distribution (one-sample KS test) or to compare two independent samples (two-sample KS test). It is used to determine whether a given sample follows a specified distribution or whether two samples are drawn from the same distribution.

### 1. **One-Sample KS Test**

This test compares an empirical cumulative distribution function (**ECDF**) of a sample with a theoretical cumulative distribution function (**CDF**) of a reference distribution (e.g., normal, uniform). It tests the null hypothesis:

$$
H_0: F(x) = F_0(x) \quad \text{for all } x
$$

where:

- $ F(x) $ is the empirical CDF of the sample.
- $ F_0(x) $ is the CDF of the theoretical distribution.

The **test statistic** is:

$$
D_n = \sup_x |F_n(x) - F_0(x)|
$$

where:

- $ F_n(x) $ is the empirical CDF based on $ n $ observations.
- $ \sup_x $ denotes the **supremum** (maximum absolute difference).

A large $ D_n $ suggests that the sample does not follow $ F_0(x) $, and the null hypothesis is rejected.

### 2. **Two-Sample KS Test**

This test compares the empirical distributions of two independent samples, testing whether they come from the same underlying distribution. The null hypothesis is:

$$
H_0: F_1(x) = F_2(x) \quad \text{for all } x
$$

where:

- $ F_1(x) $ and $ F_2(x) $ are the empirical CDFs of the two samples.

The **test statistic** is:

$$
D_{n,m} = \sup_x |F_n(x) - G_m(x)|
$$

where:

- $ F_n(x) $ is the empirical CDF of the first sample (size $ n $).
- $ G_m(x) $ is the empirical CDF of the second sample (size $ m $).

A large $ D_{n,m} $ suggests that the two distributions differ significantly.

### 3. **Interpretation**

- The **p-value** indicates the probability of observing the test statistic under $ H_0 $. A small p-value (e.g., $ p < 0.05 $) suggests rejecting $ H_0 $.
- The test is **sensitive to differences in both location and shape** between distributions.
- It works well with **continuous distributions** but may be less reliable for discrete distributions.

### 4. **Implementation in Python**

Using `scipy.stats.kstest` for a **one-sample KS test**:

```python
import numpy as np
from scipy.stats import kstest, norm

# Generate sample data
sample = np.random.normal(loc=0, scale=1, size=100)  # Standard normal sample

# Perform KS test against normal distribution
ks_stat, p_value = kstest(sample, 'norm')

print(f"KS Statistic: {ks_stat}, P-value: {p_value}")
```

Using `scipy.stats.ks_2samp` for a **two-sample KS test**:

```python
from scipy.stats import ks_2samp

# Generate two different distributions
sample1 = np.random.normal(0, 1, 100)
sample2 = np.random.uniform(-1, 1, 100)

# Perform two-sample KS test
ks_stat, p_value = ks_2samp(sample1, sample2)

print(f"KS Statistic: {ks_stat}, P-value: {p_value}")
```

### 5. **Advantages**

- **Non-parametric**: No assumptions about the data distribution.
- **Sensitive to shape differences** between distributions.
- **Works with small sample sizes**.

### 6. **Limitations**

- Less powerful than parametric tests when assumptions hold (e.g., t-test for normal distributions).
- May be **too sensitive** with large sample sizes, detecting minor differences that are not practically significant.
- Less reliable for discrete data.
