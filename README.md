# Numerical Analysis for Non-Linear Optimization

## Module 2

### Random Number Generators

#### Introduction

Random number generators (RNGs) are algorithms that produce sequences of numbers that appear to be random. These sequences are crucial in scientific computing for applications such as:

- **Monte Carlo methods:** Used in numerical integration, optimization, and simulation.
- **Stochastic simulations:** For modeling systems with inherent randomness.
- **Cryptography:** Ensures secure communication and data protection.
- **Statistical analysis:** Helps in simulating random variables and testing hypotheses.

#### Pseudo-Random Number Generators (PRNGs)

Most random number generators used in computers are **pseudo-random number generators (PRNGs)**. These algorithms are deterministic, meaning that given the same initial conditions (seed), they will produce the same sequence of numbers. PRNGs aim to mimic the statistical properties of truly random sequences. The numbers are usually generated from a uniform distribution over the interval \([0,1)\).

##### Key Characteristics of PRNGs

- **Deterministic:** Given the same initial state (seed), the generator produces the same sequence of numbers.
- **Statistical randomness:** PRNGs pass various statistical tests for randomness.
- **Uniform distribution:** Many PRNGs generate numbers from a uniform distribution over a specific interval, typically \([0,1)\).
- **Periodicity:** PRNGs eventually repeat their sequences, though well-designed PRNGs have very long periods.

##### Common PRNG Functions in Python

Python provides several tools for generating random numbers, primarily through the NumPy and SciPy libraries:

- **`numpy.random` module:**
  - **`rand`:** Generates uniformly distributed floating-point numbers in \([0,1)\).
  - **`randn`:** Produces samples from the standard normal (Gaussian) distribution.
  - **`randint`:** Generates random integers within a specified range.
  - **`choice`:** Randomly selects items from a list or array, with or without replacement.
- **`scipy.stats` module:** Provides a higher-level interface for working with probability distributions, offering random sampling, probability density computation, and statistical calculations.

##### Managing Random Number Generation

- **`RandomState` Class:** NumPy's `RandomState` class allows users to manage the state of the random number generator, improving reproducibility.
- **Seeding:** Setting a seed ensures that random sequences are reproducible, which is useful for testing and debugging.
- **Reproducibility:** Using a `RandomState` instance rather than direct function calls in `np.random` is recommended for isolating randomness in different parts of a program.

##### Various Statistical Distributions

Beyond uniform and normal distributions, `numpy.random` and `scipy.stats` support other statistical distributions, including:

- **Discrete distributions:** Bernoulli, binomial, Poisson.
- **Continuous distributions:** Exponential, chi-squared, Student's t, F-distributions.

These distributions are critical for simulating different random processes in scientific computing and data science.

##### Randomness Testing and Quality Assessment

It is important to assess the quality of PRNGs using **randomness tests**, such as:

- **Chi-square test:** Tests if a sample follows a given distribution.
- **Kolmogorov-Smirnov test:** Compares a sample’s empirical distribution to a theoretical distribution.
- **Autocorrelation tests:** Checks for dependency between values in a sequence.
- **Diehard tests & TestU01 suite:** Advanced statistical tests for PRNG evaluation.

##### Quasi-Random Number Generators

While PRNGs produce seemingly random sequences, **quasi-random number generators (QRNGs)** produce low-discrepancy sequences that fill space more uniformly. These are often used in **quasi-Monte Carlo methods** to improve convergence in numerical integration.

Examples of QRNGs include:

- **Sobol sequences:** Used in high-dimensional integration.
- **Halton sequences:** Suitable for moderate-dimensional spaces.
- **Faure sequences:** Alternative to Halton sequences with better uniformity.

#### Hardware Random Number Generators (HRNGs)

Unlike PRNGs, **hardware random number generators (HRNGs)** use physical processes (e.g., electrical noise, radioactive decay) to generate truly random numbers. These are essential in cryptography and secure computing.

### Monte Carlo Applications

Monte Carlo methods use random sampling to approximate deterministic problems. Some important applications include:

1. **Estimating Pi:** Randomly placing points in a square and counting how many fall within an inscribed circle.
2. **Solving integrals:** Approximating integrals in higher dimensions using Monte Carlo techniques.
3. **Stochastic optimization:** Techniques like simulated annealing rely on PRNGs for exploration of solution spaces.

### Exercises

#### Exercises on Basic Random Number Generation

1. **Uniform Random Numbers**
   - Use `numpy.random.rand()` to generate a \(10 \times 10\) array of uniformly distributed random numbers in \([0,1)\).
   - Compute the mean and standard deviation. Do these match the expected values?

2. **Integer Random Numbers**
   - Generate a 1D array of 20 random integers between 1 and 10 (inclusive).
   - Count the frequency of each integer and display the results.

3. **Gaussian Random Numbers**
   - Generate 100 samples from a standard normal distribution using `numpy.random.randn()`.
   - Plot a histogram and verify whether it follows a bell curve.
   - Compute the mean and standard deviation.

4. **Reproducibility with Seeding**
   - Generate random numbers using a fixed seed and verify reproducibility.

#### Exercises on Sampling and Choices

5. **Random Choices**
   - Select random colors from a predefined list, with and without replacement.

6. **Sampling with Probabilities**
   - Assign probabilities to five different fruits and sample 100 times.
   - Compare sampled frequencies with expected probabilities.

#### Exercises on Statistical Distributions

7. **Exponential Distribution**
   - Generate and plot 100 samples from an exponential distribution.

8. **Binomial Distribution**
   - Simulate 100 binomial trials with \( n = 10 \), \( p = 0.4 \) and analyze the results.

9. **Chi-Squared Distribution**
   - Generate and plot 1000 samples from a chi-squared distribution.

#### Advanced Exercises on Random Number Generators

10. **Monte Carlo Integration**
    - Estimate Pi using Monte Carlo methods.

11. **Simulating a Random Walk**
    - Create a 2D random walk and visualize the trajectory.

12. **Comparing PRNGs**
    - Generate random sequences using different PRNGs and compare their statistical properties.

13. **Implementing a Simple PRNG**
    - Implement a Linear Congruential Generator (LCG) and analyze its output.

By completing these exercises, one gains a comprehensive understanding of PRNGs and their applications in scientific computing.

