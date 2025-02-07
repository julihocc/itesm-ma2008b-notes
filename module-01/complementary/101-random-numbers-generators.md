# Numerical Analysis for Non-Linear Optimization

## Module 2

### Random Number Generators

#### Introduction

Random number generators are algorithms that produce sequences of numbers that appear to be random. These sequences are essential for a wide range of applications in scientific computing, including:

* **Monte Carlo methods:** Numerical integration, optimization, and simulation
* **Stochastic simulations:** Modeling complex systems with inherent randomness
* **Cryptography:** Secure communication and data protection
* **Statistical analysis:** Simulating random variables and testing hypotheses

#### Pseudo-Random Number Generators (PRNGs)

Most random number generators used in computers are **pseudo-random number generators (PRNGs)**. These algorithms are deterministic, meaning that given the same initial conditions (seed), they will produce the same sequence of numbers. PRNGs are designed to mimic the statistical properties of truly random sequences. The numbers are usually generated from a uniform distribution, usually over the interval [0, 1).

##### Key Characteristics of PRNGs

* **Deterministic:** Given the same initial state (seed), the generator will produce the same sequence of numbers.
* **Statistical randomness:** PRNGs are designed to pass various statistical tests for randomness.
* **Uniform distribution:** Many PRNGs generate numbers from a uniform distribution over a specific interval, typically [0, 1).
* **Periodicity:** PRNGs eventually repeat their sequences, though well-designed PRNGs have very long periods.

##### Common PRNG Functions in Python

Python provides several tools for generating random numbers, primarily through the NumPy and SciPy libraries:

* **`numpy.random` module:** This module is the workhorse for generating random numbers in scientific Python. It offers functions for various distributions:
  * **`rand`:** Generates uniformly distributed floating-point numbers between 0 (inclusive) and 1 (exclusive).
  * **`randn`:** Produces samples from the standard normal (Gaussian) distribution, with a mean of 0 and a standard deviation of 1.
  * **`randint`:** Creates arrays of random integers within a specified range. The lower limit is inclusive, while the upper limit is exclusive.
  * **`choice`:** Randomly samples items from a given list or array, with or without replacement.
* **`scipy.stats` module:** This module provides a higher-level interface for working with random variables and probability distributions. It includes functions to sample random numbers from many distributions, compute probability densities (PDF), and other statistical calculations.

##### Managing Random Number Generation

* **`RandomState` Class:** The `RandomState` class in NumPy provides a way to manage the state of the random number generator. By instantiating a `RandomState` object with or without a seed, you can create independent random number generators. This practice improves the isolation of code when using random numbers, and is considered good programming practice.
* **Seeding:** Setting a seed for the random number generator allows you to obtain the same sequence of random numbers in different runs of your code, which is essential for reproducibility. This is useful for testing and debugging.
* **Reproducibility:** It is considered good programming practice to use a RandomState instance rather than directly using the functions in the `np.random` module.

##### Various Statistical Distributions

Beyond uniform and normal distributions, many other statistical distributions are available in `numpy.random` and `scipy.stats`:

* **Discrete distributions:** Bernoulli, binomial, Poisson
* **Continuous distributions:** Exponential, chi-squared, Student's t, F distributions

These distributions are critical for simulating various random processes and phenomena in scientific modeling and data science.

##### Quasi-Random Number Generators

While most applications use PRNGs, **quasi-random number generators** offer another approach that might be useful for certain numerical methods. Quasi-random numbers, also known as low-discrepancy sequences, are designed to fill a space more uniformly than pseudo-random numbers. These are often used in **quasi-Monte Carlo methods** to improve the convergence rate of numerical integration.

#### Conclusion

Random number generators are fundamental tools in scientific computing and data science, enabling simulations and statistical analysis. These generators provide a way to model random phenomena and explore complex systems under uncertainty. The Python ecosystem with libraries like NumPy and SciPy provides comprehensive support for random number generation from various statistical distributions. Proper use of PRNGs, including seeding and using `RandomState` objects, is essential for reliable and reproducible scientific computing.

Here are some exercises on random number generators, drawing from the concepts discussed in the sources and our conversation history:

#### Exercises

##### Exercises on Basic Random Number Generation

* **Exercise 1: Uniform Random Numbers**
  * Use NumPy's `random.rand()` function to generate a 10x10 array of uniformly distributed random numbers between 0 and 1.
  * Calculate and display the mean and standard deviation of the generated numbers. Do these match what is expected for a uniform distribution?
* **Exercise 2: Integer Random Numbers**
  * Use NumPy's `random.randint()` function to create a 1D array of 20 random integers between 1 and 10 (inclusive).
  * Count the frequency of each unique integer and display the result.
* **Exercise 3: Gaussian Random Numbers**
  * Use NumPy's `random.randn()` function to generate a 100x1 array of random numbers following a standard normal distribution.
  * Plot a histogram of the generated numbers. Do they resemble a bell curve?
  * Calculate and display the mean and standard deviation of the generated numbers. Do they match the mean and standard deviation for the standard normal distribution?
* **Exercise 4: Reproducibility with Seeding**
  * Create a `RandomState` object using a seed value of 42.
  * Generate 5 random numbers with `rand()`.
  * Repeat the process. Should you get the same 5 random numbers in both cases?
  * Create another `RandomState` object with a different seed and generate 5 random numbers with `rand()`. Are these results different than with the first seed?

##### Exercises on Sampling and Choices

* **Exercise 5: Random Choices**
  * Create a list of 10 colors.
  * Use `random.choice()` to randomly select 3 colors from the list, with replacement.
  * Repeat the selection, but now *without* replacement. How does this change the outcome?
* **Exercise 6: Sampling with Probabilities**
  * Create a list of 5 fruits.
  * Create another list of corresponding probabilities for each fruit (they should add up to 1).
  * Use `random.choice()` to simulate sampling 100 fruits based on the given probabilities.
  * Compare the frequencies of each sampled fruit to their defined probabilities.

##### Exercises on Statistical Distributions

* **Exercise 7: Exponential Distribution**
  * Use SciPy's `stats.expon.rvs()` to generate 100 random numbers following an exponential distribution with a rate parameter of 0.5.
  * Plot the distribution with a histogram.
* **Exercise 8: Binomial Distribution**
  * Use SciPy's `stats.binom.rvs()` to simulate 100 trials of a binomial experiment, where the number of trials per experiment is 10 and the probability of success is 0.4.
  * Calculate and plot the distribution of the number of successes.
* **Exercise 9: Chi-Squared Distribution**
  * Use SciPy's `stats.chi2.rvs()` to generate 1000 random numbers from a chi-squared distribution with 5 degrees of freedom.
  * Plot the distribution of random numbers.

##### Advanced Exercises on Random Number Generators

* **Exercise 10: Monte Carlo Integration**
  * Use random numbers to estimate the value of Pi by randomly generating points in a square and counting how many fall inside a circle inscribed within it. Compare your results to an analytic value of Pi, and discuss any deviations.
* **Exercise 11: Simulating a Random Walk**
  * Use random number generation to simulate a 2D random walk of 100 steps.
  * Plot the path of the walk.
* **Exercise 12: Comparing PRNGs**
  * Generate sequences of random numbers from two different PRNG algorithms.
  * Apply statistical tests (e.g., chi-squared test for uniformity) to compare their apparent randomness.
* **Exercise 13: Implementing a Simple PRNG**
  * Implement a simple Linear Congruential Generator (LCG) from scratch.
  * Compare the statistical properties of your LCG to NumPy's built-in PRNG by generating sequences and plotting them.

These exercises progressively cover different aspects of random number generation, from basic usage to more advanced applications. By completing these exercises, you can gain a deeper understanding of how PRNGs work and how they are used in scientific computing.
