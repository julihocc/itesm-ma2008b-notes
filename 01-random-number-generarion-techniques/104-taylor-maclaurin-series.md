# **Numerical Analysis for Non-Linear Optimization | Module 1**

## **Taylor and Maclaurin Series**

### **1. Introduction**

The **Taylor series** and its special case, the **Maclaurin series**, are fundamental tools in numerical analysis, particularly for **function approximation**, **error analysis**, and **stochastic simulations**. These series allow functions to be expressed as infinite polynomial expansions, which are useful for efficient numerical computation.

---

### **2. Taylor Series Expansion**

A **Taylor series** expands a function \( f(x) \) about a point \( x_0 \) as:
\[
 f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \frac{f'''(x_0)}{3!}(x - x_0)^3 + \dots
\]
where \( f^n(x_0) \) is the \( n \)th derivative of \( f(x) \) evaluated at \( x_0 \).

#### **2.1 Properties of Taylor Series**
- **Convergence**: The series converges to \( f(x) \) if the remainder term tends to zero.
- **Error Approximation**: Higher-order terms provide better accuracy.
- **Local Approximation**: Good for functions that are smooth and differentiable.

#### **2.2 Example: Approximating \( e^x \) Using Taylor Series**

```python
import numpy as np
import matplotlib.pyplot as plt

def taylor_expansion_e_x(x, n_terms):
    return sum([(x**n) / np.math.factorial(n) for n in range(n_terms)])

x_values = np.linspace(-2, 2, 100)
y_actual = np.exp(x_values)
y_approx = [taylor_expansion_e_x(x, 5) for x in x_values]

plt.plot(x_values, y_actual, label="Actual e^x")
plt.plot(x_values, y_approx, '--', label="Taylor Approximation (n=5)")
plt.legend()
plt.xlabel("x")
plt.ylabel("e^x")
plt.title("Taylor Series Approximation of e^x")
plt.show()
```

---

### **3. Maclaurin Series: Special Case of Taylor Series**

A **Maclaurin series** is a Taylor series expanded about \( x_0 = 0 \):
\[
 f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \dots
\]

#### **3.1 Common Maclaurin Series Expansions**

| Function | Maclaurin Series |
|----------|-----------------|
| \( e^x \) | \( 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots \) |
| \( \sin x \) | \( x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots \) |
| \( \cos x \) | \( 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \dots \) |

#### **3.2 Example: Approximating \( \sin(x) \) Using Maclaurin Series**

```python
def maclaurin_sin(x, n_terms):
    return sum([((-1)**n * x**(2*n+1)) / np.math.factorial(2*n+1) for n in range(n_terms)])

x_values = np.linspace(-np.pi, np.pi, 100)
y_actual = np.sin(x_values)
y_approx = [maclaurin_sin(x, 5) for x in x_values]

plt.plot(x_values, y_actual, label="Actual sin(x)")
plt.plot(x_values, y_approx, '--', label="Maclaurin Approximation (n=5)")
plt.legend()
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Maclaurin Series Approximation of sin(x)")
plt.show()
```

---

### **4. Applications of Taylor and Maclaurin Series**

#### **4.1 Random Number Generation**
- Used in **approximating inverse cumulative distribution functions (CDFs)**.
- Faster than direct computation of transcendental functions.

#### **4.2 Error Analysis in Monte Carlo Simulations**
- Taylor expansion helps **quantify truncation errors** in numerical estimations.
- Used in **stochastic differential equations (SDEs)** for **Brownian motion approximations**.

#### **4.3 Optimization Algorithms**
- **Gradient Descent**: Taylor series provides first and second-order approximations.
- **Newton's Method**: Uses second-order derivatives for root finding.

```python
def newton_method(f, f_prime, x0, tol=1e-5, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / f_prime(x)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x

# Example: Finding root of f(x) = x^2 - 2
print("Root approximation:", newton_method(lambda x: x**2 - 2, lambda x: 2*x, 1))
```

---

### **5. Conclusion**

Taylor and Maclaurin series are essential in numerical methods for **approximating functions**, **optimizing algorithms**, and **analyzing errors**. Their applications extend to **random number generation**, **stochastic simulations**, and **scientific computing**, making them fundamental in numerical analysis.

---

### **6. Exercises**

#### **Basic Implementations**
1. Compute the **Taylor series approximation of \( \cos(x) \)** up to 6 terms and compare it with NumPy's implementation.
2. Implement and plot the **Maclaurin series of \( \tan(x) \)**.

#### **Advanced Applications**
1. Use the **Taylor series to approximate the inverse of a function** numerically.
2. Apply **Taylor expansion to analyze truncation error** in Monte Carlo integration.

