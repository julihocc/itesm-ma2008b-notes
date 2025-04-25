# MA2008B | Period 02

## Problem 1 (Solving IVPs Using the Variation of Constants Formula)

Use the variation of constants formula from Theorem 1.6 to solve the following initial value problems of the form:
$$
x' = p(t)x + q(t), \quad x(t_0) = x_0
$$

**Step 1.**  
Identify the functions $p(t)$, $q(t)$, the initial time $t_0$, and the initial value $x_0$ from the given differential equation and initial condition.

**Step 2.**  
Compute the inner integral:
$$
\int_{t_0}^s p(\tau)\, d\tau
$$
This will appear inside the integrand of the second term in the variation of constants formula.

**Step 3.**  
Simplify the integrand:
$$
e^{-\int_{t_0}^s p(\tau)\, d\tau} \cdot q(s)
$$
Write this expression in the simplest possible form using the result from Step 2.

**Step 4.**  
Evaluate the integral:
$$
\int_{t_0}^t \left(e^{-\int_{t_0}^s p(\tau)\, d\tau} \cdot q(s)\right) ds
$$
Use standard calculus techniques to compute it explicitly.

**Step 5.**  
Compute the exponential factor:
$$
\int_{t_0}^t p(\tau)\, d\tau \quad \text{and} \quad e^{\int_{t_0}^t p(\tau)\, d\tau}
$$

**Step 6.**  
Write the full expression for the solution using the formula:
$$
x(t) = e^{\int_{t_0}^t p(\tau)\, d\tau} \cdot x_0 + e^{\int_{t_0}^t p(\tau)\, d\tau} \cdot \left( \text{value from Step 4} \right)
$$

**Step 7.**  
Simplify the final expression for $x(t)$. Use algebraic techniques and common factorizations to combine terms and eliminate redundant expressions. Aim to write a final answer that is easy to interpret, avoids nested exponentials, and shows clearly how $x(t)$ depends on $t$.

**Step 8.**  
Verify that your final expression satisfies the initial condition $x(t_0) = x_0$ and differentiate $x(t)$ and check that it satisfies the original differential equation.

## Problem 2 (Fixed-Point Iteration and Root-Finding Methods)

Let $E: x = e^{-x}$ be the equation under study.

**Step 1.**  
Apply the fixed-point iterative method defined by $x_{n+1} = e^{-x_n}$ with the initial value $x_0 = 0.5$. Continue the iteration until the difference between successive approximations satisfies $|x_{n+1} - x_n| < \varepsilon$, with $\varepsilon = 0.001$. Based on the iteration results, conjecture an approximate value $x^*$ for the solution of the equation.

**Step 2.**  
Prove that the function $f(x) = e^{-x}$ is Lipschitz continuous in a neighborhood of the conjectured root $x^*$ that also contains the initial value $x_0$.

**Step 3.**  
Apply the Newton-Raphson method to approximate the solution of $E$, using the same initial value $x_0 = 0.5$ and the same stopping criterion $|x_{n+1} - x_n| < \varepsilon$.

**Step 4.**  
Use the bisection method to approximate the solution of $E$, with stopping criterion $|b_n - a_n| < \varepsilon$, where $[a_n, b_n]$ is the current interval. Choose a valid initial interval $[a, b]$ that contains the root, based on your observations in Step 1.

**Step 5.**  
Apply the *regula falsi* (false position) method to the same interval and stopping criterion used in Step 4. Compare its efficiency to the bisection method.

## Problem 3 (Numerical Integration of the Gaussian Density Function)

Let
$$
f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
$$
be the standard normal (Gaussian) probability density function. Consider the integral
$$
I = \int_{-2}^{2} f(x)\,dx,
$$
which represents the probability that a standard normal random variable lies within two standard deviations of the mean.

**Step 1.**  
Use the **composite trapezoidal rule** with $n = 8$ subintervals to approximate the integral $I$. Write out the general form of the rule and substitute the function values. Report your final approximation to at least six decimal digits.

**Step 2.**  
Repeat the computation using the **composite Simpson’s rule** with the same number of subintervals $n = 8$. Use the alternating coefficients structure $1, 4, 2, 4, \dots, 4, 1$, and compute the result to the same level of accuracy.

**Step 3.**  
Apply the **composite Simpson’s 3/8 rule** with $n = 9$ subintervals. Use the appropriate coefficient pattern $1, 3, 3, 2, 3, 3, 2, \dots, 3, 3, 1$, and write out the weighted sum explicitly before computing the approximation.

**Step 4.**  
Determine the **exact value** of the integral using the error function:
$$
I_{\text{exact}} = \operatorname{erf}\left(\frac{2}{\sqrt{2}}\right) = \operatorname{erf}(\sqrt{2}).
$$
Use a reliable calculator, table, or software to evaluate this quantity numerically to at least six decimal digits.

**Step 5.**  
For each method in Steps 1–3, compute the **absolute error** by subtracting the exact value obtained in Step 4. Discuss the relative accuracy of the methods and comment on how the number of subintervals and the rule used influence the error.

## Problem 4

**Reference:**

> 1. Chasnov, J. R. (2012). Numerical methods. Hong Kong University of Science and Technology. <https://www.math.hkust.edu.hk/~machas/numerical-methods.pdf>

Consider the system
$$
F' = (2-S)F, \quad F(0) = F_0 \\
S' = (F-1)S, \quad S(0) = S_0
$$

**Step 1**
Determine an numerical scheme to approximate the solution and implement  it in vanilla Python. Consider the modified Euler Method for systems, described in the reference, but you're allowed any other method that you consider even better.

**Step 2**
Find a numerical approximation to the solution at $t=1$ if $F_0 = 1.9$, $S_0=0.1$, and $\Delta t = 0.001$. Plot the numerical solution that you found.

**Step 3**
Now find a numerical approximation to the solution at $t=1$ if $F_0 = 1.9$, $S_0=0.1$, and $\Delta t = 0.001$, but now utilize [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). Plot the numerical solution that you found in a different view and compare with the previous solution.

**Step 4**
Find a numerical approximation to the solution at $t=1$ if $F_0 = 1$, $S_0=2$, and $\Delta t = 0.001$, using your own implementation. Plot the numerical solution that you found.

**Step 5**
Based on the system of equations and the corresponding numerical scheme, explain the results from the step 4.
