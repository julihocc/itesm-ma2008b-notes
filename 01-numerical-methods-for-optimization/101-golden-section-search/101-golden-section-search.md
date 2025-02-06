# Golden Section Search

## Overview

The golden-section search is a method for finding the extremum (minimum or maximum) of a single-variable function. It's a bracketing method, meaning it requires initial guesses that bracket a single optimum. It is similar to the bisection method for finding roots, but instead of finding where a function equals zero, it seeks the highest or lowest point of the function.

Here are some key concepts about the golden-section search from the sources:

* **Goal:** The goal of single-variable optimization is to find the value of x that yields an extremum (maximum or minimum) of f(x).
* **Bracketing:** The golden-section search starts with an interval defined by a lower guess (xl) and an upper guess (xu) that bracket a single optimum.
* **Golden Ratio:** The method uses the golden ratio to choose two interior points within the interval. The golden ratio is calculated as  (√5 - 1)/2, which is approximately 0.618.
* **Interior Points:** Two interior points, x1 and x2, are determined using the golden ratio. These points divide the interval in such a way that the ratio of the lengths of the segments is equal to the golden ratio.
* **Interval Reduction:**  The function is evaluated at the two interior points, f(x1) and f(x2). Based on whether you are looking for a maximum or a minimum, the algorithm discards part of the interval that does not contain the extremum. The remaining interval becomes the new interval for the next iteration.  If you are looking for a maximum, the interval defined by xl, x2, and x1 is used if f(x2) > f(x1). If f(x1)> f(x2), the maximum is in the interval defined by x2, x1 and xu. The old value of x2 becomes the new x1, and a new x2 is calculated within the new interval.
* **Iteration:** This process of selecting new interior points and reducing the interval is repeated until a stopping criterion is met. This criterion is often a tolerance based on an approximate relative error.
* **Reduced Function Evaluations:**  A key advantage of the golden-section search is that it minimizes function evaluations by replacing old values with new values. This can be important when the golden-section search is part of a larger calculation or when function evaluations are costly.
* **Convergence:** The golden-section search always converges to the optimum.  However, it is slower than methods like parabolic interpolation, but unlike parabolic interpolation, it does not diverge.

The steps of the golden-section search algorithm are:

1. Define the initial interval with a lower bound, xl and an upper bound, xu
2. Calculate the golden ratio, R = (50.5 - 1)/2.
3. Compute d = R * (xu - xl).
4. Compute x1 = xl + d and x2 = xu - d.
5. Evaluate the function at x1 and x2, f1 = f(x1) and f2 = f(x2).
6. If f1 > f2 (for maximization, or f1 < f2 for minimization), then
    * Update xl = x2
    * Update x2 = x1
    * Update x1 = xl + d
    * Update f2 = f1
    * Update f1 = f(x1)
7. Else
    * Update xu = x1
    * Update x1 = x2
    * Update x2 = xu - d
    * Update f1 = f2
    * Update f2 = f(x2)
8. Calculate an approximate relative error (ea) based on the interval and the optimum, or continue iterating until a maximum number of iterations is reached.
9. Return the optimal x and f(x).

 The golden-section search can be used in combination with other methods. For example, it can be combined with parabolic interpolation in Brent's method to get a method that combines the reliability of golden-section search with the speed of parabolic interpolation.

The golden section search is implemented in the GNU Scientific Library (GSL).
