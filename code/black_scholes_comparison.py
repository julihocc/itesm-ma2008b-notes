#!/usr/bin/env python3
"""
Black-Scholes Methods Comparison
===============================

This script compares different implementations of Black-Scholes option pricing:
- Analytical solution (exact)
- Explicit finite difference (numerical)
- Implicit finite difference (numerical)

The comparison uses the same parameters from w303 example and analyzes
the accuracy and performance of each method.

Author: Based on w303 example
"""

import os
import time
from black_scholes_analytical import black_scholes_call_analytical
from black_scholes_explicit import black_scholes_call_explicit
from black_scholes_implicit import black_scholes_call_implicit


def compare_black_scholes_methods(S0, K, T, r, sigma, S_max=200, M=100, N=1000):
    """
    Compare analytical, explicit, and implicit finite difference methods for Black-Scholes pricing.

    Parameters:
    -----------
    S0 : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free rate (annual)
    sigma : float
        Volatility (annual)
    S_max : float, optional
        Maximum stock price for numerical methods (default: 200)
    M : int, optional
        Number of stock price steps for numerical methods (default: 100)
    N : int, optional
        Number of time steps for numerical methods (default: 1000)

    Returns:
    --------
    dict: Comparison results including prices, errors, and timing for all three methods
    """

    # Calculate analytical solution with timing
    start_time = time.time()
    analytical_price = black_scholes_call_analytical(S0, K, T, r, sigma)
    analytical_time = time.time() - start_time

    # Calculate explicit finite difference solution with timing
    start_time = time.time()
    explicit_price = black_scholes_call_explicit(S0, K, T, r, sigma, S_max, M, N)
    explicit_time = time.time() - start_time

    # Calculate implicit finite difference solution with timing
    start_time = time.time()
    implicit_price = black_scholes_call_implicit(S0, K, T, r, sigma, S_max, M, N)
    implicit_time = time.time() - start_time

    # Calculate error metrics relative to analytical solution
    explicit_absolute_error = abs(explicit_price - analytical_price)
    implicit_absolute_error = abs(implicit_price - analytical_price)

    explicit_relative_error = (
        (explicit_absolute_error / analytical_price) * 100
        if analytical_price != 0
        else 0
    )
    implicit_relative_error = (
        (implicit_absolute_error / analytical_price) * 100
        if analytical_price != 0
        else 0
    )

    # Return comparison results
    return {
        "analytical_price": analytical_price,
        "explicit_price": explicit_price,
        "implicit_price": implicit_price,
        "explicit_absolute_error": explicit_absolute_error,
        "implicit_absolute_error": implicit_absolute_error,
        "explicit_relative_error": explicit_relative_error,
        "implicit_relative_error": implicit_relative_error,
        "analytical_time": analytical_time,
        "explicit_time": explicit_time,
        "implicit_time": implicit_time,
        "grid_params": {"S_max": S_max, "M": M, "N": N},
    }


def write_comparison_report(
    S0,
    K,
    T,
    r,
    sigma,
    S_max=200,
    M=100,
    N=1000,
    filename="black_scholes_comparison_report.txt",
):
    """
    Write a comprehensive comparison report to a file.

    Parameters:
    -----------
    S0 : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free rate (annual)
    sigma : float
        Volatility (annual)
    S_max : float, optional
        Maximum stock price for explicit method (default: 200)
    M : int, optional
        Number of stock price steps for explicit method (default: 100)
    N : int, optional
        Number of time steps for explicit method (default: 1000)
    filename : str, optional
        Output filename (default: "black_scholes_comparison_report.txt")

    Returns:
    --------
    str: Path to the generated report file
    """

    # Perform comparison
    results = compare_black_scholes_methods(S0, K, T, r, sigma, S_max, M, N)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, filename)

    # Write comprehensive report to file
    with open(report_path, "w") as f:
        f.write("Black-Scholes Methods Comparison\n")
        f.write("===============================\n\n")

        # Problem parameters
        f.write("Problem Parameters:\n")
        f.write("------------------\n")
        f.write(f"Current stock price (S₀): ${S0}\n")
        f.write(f"Strike price (K): ${K}\n")
        f.write(f"Time to expiration (T): {T} years\n")
        f.write(f"Risk-free rate (r): {r:.1%}\n")
        f.write(f"Volatility (σ): {sigma:.1%}\n\n")

        # Grid parameters for numerical methods
        f.write("Numerical Methods Grid Parameters:\n")
        f.write("---------------------------------\n")
        f.write(f"Maximum stock price (S_max): {S_max}\n")
        f.write(f"Stock price steps (M): {M}\n")
        f.write(f"Time steps (N): {N}\n")
        f.write(f"Grid spacing (dS): {S_max/M:.2f}\n")
        f.write(f"Time step (dt): {T/N:.6f}\n\n")

        # Results comparison
        f.write("Results Comparison:\n")
        f.write("------------------\n")
        f.write(f"Analytical solution:        ${results['analytical_price']:.4f}\n")
        f.write(f"Explicit finite difference: ${results['explicit_price']:.4f}\n")
        f.write(f"Implicit finite difference: ${results['implicit_price']:.4f}\n\n")

        # Error analysis
        f.write("Error Analysis:\n")
        f.write("--------------\n")
        f.write(f"Explicit method:\n")
        f.write(f"  Absolute error:  ${results['explicit_absolute_error']:.4f}\n")
        f.write(f"  Relative error:  {results['explicit_relative_error']:.2f}%\n")
        f.write(f"Implicit method:\n")
        f.write(f"  Absolute error:  ${results['implicit_absolute_error']:.4f}\n")
        f.write(f"  Relative error:  {results['implicit_relative_error']:.2f}%\n\n")

        # Performance comparison
        f.write("Performance Comparison:\n")
        f.write("----------------------\n")
        f.write(
            f"Analytical method time:     {results['analytical_time']:.6f} seconds\n"
        )
        f.write(f"Explicit method time:       {results['explicit_time']:.6f} seconds\n")
        f.write(f"Implicit method time:       {results['implicit_time']:.6f} seconds\n")
        f.write(f"Speed ratios (vs analytical):\n")
        f.write(
            f"  Explicit/Analytical:      {results['explicit_time']/results['analytical_time']:.1f}x\n"
        )
        f.write(
            f"  Implicit/Analytical:      {results['implicit_time']/results['analytical_time']:.1f}x\n"
        )
        f.write(
            f"  Implicit/Explicit:        {results['implicit_time']/results['explicit_time']:.1f}x\n\n"
        )

        # Analysis
        f.write("Analysis:\n")
        f.write("--------\n")
        f.write(
            f"The explicit finite difference method achieves {100-results['explicit_relative_error']:.2f}% accuracy\n"
        )
        f.write(
            f"The implicit finite difference method achieves {100-results['implicit_relative_error']:.2f}% accuracy\n"
        )
        f.write(
            f"relative to the analytical solution with a grid of {M}×{N} points.\n\n"
        )

        f.write("Method Characteristics:\n")
        f.write("----------------------\n")
        f.write("Analytical method:\n")
        f.write("  + Exact solution (up to numerical precision)\n")
        f.write("  + Fastest computation\n")
        f.write("  - Only works for standard European options\n\n")

        f.write("Explicit finite difference:\n")
        f.write("  + Simple implementation\n")
        f.write("  + Direct time stepping\n")
        f.write("  - Stability constraints on time step\n")
        f.write("  - Requires small time steps for stability\n\n")

        f.write("Implicit finite difference:\n")
        f.write("  + Unconditionally stable\n")
        f.write("  + Can use larger time steps\n")
        f.write("  - Requires solving linear system at each step\n")
        f.write("  - More complex implementation\n")
        f.write("  + Similar accuracy to explicit method\n")

    return report_path


if __name__ == "__main__":
    # Parameters from w303 example
    S0 = 100  # Current stock price
    K = 100  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate (5% per annum)
    sigma = 0.2  # Volatility (20% per annum)

    # Grid parameters for explicit method
    S_max = 200  # Maximum stock price
    M = 100  # Number of stock price steps
    N = 1000  # Number of time steps

    # Generate detailed report
    write_comparison_report(S0, K, T, r, sigma, S_max, M, N)
