#!/usr/bin/env python3
"""
Black-Scholes Explicit Finite Difference Method
===============================================

This script calculates the European call option price using the explicit finite
difference method to solve the Black-Scholes PDE based on the example from w303.

Example from w303:
- S₀ = 100, K = 100, T = 0.25, r = 0.05, σ = 0.2

Author: Based on w303 example
"""

import math
import os
import numpy as np


def black_scholes_call_explicit(S0, K, T, r, sigma, S_max=200, M=100, N=1000):
    """
    Calculate the Black-Scholes price for a European call option using explicit finite difference.

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
        Maximum stock price in the grid (default: 200)
    M : int, optional
        Number of stock price steps (default: 100)
    N : int, optional
        Number of time steps (default: 1000)

    Returns:
    --------
    float: Black-Scholes call option price (explicit finite difference solution)
    """

    # Grid setup
    dS = S_max / M
    dt = T / N

    # Stock price grid
    S = np.linspace(0, S_max, M + 1)

    # Initialize option value matrix
    V = np.zeros((M + 1, N + 1))

    # Boundary conditions at expiration (t = T)
    V[:, N] = np.maximum(S - K, 0)  # Call option payoff

    # Time stepping (explicit scheme)
    for i in range(N - 1, -1, -1):  # Time index (backward in time)
        # Boundary conditions for current time step
        V[0, i] = 0  # V(0, t) = 0 for call option
        V[M, i] = S_max - K * np.exp(-r * (T - i * dt))  # V(S_max, t)

        # Interior points
        for j in range(1, M):  # Stock price index (skip boundaries)
            # Calculate coefficients for the explicit scheme
            # PDE: dV/dt + 0.5*σ²*S²*d²V/dS² + r*S*dV/dS - r*V = 0

            # Finite difference approximations
            dV_dS = (V[j + 1, i + 1] - V[j - 1, i + 1]) / (2 * dS)
            d2V_dS2 = (V[j + 1, i + 1] - 2 * V[j, i + 1] + V[j - 1, i + 1]) / (dS**2)

            # Explicit finite difference formula
            V[j, i] = V[j, i + 1] + dt * (
                0.5 * sigma**2 * S[j] ** 2 * d2V_dS2
                + r * S[j] * dV_dS
                - r * V[j, i + 1]
            )

    # Interpolate to find option value at S0
    i_S0 = int(S0 / dS)
    if i_S0 < M and S0 > 0:
        # Linear interpolation
        weight = (S0 - S[i_S0]) / dS
        option_price = (1 - weight) * V[i_S0, 0] + weight * V[i_S0 + 1, 0]
    else:
        option_price = V[i_S0, 0]

    return option_price


def write_black_scholes_explicit_report(
    S0,
    K,
    T,
    r,
    sigma,
    S_max=200,
    M=100,
    N=1000,
    filename="black_scholes_explicit_report.txt",
):
    """
    Write complete Black-Scholes explicit finite difference analysis to a file.

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
        Maximum stock price in the grid (default: 200)
    M : int, optional
        Number of stock price steps (default: 100)
    N : int, optional
        Number of time steps (default: 1000)
    filename : str, optional
        Output filename (default: "black_scholes_explicit_report.txt")

    Returns:
    --------
    str: Path to the generated report file
    """

    # Calculate option price
    option_price = black_scholes_call_explicit(S0, K, T, r, sigma, S_max, M, N)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, filename)

    # Write report to file
    with open(report_path, "w") as f:
        f.write("Black-Scholes Explicit Finite Difference Solution\n")
        f.write("=================================================\n")
        f.write(f"Parameters: S₀=${S0}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}\n")
        f.write(f"Grid: S_max={S_max}, M={M}, N={N}\n")
        f.write(f"Grid spacing: dS={S_max/M:.2f}, dt={T/N:.6f}\n")
        f.write("\n")
        f.write(f"European call option value = ${option_price:.2f}\n")

    print(f"Report written to: {report_path}")
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

    # Run the complete analysis
    write_black_scholes_explicit_report(S0, K, T, r, sigma, S_max, M, N)
