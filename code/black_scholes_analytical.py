#!/usr/bin/env python3
"""
Simple Black-Scholes Analytical Solution
========================================

This script calculates the exact analytical solution for a European call option
using the Black-Scholes formula based on the example from w303.

Example from w303:
- S₀ = 100, K = 100, T = 0.25, r = 0.05, σ = 0.2
- Expected result: V = $4.61

Author: Based on w303 example
"""

import math
from scipy.stats import norm


def black_scholes_call_analytical(S0, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option using the analytical formula.

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

    Returns:
    --------
    float: Black-Scholes call option price (analytical solution)
    """

    # Step 1: Calculate d1
    ln_term = math.log(S0 / K)
    drift_term = (r + 0.5 * sigma**2) * T
    vol_term = sigma * math.sqrt(T)
    d1 = (ln_term + drift_term) / vol_term

    # Step 2: Calculate d2
    d2 = d1 - vol_term

    # Step 3: Calculate normal distribution values
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    # Step 4: Apply Black-Scholes formula
    discount_factor = math.exp(-r * T)
    first_term = S0 * N_d1
    second_term = K * discount_factor * N_d2
    option_price = first_term - second_term

    return option_price


def print_black_scholes_analysis(S0, K, T, r, sigma):
    """
    Print complete Black-Scholes analysis.

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
    """

    print("Black-Scholes Solution")
    print("=====================")
    print(f"Parameters: S₀=${S0}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
    print()

    # Calculate option price
    option_price = black_scholes_call_analytical(S0, K, T, r, sigma)

    print(f"European call option value = ${option_price:.2f}")


# Parameters from w303 example
S0 = 100  # Current stock price
K = 100  # Strike price
T = 0.25  # Time to expiration (3 months)
r = 0.05  # Risk-free rate (5% per annum)
sigma = 0.2  # Volatility (20% per annum)

# Run the complete analysis
print_black_scholes_analysis(S0, K, T, r, sigma)
