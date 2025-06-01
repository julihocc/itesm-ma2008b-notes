#!/usr/bin/env python3
"""Comparison of option pricing methods"""

from typing import Dict, Final
import numpy as np
from black_scholes_call import black_scholes_call
from explicit_fd_call import explicit_fd_call


def compare_methods(
    S0: float, K: float, T: float, r: float, sigma: float
) -> Dict[str, float]:
    """Compare Black-Scholes and Explicit FD methods.

    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'black_scholes': Black-Scholes price
            - 'explicit_fd': Explicit Finite Difference price
            - 'error': Absolute error
            - 'relative_error': Relative error (%)
    """
    # Fixed parameters for finite difference method
    S_max: Final[float] = 200.0
    M: Final[int] = 20
    N: Final[int] = 25

    # Calculate prices using both methods
    bs_price: float = black_scholes_call(S0, K, T, r, sigma)
    fd_result: np.ndarray = explicit_fd_call(S0, K, T, r, sigma, S_max, M, N)
    fd_price: float = float(fd_result[int(M * S0 / S_max), 0])

    # Calculate errors
    error: float = abs(fd_price - bs_price)
    rel_error: float = error / bs_price * 100

    return {
        "black_scholes": bs_price,
        "explicit_fd": fd_price,
        "error": error,
        "relative_error": rel_error,
    }


if __name__ == "__main__":
    # Parameters
    S0: Final[float] = 100.0
    K: Final[float] = 100.0
    T: Final[float] = 0.25  # 3 months
    r: Final[float] = 0.05  # 5%
    sigma: Final[float] = 0.2  # 20%

    results: Dict[str, float] = compare_methods(S0, K, T, r, sigma)

    print(f"Black-Scholes: {results['black_scholes']:.2f}")
    print(f"Explicit FD:   {results['explicit_fd']:.2f}")
    print(f"Error:         {results['error']:.2f}")
    print(f"Relative:      {results['relative_error']:.1f}%")
