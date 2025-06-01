#!/usr/bin/env python3
"""Black-Scholes analytical function for European call option pricing."""

import math
from typing import Final
from scipy.stats import norm


def black_scholes_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes price for a European call option.

    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)

    Returns:
        float: Call option price
    """
    d1: float = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2: float = d1 - sigma * math.sqrt(T)

    # Calculate call option price using Black-Scholes formula
    call_price: float = float(S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))

    return call_price


if __name__ == "__main__":
    # Parameters
    S0: Final[float] = 100.0
    K: Final[float] = 100.0
    T: Final[float] = 0.25  # 3 months
    r: Final[float] = 0.05  # 5%
    sigma: Final[float] = 0.2  # 20%

    result: float = black_scholes_call(S0, K, T, r, sigma)
    print(f"Black-Scholes: {result:.2f}")
