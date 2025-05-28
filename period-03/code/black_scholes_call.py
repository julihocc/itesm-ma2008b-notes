#!/usr/bin/env python3
"""Black-Scholes analytical function"""

import math
from scipy.stats import norm

def black_scholes_call(S0, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (math.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S0 * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)

if __name__ == "__main__":
    # Parameters
    S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    
    result = black_scholes_call(S0, K, T, r, sigma)
    print(f"Black-Scholes: {result:.2f}")