#!/usr/bin/env python3
"""Comparison of option pricing methods"""

from black_scholes_call import black_scholes_call
from explicit_fd_call import explicit_fd_call

def compare_methods(S0, K, T, r, sigma):
    """Compare Black-Scholes and Explicit FD methods"""
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    fd_price = explicit_fd_call(S0, K, T, r, sigma, 200, 20, 25)[int(20 * S0 / 200), 0]
    
    error = abs(fd_price - bs_price)
    rel_error = error / bs_price * 100
    
    return {
        'black_scholes': bs_price,
        'explicit_fd': fd_price,
        'error': error,
        'relative_error': rel_error
    }

if __name__ == "__main__":
    # Parameters
    S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2
    
    results = compare_methods(S0, K, T, r, sigma)
    
    print(f"Black-Scholes: {results['black_scholes']:.2f}")
    print(f"Explicit FD:   {results['explicit_fd']:.2f}")
    print(f"Error:         {results['error']:.2f}")
    print(f"Relative:      {results['relative_error']:.1f}%")