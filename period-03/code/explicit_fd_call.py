#!/usr/bin/env python3
"""Explicit Finite Difference function"""

import numpy as np

def explicit_fd_call(S0, K, T, r, sigma, S_max, M, N):
    """Calculate call option price using explicit finite difference"""
    dS, dt = S_max / M, T / N
    V = np.zeros((M+1, N+1))
    
    # Terminal condition: V(S, T) = max(S - K, 0)
    for i in range(M+1):
        V[i, N] = max(i * dS - K, 0)
    
    # Boundary conditions
    for j in range(N+1):
        t = j * dt
        V[0, j] = 0
        V[M, j] = S_max - K * np.exp(-r * (T - t))
    
    # Explicit scheme
    for j in range(N-1, -1, -1):
        for i in range(1, M):
            a = 0.5 * sigma**2 * (i * dS)**2
            b = r * (i * dS)
            c = r
            
            alpha = dt * (a / (dS**2) - b / (2 * dS))
            beta = 1 - dt * (2 * a / (dS**2) + c)
            gamma = dt * (a / (dS**2) + b / (2 * dS))
            
            V[i, j] = alpha * V[i-1, j+1] + beta * V[i, j+1] + gamma * V[i+1, j+1]
            V[i, j] = max(V[i, j], 0)
    
    # Return V(S0, 0)
    # idx = int(S0 / dS)
    return V

if __name__ == "__main__":
    # Parameters
    S0, K, T, r, sigma, S_max, M, N = 100, 100, 0.25, 0.05, 0.2, 200, 20, 25
    
    V = explicit_fd_call(S0, K, T, r, sigma, S_max, M, N)
    # Print the matrix V for debugging
    np.set_printoptions(precision=2, suppress=True)
    print("Matrix V:", V)
    # Get the value at S0 at time 0
    result = V[ int( M * S0 / S_max ), 0]
    # Print the result
    print(f"Explicit FD: {result:.2f}")