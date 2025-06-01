#!/usr/bin/env python3
"""Explicit Finite Difference function"""

import numpy as np
from numpy.typing import NDArray


def explicit_fd_call(
    S0: float, K: float, T: float, r: float, sigma: float, S_max: float, M: int, N: int
) -> NDArray[np.float64]:
    """Calculate call option price using explicit finite difference

    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        S_max: Maximum stock price
        M: Number of spatial steps
        N: Number of time steps

    Returns:
        NDArray[np.float64]: Matrix of option values
    """
    dS: float = S_max / M
    dt: float = T / N
    V: NDArray[np.float64] = np.zeros((M + 1, N + 1), dtype=np.float64)

    # Terminal condition: V(S, T) = max(S - K, 0)
    for i in range(M + 1):
        V[i, N] = max(i * dS - K, 0.0)

    # Boundary conditions
    for j in range(N + 1):
        t: float = j * dt
        V[0, j] = 0.0
        V[M, j] = S_max - K * np.exp(-r * (T - t))

    # Explicit scheme
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            a: float = 0.5 * sigma**2 * (i * dS) ** 2
            b: float = r * (i * dS)
            c: float = r

            alpha: float = dt * (a / (dS**2) - b / (2 * dS))
            beta: float = 1.0 - dt * (2 * a / (dS**2) + c)
            gamma: float = dt * (a / (dS**2) + b / (2 * dS))

            V[i, j] = (
                alpha * V[i - 1, j + 1] + beta * V[i, j + 1] + gamma * V[i + 1, j + 1]
            )
            V[i, j] = max(V[i, j], 0.0)

    return V


if __name__ == "__main__":
    # Parameters
    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25
    r: float = 0.05
    sigma: float = 0.2
    S_max: float = 200.0
    M: int = 20
    N: int = 25

    V: NDArray[np.float64] = explicit_fd_call(S0, K, T, r, sigma, S_max, M, N)
    # Print the matrix V for debugging
    np.set_printoptions(precision=2, suppress=True)
    print("Matrix V:", V)
    # Get the value at S0 at time 0
    result: float = float(V[int(M * S0 / S_max), 0])
    # Print the result
    print(f"Explicit FD: {result:.2f}")
