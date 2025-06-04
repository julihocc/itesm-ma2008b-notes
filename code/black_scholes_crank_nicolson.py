import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time


def black_scholes_call_crank_nicolson(S0, K, T, r, sigma, S_max=200, M=100, N=1000):
    """
    Solve Black-Scholes equation using Crank-Nicolson finite difference method.

    Parameters:
    S0 : float - Current stock price
    K : float - Strike price
    T : float - Time to maturity
    r : float - Risk-free rate
    sigma : float - Volatility
    S_max : float - Maximum stock price in grid
    M : int - Number of stock price steps
    N : int - Number of time steps

    Returns:
    float - Option value at (S0, 0)
    """
    # Grid setup
    dS = S_max / M
    dt = T / N

    # Stock price grid (from 0 to S_max)
    S = np.linspace(0, S_max, M + 1)

    # Initialize option value vector
    V = np.maximum(S - K, 0)  # Terminal condition at t = T

    # Build coefficient matrices for interior points (excluding boundaries)
    # Interior points are i = 1, 2, ..., M-1
    n_interior = M - 1

    # Initialize matrices A and B (both tridiagonal)
    # A is the left-hand side matrix (I - dt/2 * L)
    # B is the right-hand side matrix (I + dt/2 * L)
    A_diag_main = np.zeros(n_interior)
    A_diag_upper = np.zeros(n_interior - 1)
    A_diag_lower = np.zeros(n_interior - 1)

    B_diag_main = np.zeros(n_interior)
    B_diag_upper = np.zeros(n_interior - 1)
    B_diag_lower = np.zeros(n_interior - 1)

    # Fill matrix coefficients for interior points
    for i in range(1, M):  # Interior points
        Si = S[i]
        idx = i - 1  # Index in the matrix (0-based for interior points)

        # Coefficients from the Crank-Nicolson scheme
        # L operator coefficients
        alpha = 0.5 * sigma**2 * Si**2 / (dS**2)
        beta = r * Si / (2 * dS)
        gamma = r

        # Left-hand side matrix A = I - dt/2 * L
        A_diag_main[idx] = 1 + dt / 2 * (2 * alpha + gamma)

        if idx < n_interior - 1:
            A_diag_upper[idx] = -dt / 2 * (alpha + beta)

        if idx > 0:
            A_diag_lower[idx - 1] = -dt / 2 * (alpha - beta)

        # Right-hand side matrix B = I + dt/2 * L
        B_diag_main[idx] = 1 - dt / 2 * (2 * alpha + gamma)

        if idx < n_interior - 1:
            B_diag_upper[idx] = dt / 2 * (alpha + beta)

        if idx > 0:
            B_diag_lower[idx - 1] = dt / 2 * (alpha - beta)

    # Create the tridiagonal matrices
    A = sp.diags(
        [A_diag_lower, A_diag_main, A_diag_upper],
        [-1, 0, 1],
        shape=(n_interior, n_interior),
        format="csc",
    )

    B = sp.diags(
        [B_diag_lower, B_diag_main, B_diag_upper],
        [-1, 0, 1],
        shape=(n_interior, n_interior),
        format="csc",
    )

    # Time stepping (backward from T to 0)
    for j in range(N):
        # Current time for boundary condition
        current_time = T - (j + 1) * dt

        # Boundary condition at S = S_max (for call option, linear growth)
        V_boundary = (
            S_max - K * np.exp(-r * current_time) if current_time > 0 else S_max - K
        )

        # Right-hand side calculation: B * V_interior
        rhs = B.dot(V[1:M])

        # Adjust RHS for boundary conditions
        # At S=0: V(0,t) = 0 (no adjustment needed for first interior point)

        # Adjust for upper boundary condition
        Si_last = S[M - 1]
        alpha_last = 0.5 * sigma**2 * Si_last**2 / (dS**2)
        beta_last = r * Si_last / (2 * dS)

        # Adjust RHS for boundary at current time step
        rhs[-1] += dt / 2 * (alpha_last + beta_last) * V_boundary

        # Adjust LHS for boundary at next time step
        next_time = T - (j + 2) * dt if j < N - 1 else 0
        V_boundary_next = (
            S_max - K * np.exp(-r * next_time) if next_time > 0 else S_max - K
        )
        rhs[-1] += dt / 2 * (alpha_last + beta_last) * V_boundary_next

        # Solve the linear system A * V_new = rhs
        V_interior = spla.spsolve(A, rhs)

        # Update the solution vector
        V[0] = 0  # Boundary condition at S = 0
        V[1:M] = V_interior
        V[M] = V_boundary_next  # Boundary condition at S = S_max

    # Interpolate to find V(S0, 0)
    if S0 <= 0:
        return 0
    elif S0 >= S_max:
        return S0 - K
    else:
        # Linear interpolation
        i = int(S0 / dS)
        if i >= M:
            i = M - 1

        if i < M - 1:
            weight = (S0 - S[i]) / dS
            option_value = V[i] * (1 - weight) + V[i + 1] * weight
        else:
            option_value = V[i]

        return option_value


def write_black_scholes_crank_nicolson_report(
    S0,
    K,
    T,
    r,
    sigma,
    S_max=200,
    M=100,
    N=1000,
    filename="black_scholes_crank_nicolson_report.txt",
):
    """
    Calculate Black-Scholes option value using Crank-Nicolson method and write detailed report.

    Parameters:
    S0 : float - Current stock price
    K : float - Strike price
    T : float - Time to maturity
    r : float - Risk-free rate
    sigma : float - Volatility
    S_max : float - Maximum stock price in grid
    M : int - Number of stock price steps
    N : int - Number of time steps
    filename : str - Output filename
    """
    import os

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    # Calculate option value and measure computation time
    start_time = time.time()
    option_value = black_scholes_call_crank_nicolson(S0, K, T, r, sigma, S_max, M, N)
    computation_time = time.time() - start_time

    # Write comprehensive report
    with open(filepath, "w") as f:
        f.write("BLACK-SCHOLES CRANK-NICOLSON FINITE DIFFERENCE METHOD REPORT\n")
        f.write("=" * 65 + "\n\n")

        f.write("METHOD DESCRIPTION:\n")
        f.write("- Uses Crank-Nicolson finite difference scheme\n")
        f.write("- Time-centered average of explicit and implicit schemes\n")
        f.write("- Unconditionally stable (no restrictions on time step)\n")
        f.write("- Second-order accurate in both time and space\n")
        f.write("- Most commonly used method in practice\n\n")

        f.write("INPUT PARAMETERS:\n")
        f.write(f"- Current stock price (S0): ${S0:.2f}\n")
        f.write(f"- Strike price (K): ${K:.2f}\n")
        f.write(f"- Time to maturity (T): {T:.2f} years\n")
        f.write(f"- Risk-free rate (r): {r:.1%}\n")
        f.write(f"- Volatility (σ): {sigma:.1%}\n\n")

        f.write("NUMERICAL PARAMETERS:\n")
        f.write(f"- Maximum stock price: ${S_max:.2f}\n")
        f.write(f"- Number of stock price steps: {M}\n")
        f.write(f"- Number of time steps: {N}\n")
        f.write(f"- Stock price step size (ΔS): ${S_max/M:.2f}\n")
        f.write(f"- Time step size (Δt): {T/N:.6f} years\n\n")

        f.write("COMPUTATIONAL DETAILS:\n")
        f.write("- Matrix system: Two tridiagonal matrices (A and B)\n")
        f.write("- Left-hand side: (I - Δt/2·L)·V^{j+1}\n")
        f.write("- Right-hand side: (I + Δt/2·L)·V^j\n")
        f.write("- Boundary conditions:\n")
        f.write("  * At S=0: V(0,t) = 0 (call worthless)\n")
        f.write("  * At S=S_max: Linear extrapolation\n")
        f.write("- Time integration: Backward from expiry to present\n")
        f.write(
            "- Spatial operator L includes diffusion, convection, and reaction terms\n\n"
        )

        f.write("RESULTS:\n")
        f.write(f"- Black-Scholes call option value: ${option_value:.4f}\n")
        f.write(f"- Computation time: {computation_time:.6f} seconds\n\n")

        f.write("METHOD CHARACTERISTICS:\n")
        f.write("ADVANTAGES:\n")
        f.write("- Second-order accuracy in both time and space\n")
        f.write("- Unconditionally stable for any time step size\n")
        f.write("- Optimal balance between accuracy and stability\n")
        f.write("- Industry standard for parabolic PDEs\n")
        f.write("- No oscillations or artificial damping\n\n")

        f.write("DISADVANTAGES:\n")
        f.write("- Requires solving linear system at each time step\n")
        f.write("- More complex implementation than explicit method\n")
        f.write("- Requires two matrix operations per time step\n")
        f.write("- Higher memory requirements for matrix storage\n\n")

        f.write("CONVERGENCE NOTES:\n")
        f.write("- Error decreases as O(Δt²) + O(ΔS²)\n")
        f.write("- Optimal convergence rate among finite difference methods\n")
        f.write("- Time step can be chosen based on accuracy needs only\n")
        f.write("- Spatial grid refinement improves accuracy quadratically\n")
        f.write("- Most accurate method for given grid resolution\n\n")

        f.write("COMPARISON WITH OTHER METHODS:\n")
        f.write("- More accurate than explicit or implicit methods\n")
        f.write("- Computational cost between explicit and implicit\n")
        f.write("- Best choice for production applications\n")
        f.write("- Standard in financial software packages\n\n")


if __name__ == "__main__":
    # Test parameters (same as other methods)
    S0 = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 0.25  # Time to maturity (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)

    # Generate the report
    write_black_scholes_crank_nicolson_report(S0, K, T, r, sigma)
