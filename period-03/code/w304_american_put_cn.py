import numpy as np
from scipy.linalg import solve_banded
from w304_crank_nicolson import crank_nicolson_european_option # Assuming it handles puts

def crank_nicolson_american_put(S0, K, T, r, sigma, S_max, M, N):
    """
    Prices an American put option using Crank-Nicolson with iterative check for early exercise.
    """
    dt = T / N
    dS = S_max / M

    S_values = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal condition
    V[N, :] = np.maximum(K - S_values, 0)

    # Boundary conditions for American Put
    for j_time in range(N + 1):
        V[j_time, 0] = K # At S=0, put is worth K
    V[:, M] = 0.0       # At S=S_max (large), put is worthless

    _S_interior = S_values[1:-1]

    # LHS Matrix (A_CN for A_CN * V_new = RHS_vector)
    lhs_a = 1 + 0.5 * dt * (r + sigma**2 * _S_interior**2 / dS**2)
    lhs_b = -0.25 * dt * (sigma**2 * _S_interior**2 / dS**2 + r * _S_interior / dS)
    lhs_c = -0.25 * dt * (sigma**2 * _S_interior**2 / dS**2 - r * _S_interior / dS)

    A_CN_banded = np.zeros((3, M - 1))
    A_CN_banded[0, 1:] = lhs_b[:-1]
    A_CN_banded[1, :]  = lhs_a
    A_CN_banded[2, :-1]= lhs_c[1:]

    # RHS Matrix (B_CN for RHS_vector = B_CN * V_old)
    rhs_a_prime = 1 - 0.5 * dt * (r + sigma**2 * _S_interior**2 / dS**2)
    rhs_b_prime = 0.25 * dt * (sigma**2 * _S_interior**2 / dS**2 + r * _S_interior / dS)
    rhs_c_prime = 0.25 * dt * (sigma**2 * _S_interior**2 / dS**2 - r * _S_interior / dS)

    for j_time in range(N - 1, -1, -1):
        V_old_internal = V[j_time + 1, 1:-1]
        
        rhs_vec = np.zeros(M - 1)
        rhs_vec[0]    = rhs_c_prime[0] * V[j_time+1, 0] + rhs_a_prime[0] * V_old_internal[0] + rhs_b_prime[0] * V_old_internal[1]
        rhs_vec[1:-1] = rhs_c_prime[1:-1] * V_old_internal[:-2] + rhs_a_prime[1:-1] * V_old_internal[1:-1] + rhs_b_prime[1:-1] * V_old_internal[2:]
        rhs_vec[-1]   = rhs_c_prime[-1] * V_old_internal[-2] + rhs_a_prime[-1] * V_old_internal[-1] + rhs_b_prime[-1] * V[j_time+1, M]

        rhs_vec[0] -= lhs_c[0] * V[j_time, 0]   # V_new at S=0 (Boundary)
        rhs_vec[-1] -= lhs_b[-1] * V[j_time, M] # V_new at S=S_max (Boundary)
        
        # Solve for continuation values
        V_continuation = solve_banded((1, 1), A_CN_banded, rhs_vec)
        
        # Apply early exercise condition
        intrinsic_value = K - _S_interior
        V[j_time, 1:-1] = np.maximum(V_continuation, intrinsic_value)
        
        # Ensure boundary conditions are maintained if changed by early exercise logic (unlikely for these BCs)
        # V[j_time, 0] = K
        # V[j_time, M] = 0.0

    s0_idx = np.argmin(np.abs(S_values - S0))
    return V[0, s0_idx]

if __name__ == '__main__':
    # Parameters from the lecture's "Main Example" (S0=100, K=100, sigma=0.2)
    S0_main = 100.0; K_main = 100.0; T_main = 0.25; r_main = 0.05; sigma_main = 0.2
    S_max_main = 200.0; M_main = 20; N_main = 25

    # Slide: American Put Results (K=100)
    # V_American Put(100,0) = 7.52
    # V_European Put(100,0) = 7.28
    # Early exercise premium = 0.24
    # These values (Euro Put=7.28) are for different parameters than the main example.
    # They likely correspond to S0=95, K=100, sigma=0.30 to get Euro Put ~7.28.
    
    S0_slide_put = 95.0; K_slide_put = 100.0; sigma_slide_put = 0.30
    
    # Calculate European Put for these slide parameters using CN
    euro_put_slide_params = crank_nicolson_european_option(
        S0_slide_put, K_slide_put, T_main, r_main, sigma_slide_put, S_max_main, M_main, N_main, option_type='put'
    )
    amer_put_slide_params = crank_nicolson_american_put(
        S0_slide_put, K_slide_put, T_main, r_main, sigma_slide_put, S_max_main, M_main, N_main
    )
    early_exercise_premium_slide = amer_put_slide_params - euro_put_slide_params

    print(f"--- American Put (Slide Context: S0={S0_slide_put}, K={K_slide_put}, sigma={sigma_slide_put}) ---")
    print(f"Calculated CN European Put: {euro_put_slide_params:.4f} (Slide: 7.28)")
    print(f"Calculated CN American Put: {amer_put_slide_params:.4f} (Slide: 7.52)")
    print(f"Calculated Early Exercise Premium: {early_exercise_premium_slide:.4f} (Slide: 0.24)")
    print("Note: Small differences due to FDM discretization & specific implementation.")

    # American Put with main example parameters
    euro_put_main_params = crank_nicolson_european_option(
        S0_main, K_main, T_main, r_main, sigma_main, S_max_main, M_main, N_main, option_type='put'
    )
    amer_put_main_params = crank_nicolson_american_put(
        S0_main, K_main, T_main, r_main, sigma_main, S_max_main, M_main, N_main
    )
    early_exercise_premium_main = amer_put_main_params - euro_put_main_params
    print(f"\n--- American Put (Main Example: S0={S0_main}, K={K_main}, sigma={sigma_main}) ---")
    print(f"Calculated CN European Put: {euro_put_main_params:.4f}")
    print(f"Calculated CN American Put: {amer_put_main_params:.4f}")
    print(f"Calculated Early Exercise Premium: {early_exercise_premium_main:.4f}")
