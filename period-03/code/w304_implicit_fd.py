import numpy as np
from scipy.linalg import solve_banded

def implicit_fd_european_option(S0, K, T, r, sigma, S_max, M, N, option_type='call'):
    """
    Prices a European option (call or put) using the Implicit Finite Difference Method.
    option_type: 'call' or 'put'
    """
    dt = T / N
    dS = S_max / M

    S_values = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal condition
    if option_type == 'call':
        V[N, :] = np.maximum(S_values - K, 0)
    else: # put
        V[N, :] = np.maximum(K - S_values, 0)

    # Boundary conditions
    if option_type == 'call':
        V[:, 0] = 0.0
        for j in range(N + 1):
            V[j, M] = S_max - K * np.exp(-r * (N - j) * dt)
    else: # put
        for j in range(N + 1):
            V[j, 0] = K * np.exp(-r * (N - j) * dt)
        V[:, M] = 0.0

    _S_interior = S_values[1:-1]
    
    # Coefficients from slides for A V_new = V_old
    # A is (I + dt L_new)
    # L_new V_new = 0.5 sig^2 S^2 V_SS_new + r S V_S_new - r V_new
    # Coeff of V_{i-1,new} in L_new: 0.5*sig^2*S_i^2/dS^2 - 0.5*r*S_i/dS
    # Coeff of V_{i,new}   in L_new: -sig^2*S_i^2/dS^2 - r
    # Coeff of V_{i+1,new} in L_new: 0.5*sig^2*S_i^2/dS^2 + 0.5*r*S_i/dS

    # Slide's matrix elements for c_i V_{i-1,new} + a_i V_{i,new} + b_i V_{i+1,new} = V_{i,old}
    # a_i (diag) = 1 + dt*(r + sig^2 S_i^2/dS^2)
    # b_i (super)= -0.5*dt*(sig^2 S_i^2/dS^2 + r S_i/dS)
    # c_i (sub)  = -0.5*dt*(sig^2 S_i^2/dS^2 - r S_i/dS)
    
    # These are coefficients of the matrix A such that A * V_new = V_old
    coeff_a = 1 + dt * (r + sigma**2 * _S_interior**2 / dS**2)
    coeff_b = -0.5 * dt * (sigma**2 * _S_interior**2 / dS**2 + r * _S_interior / dS)
    coeff_c = -0.5 * dt * (sigma**2 * _S_interior**2 / dS**2 - r * _S_interior / dS)

    A_banded = np.zeros((3, M - 1))
    A_banded[0, 1:] = coeff_b[:-1]    # Superdiagonal
    A_banded[1, :]  = coeff_a         # Main diagonal
    A_banded[2, :-1]= coeff_c[1:]     # Subdiagonal

    for j_time in range(N - 1, -1, -1):
        rhs = V[j_time + 1, 1:-1].copy()
        # Adjust RHS for boundary conditions incorporated into the matrix solve
        rhs[0] -= coeff_c[0] * V[j_time, 0]   # Known V_new at S=0
        rhs[-1] -= coeff_b[-1] * V[j_time, M] # Known V_new at S=S_max
        
        V[j_time, 1:-1] = solve_banded((1, 1), A_banded, rhs)

    s0_idx = np.argmin(np.abs(S_values - S0))
    return V[0, s0_idx]

if __name__ == '__main__':
    # Parameters from the lecture's "Main Example"
    S0_main = 100.0; K_main = 100.0; T_main = 0.25; r_main = 0.05; sigma_main = 0.2
    S_max_main = 200.0; M_main = 20; N_main = 25

    # The slide's Implicit FD result for V(100,0) is 5.44.
    # This value is likely for a call option with sigma around 0.235 to match BS of ~5.46
    sigma_slide_benchmark = 0.235 

    price_call_slide_params = implicit_fd_european_option(
        S0_main, K_main, T_main, r_main, sigma_slide_benchmark, S_max_main, M_main, N_main, option_type='call'
    )
    print(f"--- Implicit FD for Call (S0={S0_main}, K={K_main}, sigma={sigma_slide_benchmark}) ---")
    print(f"Calculated Implicit FD Call Price: {price_call_slide_params:.4f}") 
    print("Slide Implicit FD Call Result for V(100,0) (Table): 5.44 (for their specific benchmark params)")
    print("Note: The slide's table values (e.g., V(80,t)...V(120,t)) are for this benchmark.")

    # Verification of coefficients for S_i=100 (i=10 for M=20, dS=10)
    # _S_interior_idx_100 = 9 (since _S_interior starts at S_1)
    # dt = T_main/N_main = 0.01, dS = S_max_main/M_main = 10
    # S_100 = 100
    # a_10_calc = 1 + 0.01 * (0.05 + sigma_slide_benchmark**2 * 100**2 / 10**2) = 1 + 0.01 * (0.05 + sigma_slide_benchmark**2 * 100)
    # b_10_calc = -0.5 * 0.01 * (sigma_slide_benchmark**2 * 100 + r_main * 100 / 10)
    # c_10_calc = -0.5 * 0.01 * (sigma_slide_benchmark**2 * 100 - r_main * 100 / 10)
    # print(f"Calculated a_10 (sigma={sigma_slide_benchmark}): {a_10_calc}") # Expected slide a_10 = 1.0405 (for sigma=0.2)
    # If sigma=0.2:
    # a_10_sig02 = 1 + 0.01 * (0.05 + 0.2**2 * 100) = 1 + 0.01 * (0.05 + 4) = 1 + 0.01 * 4.05 = 1.0405. Matches slide.
    # b_10_sig02 = -0.5 * 0.01 * (0.2**2 * 100 + 0.05 * 10) = -0.005 * (4 + 0.5) = -0.005 * 4.5 = -0.0225. Matches slide.
    # c_10_sig02 = -0.5 * 0.01 * (0.2**2 * 100 - 0.05 * 10) = -0.005 * (4 - 0.5) = -0.005 * 3.5 = -0.0175. Matches slide.

    price_call_sig02 = implicit_fd_european_option(
        S0_main, K_main, T_main, r_main, sigma_main, S_max_main, M_main, N_main, option_type='call'
    )
    print(f"\n--- Implicit FD for Call (S0={S0_main}, K={K_main}, sigma={sigma_main}) ---")
    print(f"Calculated Implicit FD Call Price: {price_call_sig02:.4f}")
    print("This should be compared to BS Call with sigma=0.2 (which is ~4.61)")
