import numpy as np
from scipy.stats import norm


def black_scholes_european_call(S0, K, T, r, sigma):
    """
    Calculates Black-Scholes price for a European call option.
    """
    if T == 0:  # At expiry
        return max(0, S0 - K)
    if sigma == 0 or S0 == 0:  # No volatility or no stock price
        return max(0, S0 - K * np.exp(-r * T))

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def black_scholes_european_put(S0, K, T, r, sigma):
    """
    Calculates Black-Scholes price for a European put option.
    """
    if T == 0:  # At expiry
        return max(0, K - S0)
    if sigma == 0 or S0 == 0:  # No volatility or no stock price
        return max(0, K * np.exp(-r * T) - S0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price


if __name__ == "__main__":
    # Parameters from the lecture's "Main Example"
    S0_main = 100.0
    K_main = 100.0
    T_main = 0.25
    r_main = 0.05
    sigma_main = 0.2

    call_price_main = black_scholes_european_call(
        S0_main, K_main, T_main, r_main, sigma_main
    )
    put_price_main = black_scholes_european_put(
        S0_main, K_main, T_main, r_main, sigma_main
    )

    print(
        f"--- Main Example Parameters (S0={S0_main}, K={K_main}, sigma={sigma_main}) ---"
    )
    print(f"Black-Scholes European Call Price: {call_price_main:.4f}")
    print(f"Black-Scholes European Put Price: {put_price_main:.4f}")
    print("\n")

    # Parameters that would yield a Call price around 5.46 (slide's FDM benchmark)
    sigma_alt = 0.235
    call_price_alt_sigma = black_scholes_european_call(
        S0_main, K_main, T_main, r_main, sigma_alt
    )
    print(
        f"--- Parameters for Call ~5.46 (S0={S0_main}, K={K_main}, sigma={sigma_alt}) ---"
    )
    print(
        f"Black-Scholes European Call Price (sigma={sigma_alt}): {call_price_alt_sigma:.4f}"
    )
    print("\n")

    # Parameters for Put ~7.28 (slide's American Put section context)
    S0_put_context = 95.0
    K_put_context = 100.0
    sigma_put_context = 0.30
    put_price_context = black_scholes_european_put(
        S0_put_context, K_put_context, T_main, r_main, sigma_put_context
    )
    print(
        f"--- Parameters for Put ~7.28 (S0={S0_put_context}, K={K_put_context}, sigma={sigma_put_context}) ---"
    )
    print(f"Black-Scholes European Put Price: {put_price_context:.4f}")
