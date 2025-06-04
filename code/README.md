# Black-Scholes Finite D## Methods Comparison

Using the example parameters (S₀=$100, K=$100, T=0.25 years, r=5%, σ=20%):

| Method | Value | Error | Relative Error | Computation Time |
|--------|-------|-------|----------------|------------------|
| **Analytical** | $4.6150 | — | — | ~0.001s |
| **Explicit FD** | $4.5955 | $0.0195 | 0.42% | ~0.124s |
| **Implicit FD** | $4.5944 | $0.0206 | 0.45% | ~0.052s |
| **Crank-Nicolson** | $4.5949 | $0.0201 | 0.43% | ~0.041s | Methods

This directory contains implementations of Black-Scholes option pricing using different numerical methods, based on the examples from w303.tex and w304.tex.

## Files Overview

### Implementation Scripts
- **`black_scholes_analytical.py`** - Analytical Black-Scholes solution (exact)
- **`black_scholes_explicit.py`** - Explicit finite difference method 
- **`black_scholes_implicit.py`** - Implicit finite difference method
- **`black_scholes_crank_nicolson.py`** - Crank-Nicolson finite difference method
- **`black_scholes_comparison.py`** - Comparison of all four methods

### Generated Reports
- **`black_scholes_analytical_report.txt`** - Detailed analytical method results
- **`black_scholes_explicit_report.txt`** - Detailed explicit method results  
- **`black_scholes_implicit_report.txt`** - Detailed implicit method results
- **`black_scholes_crank_nicolson_report.txt`** - Detailed Crank-Nicolson method results
- **`black_scholes_comparison_report.txt`** - Comprehensive comparison of all methods

### Configuration
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation file

## Methods Comparison

Using the example parameters (S₀=$100, K=$100, T=0.25 years, r=5%, σ=20%):

| Method | Value | Error | Relative Error | Computation Time |
|--------|-------|-------|----------------|------------------|
| **Analytical** | $4.6150 | — | — | ~0.0003s |
| **Explicit FD** | $4.5955 | $0.0195 | 0.42% | ~0.12s |
| **Implicit FD** | $4.5944 | $0.0206 | 0.45% | ~0.04s |

## Method Characteristics

### Analytical Method
- ✅ **Exact solution** (up to numerical precision)
- ✅ **Fastest computation**
- ❌ **Limited scope** (only standard European options)

### Explicit Finite Difference
- ✅ **Simple implementation**
- ✅ **Direct time stepping**
- ❌ **Stability constraints** (requires small time steps)
- ⚠️ **Slower than implicit** for same accuracy

### Implicit Finite Difference
- ✅ **Unconditionally stable**
- ✅ **Can use larger time steps**
- ✅ **Faster than explicit** for same grid
- ❌ **More complex implementation** (requires linear solver)
- ✅ **Similar accuracy** to explicit method

### Crank-Nicolson Finite Difference
- ✅ **Unconditionally stable** (like implicit)
- ✅ **Second-order accurate in time**
- ✅ **Best balance of accuracy and stability**
- ✅ **Fastest among numerical methods**
- ❌ **Most complex implementation**
- ✅ **Generally most accurate** for smooth solutions

## Running the Code

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Individual Methods
```bash
# Run analytical method
python black_scholes_analytical.py

# Run explicit finite difference
python black_scholes_explicit.py

# Run implicit finite difference  
python black_scholes_implicit.py

# Run Crank-Nicolson finite difference
python black_scholes_crank_nicolson.py

# Run comprehensive comparison
python black_scholes_comparison.py
```

### Quick Test
```python
from black_scholes_analytical import black_scholes_call_analytical
from black_scholes_explicit import black_scholes_call_explicit
from black_scholes_implicit import black_scholes_call_implicit
from black_scholes_crank_nicolson import black_scholes_call_crank_nicolson

# Example parameters
S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

analytical = black_scholes_call_analytical(S0, K, T, r, sigma)
explicit = black_scholes_call_explicit(S0, K, T, r, sigma)
implicit = black_scholes_call_implicit(S0, K, T, r, sigma)
crank_nicolson = black_scholes_call_crank_nicolson(S0, K, T, r, sigma)

print(f"Analytical:     ${analytical:.4f}")
print(f"Explicit:       ${explicit:.4f}")  
print(f"Implicit:       ${implicit:.4f}")
print(f"Crank-Nicolson: ${crank_nicolson:.4f}")
```

## Grid Parameters

The numerical methods use the following default grid:
- **Stock price range**: [0, 200]
- **Stock price steps**: 100 (Δs = 2.0)
- **Time steps**: 1000 (Δt = 0.00025)
- **Total grid points**: 101 × 1001

## Implementation Notes

- All methods use the same test parameters for consistency
- Reports are automatically generated in the script directory
- Implicit method uses scipy sparse matrices for efficient solving
- Boundary conditions: V(0,t) = 0, V(S_max,t) = S_max - K*exp(-r*t)
- Time integration proceeds backward from expiry to present

## Educational Value

This implementation demonstrates:
1. **Analytical vs Numerical**: Trade-offs between exact and approximate solutions
2. **Explicit vs Implicit**: Different approaches to time discretization
3. **Stability Analysis**: How time step constraints affect numerical methods
4. **Performance Comparison**: Speed vs accuracy trade-offs
5. **Grid Convergence**: Effect of discretization on solution quality

Based on the finite difference examples from w303.tex (explicit method) and w304.tex (implicit method).
