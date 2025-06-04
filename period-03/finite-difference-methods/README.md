# Period 03 Finite Difference Methods

This folder contains a comprehensive analysis of Partial Differential Equations in Finance, focusing on finite difference methods for Black-Scholes option pricing.

## Contents

### Main Document

- **`finite-difference-methods.tex`** - Complete LaTeX document covering all finite difference methods

### Article Structure

1. **Introduction** - Overview and objectives of the study
2. **Mathematical Foundation** - Black-Scholes PDE and classification as parabolic
3. **Analytical Solution** - Exact Black-Scholes formula with detailed calculations
4. **Finite Difference Methods** - Implementation and analysis of three numerical schemes:
   - Explicit finite difference method
   - Implicit finite difference method  
   - Crank-Nicolson method
5. **Comprehensive Results and Analysis** - Complete comparison of all methods
6. **Advanced Topics and Best Practices** - Grid refinement, convergence analysis, and practical guidelines
7. **Computational Implementation** - Software architecture and implementation details
8. **Conclusion** - Key findings and recommendations

## Key Results Summary

Using standard test parameters (S₀=$100, K=$100, T=0.25 years, r=5%, σ=20%):

| Method | Value ($) | Absolute Error ($) | Relative Error (%) | Time (s) |
|--------|----------|-------------------|-------------------|----------|
| **Analytical** | 4.6150 | --- | --- | 0.001 |
| **Explicit FD** | 4.5955 | 0.0195 | 0.42 | 0.124 |
| **Implicit FD** | 4.5944 | 0.0206 | 0.45 | 0.052 |
| **Crank-Nicolson** | 4.5949 | 0.0201 | 0.43 | 0.041 |

### Key Insights

1. **All numerical methods achieve >99.5% accuracy** with proper grid refinement
2. **Crank-Nicolson offers optimal balance** of accuracy, stability, and speed
3. **Grid refinement is crucial** - improves accuracy from 12.8% to 0.42%
4. **Method selection depends on specific requirements** and constraints

## Source Material

This comprehensive article synthesizes content from:

### Week 01 (w301.tex)
- Introduction to PDEs in financial applications
- Physical interpretation and mathematical properties
- Boundary and initial conditions

### Week 02 (w302.tex) 
- Black-Scholes mathematical analysis
- Proving the equation is parabolic
- Classification theory and discriminant analysis

### Week 03 (w303.tex)
- Finite difference methods introduction  
- Explicit method implementation and results
- Numerical vs analytical comparison

### Week 04 (w304.tex)
- Implicit finite difference method
- Crank-Nicolson method
- Comprehensive methods comparison

### Week 05 (w305.tex)
- Advanced grid techniques
- Best practices and optimization
- Extensions to exotic options

### Assignment 03 (assignment-03.tex)
- Problem-solving methodology
- Practical applications and exercises

### Computational Results (code/ directory)
- Complete implementations of all methods
- Detailed performance analysis and reports
- Validation against analytical benchmarks

## Compilation

To generate the PDF:

```bash
cd /home/julihocc/ma2008b/notes.worktrees/202511.p3/period-03/finite-difference-methods/
pdflatex finite-difference-methods.tex
pdflatex finite-difference-methods.tex  # Run twice for references
```

## Educational Value

This comprehensive article provides:

1. **Complete theoretical foundation** - From PDE classification to numerical implementation
2. **Practical computational results** - Real performance data and accuracy analysis  
3. **Implementation guidance** - Best practices for method selection and grid design
4. **Extension pathways** - Framework for advanced applications

The article serves as both a complete reference for the Period 03 material and a practical guide for implementing Black-Scholes numerical methods in computational finance applications.
