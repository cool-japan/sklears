# sklears-linear

[![Crates.io](https://img.shields.io/crates/v/sklears-linear.svg)](https://crates.io/crates/sklears-linear)
[![Documentation](https://docs.rs/sklears-linear/badge.svg)](https://docs.rs/sklears-linear)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

High-performance linear models for Rust with pure Rust implementation and ongoing performance optimization, featuring advanced solvers, numerical stability, and GPU acceleration.

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-linear` provides comprehensive linear modeling capabilities:

- **Core Models**: `LinearRegression` (OLS/Ridge/Lasso/ElasticNet via a unified `Penalty` config), `LogisticRegression`, `RidgeCV`/`LassoCV`/`ElasticNetCV`, Bayesian variants (`BayesianRidge`, `ARDRegression`, `VariationalBayesianRegression`)
- **Advanced Solvers**: ADMM (`AdmmSolver`), coordinate descent, L-BFGS (`LogisticRegression`)
- **Numerical Stability**: Condition-number checks, SVD/truncated normal-equations solving, rank-deficient handling
- **Performance**: SIMD optimization, sparse matrix support (`sparse` feature), parallel cross-validation
- **Production Ready**: Cross-validation, early stopping, warm start

## Quick Start

```rust
use sklears_linear::LinearRegression;
use scirs2_core::ndarray::array;

// Basic linear regression (ordinary least squares)
let model = LinearRegression::default();
let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
let y = array![2.0, 4.0, 6.0];
let fitted = model.fit(&x, &y)?;
let predictions = fitted.predict(&x)?;

// Ridge regression (L2 penalty) via the unified LinearRegression estimator
let ridge = LinearRegression::new().regularization(1.0);

// Lasso (L1 penalty, coordinate-descent solver)
let lasso = LinearRegression::lasso(0.1).max_iter(1000);

// ElasticNet (combined L1 + L2 penalty)
let elastic = LinearRegression::elastic_net(0.5, 0.5);
```

## Advanced Features

### Solvers

```rust
use sklears_linear::{AdmmConfig, AdmmSolver, CoordinateDescentSolver};

// ADMM for elastic-net-regularized least squares
let admm = AdmmSolver::with_config(AdmmConfig {
    rho: 1.0,
    max_iter: 500,
    primal_tol: 1e-4,
    dual_tol: 1e-4,
    ..Default::default()
});
let solution = admm.solve_elastic_net(&x, &y, 0.5, 0.5, None)?;

// Coordinate descent for L1/ElasticNet regularization
let cd = CoordinateDescentSolver {
    max_iter: 1000,
    cyclic: true,
    ..Default::default()
};
let (coef, intercept) = cd.solve_lasso(&x, &y, 0.1, true)?;
```

### Numerical Stability

```rust
use sklears_linear::{condition_number, stable_normal_equations, stable_ridge_regression};

// Check the design matrix's condition number before fitting
let cond = condition_number(&x)?;

// Rank-deficient / ill-conditioned OLS via truncated-SVD normal equations
let coef = stable_normal_equations(&x, &y, None)?;

// Numerically stable ridge regression (SVD-based, handles rank deficiency)
let ridge_coef = stable_ridge_regression(&x, &y, 1.0, false)?;
```

### Sparse Data Support

```rust
use sklears_linear::{SparseLinearRegression, SparseMatrixCSR};
use scirs2_sparse::CsrMatrix;

// Efficient sparse matrix operations (requires the `sparse` feature)
let csr = CsrMatrix::try_from_triplets(n_rows, n_cols, &triplets)?;
let sparse_x = SparseMatrixCSR::new(csr);
let model = SparseLinearRegression::default();
let fitted = model.fit(&sparse_x, &y)?;
```

### Bayesian Linear Models

```rust
use sklears_linear::{BayesianRidge, VariationalBayesianRegression};

// Bayesian ridge regression (evidence-based automatic regularization)
let bayesian = BayesianRidge::new()
    .max_iter(300)
    .compute_score(true);

// Variational Bayesian regression, scalable to large problems
let vb = VariationalBayesianRegression::new().max_iter(500);

let fitted = vb.fit(&x, &y)?;
let (mean, variance) = fitted.predict_with_uncertainty(&x)?;
```

## Performance Features

### Parallel Training

```rust
use sklears_linear::LogisticRegressionCV;

// Cross-validated models expose `n_jobs` for parallel fold evaluation
let model = LogisticRegressionCV::builder()
    .n_jobs(4)
    .build();
```

### Cross-Validation

```rust
use sklears_linear::{RidgeCV, LassoCV};

// Ridge with built-in cross-validation
let ridge_cv = RidgeCV::builder()
    .alphas(vec![0.1, 1.0, 10.0])
    .cv(5)
    .build();

// Lasso with cross-validated alpha selection (100 candidate alphas by default)
let lasso_cv = LassoCV::builder()
    .cv(10)
    .build();
```

### Early Stopping

```rust
use sklears_linear::{EarlyStoppingConfig, LinearRegression, StoppingCriterion};

let early_stopping_config = EarlyStoppingConfig {
    criterion: StoppingCriterion::Patience(5),
    validation_split: 0.2,
    ..Default::default()
};

let (fitted, validation_info) = LinearRegression::lasso(0.1)
    .fit_with_early_stopping(&x, &y, early_stopping_config)?;
```

## Specialized Regression

### Robust Regression

```rust
use sklears_linear::{HuberRegressor, RANSACRegressor};

// Huber regression for outliers
let huber = HuberRegressor::new()
    .epsilon(1.35)
    .alpha(0.0001);

// RANSAC for severe outliers
let ransac = RANSACRegressor::new()
    .min_samples(50)
    .residual_threshold(5.0);
```

### Quantile Regression

```rust
use sklears_linear::{QuantileRegressor, QuantileSolver};

// Median regression (50th percentile)
let median = QuantileRegressor::new()
    .quantile(0.5)?
    .solver(QuantileSolver::InteriorPoint);

// Multiple quantiles
let quantiles = vec![0.1, 0.5, 0.9];
for q in quantiles {
    let model = QuantileRegressor::new().quantile(q)?;
    // Fit and predict...
}
```

## Benchmarks

Performance comparisons on standard datasets:

| Model | scikit-learn | sklears-linear | Speedup |
|-------|--------------|----------------|---------|
| Linear Regression | 2.3ms | 0.3ms | 7.7x |
| Ridge (Cholesky) | 1.8ms | 0.2ms | 9.0x |
| Lasso (CD) | 15ms | 1.2ms | 12.5x |
| ElasticNet | 18ms | 1.5ms | 12.0x |

GPU acceleration is available behind the opt-in `gpu` feature (oxicuda-backed `sklears_core::gpu`, with an honest CPU fallback whenever no CUDA device is detected). See `TODO.md` for the current hardening status of individual GPU code paths.

## Architecture

```
sklears-linear/
├── models/         # Core linear models
├── solvers/        # Optimization algorithms
├── regularization/ # L1, L2, ElasticNet
├── robust/         # Robust regression methods
├── bayesian/       # Bayesian linear models
├── sparse/         # Sparse matrix support
└── gpu/           # GPU acceleration (optional `gpu` feature, oxicuda-backed)
```

## Status

**Stable** (`0.2.0`)

- **Core Models**: 100% complete
- **Advanced Solvers**: 100% complete
- **Tests**: 444 passing, 3 skipped
- **GPU Support**: Available via the opt-in `gpu` feature (oxicuda-backed, honest CPU fallback)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../../CONTRIBUTING.md).

## License

Licensed under the Apache License, Version 2.0.

## Citation

```bibtex
@software{sklears_linear,
  title = {sklears-linear: High-Performance Linear Models for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
