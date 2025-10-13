# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sklears is a Rust implementation of scikit-learn's functionality, providing machine learning algorithms with 3-100x performance improvements over Python. It's a workspace with multiple crates, each implementing different ML algorithms.

## Architecture

Three-layer architecture:
1. **Data Layer**: Polars DataFrames for data manipulation
2. **Computation Layer**: NumRS2 arrays with BLAS/LAPACK for numerical operations
3. **Algorithm Layer**: ML algorithms using SciRS2 for scientific computing

Key architectural patterns:
- Type-safe state machines (Untrained → Trained states)
- Builder pattern for algorithm configuration
- Trait-based design with core traits in `sklears-core`
- Feature flags for selective compilation

## Commands

```bash
# Build
cargo build
cargo build --all-features

# Test
cargo test                          # All tests
cargo test -p sklears-linear       # Specific crate
cargo test test_name               # Specific test
cargo test --all-features          # With all features

# Benchmarks
cargo bench
cargo bench --bench benchmark_name

# Code quality
cargo fmt                          # Format code
cargo clippy -- -D warnings        # Lint with warnings as errors

# Examples
cargo run --example quickstart
cargo run --example neural_network_demo

# Documentation
cargo doc --open
```

## Crate Structure

Main crates and their purposes:
- `sklears` - Facade crate that re-exports everything
- `sklears-core` - Core traits (Estimator, Fit, Predict, Transform)
- `sklears-linear` - Linear models (LinearRegression, Ridge, Lasso)
- `sklears-tree` - Decision trees and Random Forest
- `sklears-neighbors` - K-Nearest Neighbors
- `sklears-preprocessing` - Data preprocessing and pipelines
- `sklears-metrics` - Evaluation metrics
- `sklears-utils` - Shared utilities

## Development Notes

### Dependencies
- Local dependencies: `numrs2` and `scirs2` are expected at `../numrs/` and `../scirs/`
- BLAS backend issues on macOS ARM64 - tests may fail due to OpenMP linking
- Never forget "workspace policy"
- rand crate must be over v.0.9.1
- Never forget "Latest crates policy"

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT - SciRS2 Policy

**Sklears MUST use SciRS2 as its scientific computing foundation.**

### 📋 **Comprehensive Policy Document**
For complete SciRS2 integration guidelines, proven migration patterns, and detailed crate usage documentation, see:

**➡️ [SCIRS2_INTEGRATION_POLICY.md](./SCIRS2_INTEGRATION_POLICY.md)**

### 🎯 **Quick Reference - Key Patterns**

#### **Array Operations** (Working Patterns)
```rust
// ✅ FOR TESTS: When you need array! macro (tests, initialization)
#[cfg(test)]
mod tests {
    use scirs2_autograd::ndarray::array;  // ONLY in test modules

    #[test]
    fn test_example() {
        let X = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];
    }
}

// ✅ FOR PRODUCTION: Basic types (most production code)
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, ArrayViewMut};

// ✅ FOR PRODUCTION: Explicit construction (preferred)
let X = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

// ❌ NEVER: Direct ndarray usage
use ndarray::{Array, array};  // Policy violation
```

#### **Random Number Generation** (Working Patterns)
```rust
// ✅ Essential distributions
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::{CoreRandom, seeded_rng, thread_rng};

// ❌ NEVER: Direct rand usage
use rand::{Rng, thread_rng};  // Policy violation
```

#### **Current Status (v0.1.0-RC.1)**
- **✅ SciRS2 Ecosystem**: 100% POLICY-compliant (All 23 crates)
- **✅ Workspace Dependencies**: Updated to SciRS2 v0.1.0-RC.1
- **✅ scirs2_core::random**: ALL rand_distr distributions available
- **✅ scirs2_core::ndarray**: Complete ndarray including macros (array!, s!, azip!)
- **✅ Working Crates**: sklears-compose, sklears-core, and several others
- **🔄 Sklears Migration**: Systematic API updates in progress

### Testing Strategy
- Unit tests in each module
- Property-based tests using proptest
- Integration tests (some currently blocked by BLAS issues)
- Benchmarks using criterion

### Common Patterns

1. **Creating an estimator**:
```rust
let model = LinearRegression::builder()
    .fit_intercept(true)
    .build();
```

2. **Training and prediction**:
```rust
let trained_model = model.fit(&X_train, &y_train)?;
let predictions = trained_model.predict(&X_test)?;
```

### Known Issues
- **BLAS/OpenMP linking problems on macOS ARM64**: The workspace uses `ndarray-linalg` with `default-features = false` to avoid OpenMP linking errors with OpenBLAS on ARM64 macOS. This provides compatibility but reduces linear algebra performance. Advanced numerical operations may have degraded performance or numerical stability issues.
- Some integration tests are disabled due to dependency issues
- API consistency between crates is being improved
- **sklears-mixture test failures**: Due to the minimal BLAS configuration, some tests in sklears-mixture fail with shape compatibility and numerical stability issues. This is a known limitation of the ARM64 compatibility approach.

## Key Traits to Implement

When adding new algorithms:
1. Implement `Estimator` trait
2. Add appropriate capability traits (`Fit`, `Predict`, `Transform`)
3. Use type-safe state pattern (Untrained/Trained)
4. Add property-based tests
5. Include benchmarks