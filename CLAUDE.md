# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sklears is a Rust implementation of scikit-learn's functionality, providing type-safe, memory-safe machine learning algorithms with zero system dependencies. It's a workspace with multiple crates, each implementing different ML algorithms. Performance optimization is ongoing (see Performance Status section below).

## Architecture

Three-layer architecture:
1. **Data Layer**: Polars DataFrames for data manipulation
2. **Computation Layer**: SciRS2 arrays with Pure Rust BLAS/LAPACK (OxiBLAS) for numerical operations
3. **Algorithm Layer**: ML algorithms using SciRS2 for scientific computing

Key architectural patterns:
- Type-safe state machines (Untrained → Trained states)
- Builder pattern for algorithm configuration
- Trait-based design with core traits in `sklears-core`
- Feature flags for selective compilation
- Pure Rust stack (OxiBLAS v0.1.2+ for BLAS/LAPACK, Oxicode v0.1.1+ for serialization)

### Pure Rust Migration (v0.1.0+)

**Zero System Dependencies:**
- ✅ OxiBLAS v0.1.2+ - Pure Rust BLAS/LAPACK implementation
- ✅ Oxicode v0.1.1+ - SIMD-optimized serialization
- ✅ No OpenBLAS, MKL, or system BLAS required
- ✅ No C/Fortran compiler needed
- ✅ Easy cross-compilation on all platforms

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

### Dependencies (v0.1.0-rc.1 - Published from crates.io)
- **SciRS2**: v0.1.3 from crates.io (all scirs2-* crates)
- **OxiBLAS**: v0.1.2 (transitive dependency) - Pure Rust BLAS/LAPACK
- **Oxicode**: v0.1.1 - SIMD-optimized serialization (COOLJAPAN Policy)
- **NO system dependencies**: Pure Rust stack eliminates OpenBLAS/MKL requirements
- **NO BLAS linking issues**: OxiBLAS works seamlessly on all platforms including ARM64 macOS
- Never forget "workspace policy"
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

#### **Current Status (v0.1.0-rc.1)**
- **✅ SciRS2 Ecosystem**: 100% POLICY-compliant (All 23 crates)
- **✅ Workspace Dependencies**: Updated to SciRS2 v0.1.3 from crates.io
- **✅ scirs2_core::random**: ALL rand_distr distributions available
- **✅ scirs2_core::ndarray**: Complete ndarray including macros (array!, s!, azip!)
- **✅ scirs2-linalg**: Independent pure Rust implementation with OxiBLAS (no ndarray-linalg)
- **✅ Build Status**: 36/36 crates building successfully (100%)
- **✅ API Migration**: Oxicode, SVD, Solve, QR, Eigh APIs fully migrated
- **✅ Pure Rust Stack**: OxiBLAS v0.1.2 + Oxicode v0.1.1 integrated

#### **SciRS2 Migration Status (v0.1.0-rc.1)**
Complete nalgebra → scirs2-linalg migrations:
- **✅ sklears-decomposition**: 100% complete (6 files, 255 tests passing)
  - incremental_pca.rs, kernel_pca.rs, manifold.rs, tensor_decomposition.rs, matrix_completion.rs, pls.rs
- **✅ sklears-linear**: 100% complete (5 files, 137 tests passing)
  - glm.rs, serialization.rs, quantile.rs, constrained_optimization.rs, simd_optimizations.rs
- **✅ sklears-svm**: 100% complete (7 files, 256 tests passing)
  - grid_search.rs, bayesian_optimization.rs, random_search.rs, evolutionary_optimization.rs
  - advanced_optimization.rs, semi_supervised.rs, property_tests.rs

**Migration Benefits:**
- Zero system dependencies (Pure Rust OxiBLAS)
- Simplified codebase (eliminated nalgebra ↔ ndarray conversions)
- Better cross-platform compatibility
- Consistent error handling patterns

### Testing Strategy
- Unit tests in each module
- Property-based tests using proptest
- Integration tests (all working with OxiBLAS backend)
- Benchmarks using criterion
- **Recommended**: Use `cargo nextest run` for faster parallel testing

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

### Known Issues (Resolved in v0.1.1)

**✅ RESOLVED - Pure Rust Migration:**
- ~~BLAS/OpenMP linking problems on macOS ARM64~~ - **FIXED** with OxiBLAS pure Rust implementation
- ~~ndarray-linalg system dependency issues~~ - **FIXED** with scirs2-linalg independent implementation
- ~~OpenBLAS/MKL installation requirements~~ - **ELIMINATED** with pure Rust OxiBLAS
- ~~sklears-mixture test failures~~ - **FIXED** with stable OxiBLAS backend
- ~~Cross-compilation difficulties~~ - **RESOLVED** with pure Rust dependencies

**Current Status (v0.1.0-rc.1):**
- ✅ All 36/36 crates building successfully
- ✅ Zero system dependencies required
- ✅ Works on all platforms (macOS ARM64, Linux x86-64, Windows, etc.)
- ✅ API consistency maintained across all crates
- ✅ Full BLAS/LAPACK functionality via OxiBLAS v0.1.2
- ✅ 124+ unwrap() calls eliminated (ongoing - No Unwrap Policy compliance)
- ✅ Test suite: 4,409/4,410 tests passing (99.98%)

### Performance Status (v0.1.0-rc.1)

**Current Status**: Algorithms are correct and functional. Performance optimization is ongoing.

**Benchmark Results** (vs scikit-learn):
- Small datasets (6 samples): ~Equal performance (~0.5ms)
- Medium datasets (20-50 samples): 2x slower
- Large datasets (100 samples): 2-40x slower (requires optimization)

**Implemented Optimizations**:
- **WSS1 Working Set Selection**: O(n²) → O(n) per iteration complexity
  - Files: `crates/sklears-svm/src/smo.rs` (lines 342-395)
- **Data Storage Optimization**: Smart reuse via `assign()` for in-place copies
  - Files: `crates/sklears-svm/src/smo.rs` (lines 197-208)
- **Kernel Computation**: Removed temporary allocations
  - Files: `crates/sklears-svm/src/smo.rs` (line 694), `kernels.rs` (lines 148-160, 318-331)

**Why Performance Lags**:
- Algorithm correctness prioritized over performance in v0.1.0-rc.1
- Needs profiling and targeted optimizations
- Opportunity for SIMD, GPU acceleration, and better memory layouts

**Performance Roadmap**:
- v0.1.1: Profiling and algorithmic improvements
- v0.2.0: Performance parity with scikit-learn
- v0.3.0: Exceed scikit-learn with Rust-specific optimizations

### SciRS2-Linalg API (v0.1.3 - Pure Rust)

**Key API Patterns:**
```rust
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO, qr, svd};

// SVD - Returns matrices directly (not Option-wrapped)
let (u, s, vt) = matrix.svd(true)?;

// Eigenvalue decomposition - Requires UPLO parameter
let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Lower)?;

// Linear solve - Method call
let solution = a_matrix.solve(&b_vector)?;

// QR decomposition - Single parameter
let (q, r) = qr(&matrix.view())?;

// Matrix inverse
let inv_matrix = matrix.inv()?;

// Cholesky - No UPLO parameter
let chol = matrix.cholesky()?;
```

### Serialization - Oxicode API (COOLJAPAN Policy)

```rust
use oxicode::serde::{encode_to_vec, decode_from_slice};
use oxicode::config::standard;

// Serialization
let bytes = encode_to_vec(&model, standard())?;

// Deserialization (returns tuple)
let (model, _bytes_read) = decode_from_slice::<ModelType>(&bytes, standard())?;
```

## Key Traits to Implement

When adding new algorithms:
1. Implement `Estimator` trait
2. Add appropriate capability traits (`Fit`, `Predict`, `Transform`)
3. Use type-safe state pattern (Untrained/Trained)
4. Add property-based tests
5. Include benchmarks
6. Follow SciRS2 Policy - use scirs2-core abstractions only

## Version Information

- **Current Version**: v0.1.0-rc.1 (2026-02-05)
- **SciRS2 Version**: v0.1.3 (stable, from crates.io)
- **OxiBLAS Version**: v0.1.2 (pure Rust BLAS/LAPACK)
- **Oxicode Version**: v0.1.1 (SIMD-optimized serialization)
- **Build Status**: ✅ 36/36 crates building (100%)
- **Test Status**: ✅ 4,409/4,410 tests passing (99.98%)
- **Policy Compliance**: ✅ SciRS2 Policy fully adopted
- **Migrations Complete**: ✅ sklears-decomposition, sklears-linear, sklears-svm (18 files)
- **Repository**: https://github.com/cool-japan/sklears
- **Documentation**: See SCIRS2_INTEGRATION_POLICY.md for complete guidelines