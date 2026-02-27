# SciRS2 Integration Policy for Sklears

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**Sklears MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into Sklears, following the official SciRS2 Ecosystem Policy.

**Important**: From v0.1.0, the SciRS2 POLICY is in effect. All non-core crates must use scirs2-core abstractions instead of direct external dependencies.

## Table of Contents

### Part I: Core Policies
1. [Overview](#overview)
2. [Pure Rust Migration (v0.1.0+)](#pure-rust-migration-v010)
3. [Dependency Abstraction Policy](#dependency-abstraction-policy)
4. [Core Architectural Principles](#core-architectural-principles)
5. [Implementation Guidelines](#implementation-guidelines)

### Part II: Sklears-Specific Integration
6. [Required SciRS2 Crates Analysis](#required-scirs2-crates-analysis)
7. [Sklears Module-Specific Usage](#sklears-module-specific-usage)
8. [Proven Successful Patterns](#proven-successful-patterns)

### Part III: Migration & Enforcement
9. [Standard Resolution Workflow](#standard-resolution-workflow)
10. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
11. [Enforcement & Quality Assurance](#enforcement--quality-assurance)

---

## Part I: Core Policies

## Overview

The SciRS2 ecosystem follows a strict layered architecture where **only the core crate can use external dependencies directly**, while all other crates must use SciRS2-Core abstractions.

### Architectural Hierarchy
```
Sklears (ML Library - Scikit-Learn-compatible API)
    ↓ builds upon
OptiRS (ML Optimization Specialization)
    ↓ builds upon
SciRS2 (Scientific Computing Foundation)
    ↓ builds upon
Pure Rust Stack (ndarray, OxiBLAS, oxicode, etc.)
```

This architecture ensures:
- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in SciRS2-core
- **Version Control**: Only SciRS2-core manages external dependency versions
- **Type Safety**: Prevents mixing external types with SciRS2 types

## Pure Rust Migration (v0.1.0+)

**Major architectural changes in SciRS2 v0.1.0 (January 2026):**

### OxiBLAS Migration - Pure Rust BLAS/LAPACK

**REMOVED Dependencies:**
- ❌ `ndarray-linalg` - Replaced with scirs2-linalg independent implementation
- ❌ `openblas-src` / `blas-src` / `lapack-src` - System BLAS libraries
- ❌ `accelerate-src` - macOS Accelerate Framework bindings
- ❌ `intel-mkl-src` - Intel MKL bindings
- ❌ `netlib-src` - Netlib reference implementation

**ADDED Dependencies:**
- ✅ `oxiblas-ndarray` v0.1.2+ - Pure Rust ndarray integration
- ✅ `oxiblas-blas` v0.1.2+ - Pure Rust BLAS implementation
- ✅ `oxiblas-lapack` v0.1.2+ - Pure Rust LAPACK implementation (supports Complex<f64>)

**Benefits:**
- 🚀 **Zero System Dependencies** - No need to install OpenBLAS, MKL, or system BLAS
- 🔧 **Easy Cross-Compilation** - Pure Rust works on all platforms
- 📦 **Simplified Builds** - No C/Fortran compiler required
- 🔒 **Complete Control** - Full Rust ecosystem integration
- ⚡ **SIMD Optimized** - Performance competitive with native BLAS

### Oxicode Migration - SIMD-Optimized Serialization (COOLJAPAN Policy)

**REMOVED Dependencies:**
- ❌ `bincode` - Generic binary serialization

**ADDED Dependencies:**
- ✅ `oxicode` v0.1.1+ - SIMD-optimized binary serialization
- ✅ `oxicode_derive` - Derive macros for custom types

**Benefits:**
- ⚡ **SIMD Acceleration** - Up to 4x faster than bincode
- 🎯 **Scientific Data Optimized** - Specialized for numeric arrays
- 🔒 **Type Safe** - Compile-time serialization verification

**API Usage:**
```rust
// Serialization
use oxicode::serde::{encode_to_vec, decode_from_slice};
use oxicode::config::standard;

let bytes = encode_to_vec(&data, standard())?;
let (data, _bytes_read) = decode_from_slice(&bytes, standard())?;
```

### SciRS2-Linalg Independent Implementation (v0.1.0+)

**Critical Change:** scirs2-linalg no longer depends on ndarray-linalg. It provides its own pure Rust implementation using OxiBLAS.

**Updated API:**
```rust
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};

// SVD (Singular Value Decomposition)
let (u, s, vt) = matrix.svd(true)?;  // Returns matrices directly, not Option-wrapped

// Eigenvalue decomposition
let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Lower)?;

// Linear solve
let solution = a_matrix.solve(&b_vector)?;

// QR decomposition
let (q, r) = qr(&matrix.view())?;

// Matrix inverse
let inv_matrix = matrix.inv()?;

// Cholesky decomposition
let chol = matrix.cholesky()?;  // No UPLO parameter needed
```

**Key API Changes from Legacy ndarray-linalg:**
1. SVD returns matrices directly (not `Option<Array2>`)
2. `eigh` requires `UPLO` enum parameter
3. `solve` is now a method: `matrix.solve(&b)` instead of `solve(&matrix, &b)`
4. `cholesky()` no longer takes UPLO parameter
5. All methods available through `ArrayLinalgExt` trait

## Dependency Abstraction Policy

### Core Principle: NO Direct External Dependencies in Sklears

**Applies to:** All Sklears crates
- `sklears-core`, `sklears-linear`, `sklears-tree`, `sklears-neighbors`, etc.
- All tests, examples, benchmarks in all crates
- All integration tests and documentation examples

#### ❌ PROHIBITED Direct Dependencies in Cargo.toml:
```toml
# ❌ FORBIDDEN in Sklears crates - Use scirs2-core instead
[dependencies]
rand = { workspace = true }              # ❌ Use scirs2-core::random
rand_distr = { workspace = true }        # ❌ Use scirs2-core::random
rand_core = { workspace = true }         # ❌ Use scirs2-core::random
rand_chacha = { workspace = true }       # ❌ Use scirs2-core::random
rand_pcg = { workspace = true }          # ❌ Use scirs2-core::random
ndarray = { workspace = true }           # ❌ Use scirs2-core::ndarray
ndarray-rand = { workspace = true }      # ❌ Use scirs2-core::ndarray
ndarray-stats = { workspace = true }     # ❌ Use scirs2-core::ndarray
ndarray-npy = { workspace = true }       # ❌ Use scirs2-core::ndarray
ndarray-linalg = { workspace = true }    # ❌ REMOVED v0.1.0 - scirs2-linalg independent
num-traits = { workspace = true }        # ❌ Use scirs2-core::numeric
num-complex = { workspace = true }       # ❌ Use scirs2-core::numeric
num-integer = { workspace = true }       # ❌ Use scirs2-core::numeric
nalgebra = { workspace = true }          # ❌ Use scirs2-core::linalg
bincode = { workspace = true }           # ❌ REMOVED v0.1.0 - Use oxicode (COOLJAPAN Policy)
openblas-src = { workspace = true }      # ❌ REMOVED v0.1.0 - Use OxiBLAS instead
blas-src = { workspace = true }          # ❌ REMOVED v0.1.0 - Use OxiBLAS instead
lapack-src = { workspace = true }        # ❌ REMOVED v0.1.0 - Use OxiBLAS instead
```

#### ✅ REQUIRED Core Dependency in Cargo.toml:
```toml
# ✅ REQUIRED in all Sklears crates
[dependencies]
scirs2-core = { workspace = true, features = ["array", "random", "linalg"] }
# All external dependencies accessed through scirs2-core
```

#### ❌ PROHIBITED Direct Imports in Code:
```rust
// ❌ FORBIDDEN in Sklears crates - POLICY VIOLATIONS
use rand::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Beta, Normal, StudentT};
use ndarray::*;
use ndarray::{Array, Array1, Array2};
use ndarray::{array, s};
use ndarray_linalg::*;  // REMOVED v0.1.0
use num_complex::Complex;
use num_traits::*;
use bincode::*;  // REMOVED v0.1.0 - Use oxicode instead
```

#### ✅ REQUIRED SciRS2-Core Abstractions:
```rust
// ✅ REQUIRED in Sklears crates and all tests/examples

// === Random Number Generation ===
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
// Includes: thread_rng, Rng, SliceRandom, etc.
// All distributions: Beta, Cauchy, ChiSquared, Normal, StudentT, Weibull, etc.

// === Array Operations ===
use scirs2_core::ndarray::*;          // Complete ndarray ecosystem
// Includes: Array, Array1, Array2, ArrayView, array!, s!, azip! macros
// Includes: ndarray-rand, ndarray-stats, ndarray-npy when array feature enabled
// NOTE: ndarray-linalg removed v0.1.0 - scirs2-linalg provides independent implementation

// === Numerical Traits ===
use scirs2_core::numeric::*;          // num-traits, num-complex, num-integer
// Includes: Float, Zero, One, Num, Complex, etc.

// === Advanced Types ===
use scirs2_core::array::*;            // Scientific array types
use scirs2_core::linalg::*;           // Linear algebra (when needed)
```

### Complete Dependency Mapping

| External Crate | SciRS2-Core Module | Note |
|----------------|-------------------|------|
| `rand` | `scirs2_core::random` | Full functionality |
| `rand_distr` | `scirs2_core::random` | All distributions |
| `rand_core` | `scirs2_core::random` | Core traits |
| `rand_chacha` | `scirs2_core::random` | ChaCha RNG |
| `rand_pcg` | `scirs2_core::random` | PCG RNG |
| `ndarray` | `scirs2_core::ndarray` | Full functionality |
| `ndarray-rand` | `scirs2_core::ndarray` | Via `array` feature |
| `ndarray-stats` | `scirs2_core::ndarray` | Via `array` feature |
| `ndarray-npy` | `scirs2_core::ndarray` | Via `array` feature |
| ~~`ndarray-linalg`~~ | N/A | **REMOVED** - scirs2-linalg provides independent implementation |
| `num-traits` | `scirs2_core::numeric` | All traits |
| `num-complex` | `scirs2_core::numeric` | Complex numbers |
| `num-integer` | `scirs2_core::numeric` | Integer traits |
| `nalgebra` | `scirs2_core::linalg` | When needed |
| `oxiblas-*` | `scirs2_core::linalg` | Pure Rust BLAS/LAPACK (v0.1.0+) |
| ~~`bincode`~~ | N/A | **REPLACED** by `oxicode` (SIMD-optimized) |
| `oxicode` | Direct usage | COOLJAPAN Policy - Pure Rust serialization |

## Core Architectural Principles

### 1. Foundation, Not Dependency Bloat
- Sklears extends SciRS2's capabilities with ML algorithm specialization
- Use SciRS2 crates **only when actually needed** by Sklears functionality
- **DO NOT** add SciRS2 crates "just in case"

### 2. Evidence-Based Integration
- Each SciRS2 crate must have **clear justification** based on Sklears features
- Document **specific use cases** for each integrated SciRS2 crate
- Remove unused SciRS2 dependencies during code reviews

### 3. Layered Abstraction Architecture
- Only `scirs2-core` may use external dependencies directly
- All Sklears crates must use SciRS2-Core abstractions
- Prevents mixing external types with SciRS2 types

### 4. Technical Policies (from SciRS2 Ecosystem Policy v3.0.0)

#### SIMD Operations Policy
- **ALWAYS use `scirs2-core::simd_ops::SimdUnifiedOps`** for SIMD operations
- **NEVER implement custom SIMD** code
- **ALWAYS provide scalar fallbacks** through the unified trait

#### Parallel Processing Policy
- **ALWAYS use `scirs2-core::parallel_ops`** for all parallel operations
- **NEVER add direct `rayon` dependency** to Cargo.toml
- **NEVER use `rayon::prelude::*` directly**

#### BLAS Operations Policy (v0.1.0+ Pure Rust)
- **ALL BLAS operations go through `scirs2-linalg`**
- **NEVER add direct BLAS dependencies** to individual modules
- **OxiBLAS** (Pure Rust) is the default and only backend
- **NO system dependencies required** - pure Rust implementation

#### GPU Operations Policy
- **ALWAYS use `scirs2-core::gpu` module** for GPU operations
- **NEVER implement direct CUDA/OpenCL/Metal kernels**
- **NEVER make direct GPU API calls** outside of SciRS2-core

## Implementation Guidelines

### For Developers

When writing code in Sklears crates:

1. **Never import external crates directly**
2. **Always use SciRS2-Core re-exports**
3. **Use CoreRandom instead of rand::Rng**
4. **Use SciRS2 array types instead of ndarray directly**
5. **Follow existing patterns in SciRS2 crates**

### For Tests and Examples

```rust
// ❌ Wrong - direct external usage (POLICY VIOLATIONS)
use rand::thread_rng;
use rand_distr::{Beta, Normal};
use ndarray::{Array2, array, s};
use bincode::{serialize, deserialize};  // REMOVED v0.1.0
let mut rng = thread_rng();
let arr = array![[1, 2], [3, 4]];
let slice = arr.slice(s![.., 0]);

// ✅ Correct - SciRS2-Core unified abstractions (v0.1.1+)
use scirs2_core::random::*;
use scirs2_core::ndarray::*;
use oxicode::serde::{encode_to_vec, decode_from_slice};
use oxicode::config::standard;

let mut rng = thread_rng();  // Now available through scirs2_core
let beta = RandBeta::new(2.0, 5.0)?;  // All distributions available
let arr = array![[1, 2], [3, 4]];  // array! macro works
let slice = arr.slice(s![.., 0]);  // s! macro works
let bytes = encode_to_vec(&data, standard())?;  // Oxicode serialization
```

---

## Part II: Sklears-Specific Integration

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Version**: v0.1.1 (stable)
- **Use Cases**: Core scientific primitives, random number generation, array operations
- **Sklears Modules**: All modules use core utilities
- **Status**: ✅ REQUIRED - Foundation crate

#### `scirs2-linalg` - LINEAR ALGEBRA (Independent Implementation)
- **Version**: v0.1.1 (stable)
- **Use Cases**: Matrix operations, decompositions, eigenvalue problems
- **Sklears Modules**: `sklears-linear`, `sklears-decomposition`, `sklears-gaussian-process`
- **Status**: ✅ REQUIRED - Pure Rust linear algebra with OxiBLAS
- **Note**: No longer depends on ndarray-linalg - independent implementation

### **HIGHLY LIKELY REQUIRED**

#### `scirs2-autograd` - AUTOMATIC DIFFERENTIATION
- **Version**: v0.1.1 (stable)
- **Use Cases**: Gradient computation, automatic differentiation for neural networks
- **Sklears Modules**: Gradient-based algorithms (neural networks, gradient descent)
- **Status**: ✅ REQUIRED - Automatic differentiation functionality
- **⚠️ CRITICAL**: NEVER import `scirs2_autograd::ndarray` - ALWAYS use `scirs2_core::ndarray` for arrays

#### `scirs2-optimize` - OPTIMIZATION
- **Version**: v0.1.1 (stable)
- **Use Cases**: Optimization algorithms (SGD, LBFGS, etc.)
- **Sklears Modules**: `sklears-linear`, `sklears-decomposition`, `sklears-neural`
- **Status**: ✅ REQUIRED - Core optimization functionality

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Version**: v0.1.1 (stable)
- **Use Cases**: Statistical functions, distributions, hypothesis testing
- **Sklears Modules**: `sklears-metrics`, `sklears-model-selection`
- **Status**: ✅ REQUIRED - Statistical computing

#### `scirs2-cluster` - CLUSTERING ALGORITHMS
- **Version**: v0.1.1 (stable)
- **Use Cases**: K-means, DBSCAN, hierarchical clustering
- **Sklears Modules**: `sklears-clustering`
- **Status**: ✅ REQUIRED - Clustering implementations

#### `scirs2-metrics` - EVALUATION METRICS
- **Version**: v0.1.1 (stable)
- **Use Cases**: Classification, regression, clustering metrics
- **Sklears Modules**: `sklears-metrics`
- **Status**: ✅ REQUIRED - Comprehensive evaluation

### **CONDITIONALLY REQUIRED**

#### `scirs2-datasets` - DATA HANDLING
- **Version**: v0.1.1 (stable)
- **Use Cases**: Built-in datasets, data generators
- **Sklears Modules**: `sklears-datasets`
- **Status**: ✅ REQUIRED - Data pipeline enhancement

#### `scirs2-sparse` - SPARSE MATRICES
- **Version**: v0.1.1 (stable)
- **Use Cases**: Sparse matrix operations, CSR/CSC formats
- **Sklears Modules**: `sklears-linear`, `sklears-feature-selection`
- **Status**: ✅ REQUIRED - Sparse functionality

#### `scirs2-neural` - NEURAL NETWORKS
- **Version**: v0.1.1 (stable)
- **Use Cases**: Neural network layers, activation functions
- **Sklears Modules**: `sklears-neural`
- **Status**: ✅ REQUIRED - Neural network features

### **SPECIALIZED DOMAIN-SPECIFIC**

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Version**: v0.1.1 (stable)
- **Use Cases**: Gamma, beta, error functions
- **Sklears Modules**: `sklears-stats`
- **Status**: ✅ REQUIRED - Special mathematical functions

#### `scirs2-spatial` - SPATIAL DATA PROCESSING
- **Version**: v0.1.1 (stable)
- **Use Cases**: KD-trees, spatial indexing, computational geometry
- **Sklears Modules**: `sklears-neighbors`, `sklears-clustering`
- **Status**: ✅ REQUIRED - Spatial operations

#### `scirs2-signal` - SIGNAL PROCESSING
- **Version**: v0.1.1 (stable)
- **Use Cases**: FFT, convolutions, signal processing
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - Signal processing capabilities

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Version**: v0.1.1 (stable)
- **Use Cases**: FFT operations, frequency domain transformations
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - FFT operations

#### `scirs2-series` - TIME SERIES ANALYSIS
- **Version**: v0.1.1 (stable)
- **Use Cases**: Time series decomposition, forecasting
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - Time series capabilities

#### `scirs2-text` - NLP PROCESSING
- **Version**: v0.1.1 (stable)
- **Use Cases**: Tokenization, text features
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - NLP capabilities

#### `scirs2-graph` - GRAPH ALGORITHMS
- **Version**: v0.1.1 (stable)
- **Use Cases**: Graph representations, algorithms
- **Sklears Modules**: `sklears-manifold`, `sklears-clustering`
- **Status**: ✅ REQUIRED - Graph neural networks

## Sklears Module-Specific Usage

### Array Import Patterns - **CRITICAL GUIDANCE**

```rust
// ✅ CORRECT FOR TESTS: When you need array! macro (tests, initialization)
#[cfg(test)]
mod tests {
    use scirs2_autograd::ndarray::array;  // ONLY in test modules

    #[test]
    fn test_example() {
        let X = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];
    }
}

// ✅ CORRECT FOR PRODUCTION: Basic types (most production code)
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, ArrayViewMut};

// ✅ CORRECT FOR PRODUCTION: Explicit construction (preferred)
let X = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

// ❌ POLICY VIOLATION: NEVER use ndarray directly
use ndarray::{Array, array};  // FORBIDDEN

// ❌ POLICY VIOLATION: NEVER use scirs2_autograd::ndarray in production
use scirs2_autograd::ndarray::Array;  // FORBIDDEN in production code
```

### Random Number Generation - **WORKING PATTERNS**

```rust
// ✅ WORKING PATTERN: Essential distributions
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::{CoreRandom, seeded_rng, thread_rng};

// ✅ WORKING PATTERN: RNG setup
use scirs2_core::random::rngs::StdRng;

// ✅ CLEAN TYPE DECLARATIONS
struct Algorithm {
    rng: CoreRandom<StdRng>,  // Seeded, deterministic
    // OR
    rng: CoreRandom,          // Thread-local, fast
}

// ✅ PROPER INITIALIZATION
Self { rng: seeded_rng(42) }      // For deterministic behavior
Self { rng: thread_rng() }        // For fast, non-deterministic

// ❌ WRONG: Direct rand usage
use rand::thread_rng;  // FORBIDDEN
use rand_distr::Normal;  // FORBIDDEN
```

### Linear Algebra - **v0.1.1 API** (Pure Rust with OxiBLAS)

```rust
// ✅ CORRECT: scirs2-linalg v0.1.1 API
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO, qr, svd};

// SVD - Returns matrices directly (not Option-wrapped)
let (u, s, vt) = matrix.svd(true)?;
// u, s, vt are Array2/Array1, not Option<Array2>

// Eigenvalue decomposition - Requires UPLO parameter
let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Lower)?;

// Linear solve - Now a trait method
let solution = a_matrix.solve(&b_vector)?;

// QR decomposition - Removed None parameter
let (q, r) = qr(&matrix.view())?;

// Matrix inverse - Now a trait method
let inv_matrix = matrix.inv()?;

// Cholesky - No UPLO parameter
let chol = matrix.cholesky()?;

// ❌ WRONG: Old ndarray-linalg API (REMOVED v0.1.0)
use ndarray_linalg::*;  // FORBIDDEN - No longer exists
let (u, s, vt) = matrix.svd(true, true)?;  // WRONG - Only takes 1 parameter
let solution = solve(&a.view(), &b.view(), None)?;  // WRONG - Use a.solve(&b)
let (q, r) = qr(&a.view(), None)?;  // WRONG - No None parameter
```

### Serialization - **Oxicode API** (COOLJAPAN Policy)

```rust
// ✅ CORRECT: oxicode v0.1.1 API
use oxicode::serde::{encode_to_vec, decode_from_slice};
use oxicode::config::standard;
use serde::{Serialize, Deserialize};

// Serialization
let bytes = encode_to_vec(&model, standard())?;

// Deserialization (returns tuple with bytes read)
let (model, _bytes_read) = decode_from_slice::<ModelType>(&bytes, standard())?;

// ❌ WRONG: Old bincode API (REMOVED v0.1.0)
use bincode::{serialize, deserialize};  // FORBIDDEN
let bytes = serialize(&model)?;  // WRONG - Use oxicode
let model = deserialize(&bytes)?;  // WRONG - Use oxicode
```

### Module-Specific Patterns

#### sklears-linear
```rust
use scirs2_core::ndarray::{Array, ArrayView};  // Basic arrays
use scirs2_optimize::*;                        // Solver implementations
use scirs2_linalg::compat::{ArrayLinalgExt, qr, svd};  // Linear algebra
```

#### sklears-tree
```rust
use scirs2_core::random::essentials::{Normal, Uniform};  // Random splitting
use scirs2_core::random::{CoreRandom, seeded_rng};       // Deterministic trees
```

#### sklears-neighbors
```rust
use scirs2_core::ndarray::*;       // Distance computations
use scirs2_spatial::*;             // KD-tree, spatial indexing
```

#### sklears-preprocessing
```rust
use scirs2_core::ndarray::*;       // Statistical preprocessing
use scirs2_stats::*;               // Statistical transformations
```

#### sklears-metrics
```rust
use scirs2_metrics::*;             // Foundation for all evaluation metrics
use scirs2_core::ndarray::*;       // Metric calculations
```

#### sklears-clustering
```rust
use scirs2_cluster::*;                                   // Foundation for clustering algorithms
use scirs2_core::random::essentials::{Normal, Uniform};  // Initialization
```

#### sklears-neural
```rust
use scirs2_neural::*;                      // Neural network layers
use scirs2_core::ndarray::{Array, array};  // For all array operations (NOT scirs2_autograd::ndarray)
use scirs2_autograd::*;                    // When actual autograd is needed
```

---

## Part III: Migration & Enforcement

## Proven Successful Patterns

### 🏆 Based on Successful Migrations

These patterns have been proven successful across multiple projects:

1. **Array Operations**
   ```rust
   // ✅ CORRECT
   use scirs2_core::ndarray::{Array, Array1, Array2, array, s};
   ```

2. **Random Number Generation**
   ```rust
   // ✅ CORRECT
   use scirs2_core::random::{CoreRandom, thread_rng, seeded_rng};
   use scirs2_core::random::essentials::{Normal, Uniform};
   ```

3. **Numerical Traits**
   ```rust
   // ✅ CORRECT
   use scirs2_core::numeric::{Float, Zero, One, Num};
   ```

4. **Linear Algebra (v0.1.1+ API)**
   ```rust
   // ✅ CORRECT
   use scirs2_linalg::compat::{ArrayLinalgExt, UPLO, qr, svd};

   let (u, s, vt) = matrix.svd(true)?;  // Direct return, not Option
   let solution = a.solve(&b)?;          // Method call, not function
   ```

5. **Serialization (v0.1.0+ Oxicode)**
   ```rust
   // ✅ CORRECT
   use oxicode::serde::{encode_to_vec, decode_from_slice};
   use oxicode::config::standard;

   let bytes = encode_to_vec(&data, standard())?;
   let (data, _) = decode_from_slice(&bytes, standard())?;
   ```

## Standard Resolution Workflow

### Proven 5-Step Migration Process

1. **Cargo.toml Cleanup**
   ```toml
   # ❌ REMOVE: Direct dependencies violating SciRS2 POLICY
   # rand = "0.9.2"         # REMOVED: Use scirs2_core::random
   # rand_distr = "0.5.1"   # REMOVED: Use scirs2_core::random
   # ndarray = "0.17"       # REMOVED: Use scirs2_core::ndarray
   # num-traits = "0.2"     # REMOVED: Use scirs2_core::numeric
   # num-complex = "0.4"    # REMOVED: Use scirs2_core::numeric
   # bincode = "1.3"        # REMOVED: Use oxicode v0.1.1+
   # openblas-src = "0.10"  # REMOVED: OxiBLAS used via scirs2-linalg

   # ✅ SciRS2 POLICY COMPLIANT dependencies
   scirs2-core = { workspace = true, features = ["random", "linalg"] }
   scirs2-linalg = { workspace = true }
   scirs2-autograd = { workspace = true }  # Only if autograd needed
   ```

2. **Import Path Migration**
   ```rust
   // ❌ OLD PATTERN (violated SciRS2 POLICY)
   use rand::{thread_rng, Rng};
   use ndarray::{Array, Array1, Array2};
   use ndarray_linalg::*;
   use num_traits::Float;
   use bincode::{serialize, deserialize};

   // ✅ NEW PATTERN (SciRS2 v0.1.1 compliant)
   use scirs2_core::random::{CoreRandom, thread_rng};
   use scirs2_core::ndarray::{Array, Array1, Array2, array};
   use scirs2_core::numeric::Float;
   use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
   use oxicode::serde::{encode_to_vec, decode_from_slice};
   use oxicode::config::standard;
   ```

3. **Type Declaration Fixes**
   ```rust
   // ❌ OLD: Confused nested types
   struct Algorithm {
       rng: CoreRandom<scirs2_core::Random<StdRng>>,  // WRONG
   }

   // ✅ NEW: Clean SciRS2 types
   struct Algorithm {
       rng: CoreRandom<StdRng>,  // Seeded, deterministic
   }
   ```

4. **Initialization Updates**
   ```rust
   // ❌ OLD: Incorrect initialization
   Self { rng: Random::seed(42) }

   // ✅ NEW: SciRS2 canonical patterns
   Self { rng: seeded_rng(42) }      // For deterministic behavior
   Self { rng: thread_rng() }        // For fast, non-deterministic
   ```

5. **Compilation Validation**
   - Test individual package compilation
   - Verify no remaining direct external dependencies
   - Ensure consistent patterns across codebase

## Anti-Patterns to Avoid

### ❌ Common Mistakes

```rust
// ❌ WRONG - Direct dependencies (POLICY VIOLATIONS)
use ndarray::{Array2, array, s};
use ndarray_linalg::{SVD, Eig};  // REMOVED v0.1.0
use rand::{Rng, thread_rng};
use rand_distr::{Normal, Beta, StudentT};
use num_traits::Float;
use num_complex::Complex;
use bincode::{serialize, deserialize};  // REMOVED v0.1.0

// ❌ WRONG - Nested type confusion
rng: CoreRandom<scirs2_core::Random<StdRng>>

// ❌ WRONG - Incorrect initialization
Self { rng: Random::seed(42) }

// ❌ WRONG - Using scirs2_autograd for arrays in production
use scirs2_autograd::ndarray::Array;

// ❌ WRONG - Old linalg API
let (u, s, vt) = matrix.svd(true, true)?;  // Old API
let solution = solve(&a.view(), &b.view(), None)?;  // Old API
```

### ✅ Correct Patterns (v0.1.1)

```rust
// ✅ CORRECT - SciRS2 policy compliant
use scirs2_core::ndarray::{Array2, array, s};         // All array operations
use scirs2_core::random::{CoreRandom, seeded_rng};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::numeric::{Float, Complex};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use oxicode::serde::{encode_to_vec, decode_from_slice};
use oxicode::config::standard;

// ✅ CORRECT - Clean types
rng: CoreRandom<StdRng>

// ✅ CORRECT - Proper initialization
Self { rng: seeded_rng(42) }

// ✅ CORRECT - New linalg API (v0.1.1)
let (u, s, vt) = matrix.svd(true)?;  // Single parameter, direct return
let solution = a.solve(&b)?;          // Method call
let (eigenvals, eigenvecs) = matrix.eigh(UPLO::Lower)?;  // UPLO required
```

## Enforcement & Quality Assurance

### Automated Checks
- CI pipeline checks for prohibited imports
- Documentation tests verify integration examples work
- Dependency graph analysis in builds
- `cargo deny` configuration for dependency restrictions (planned)

### Manual Reviews
- All SciRS2 integration changes require code review
- Code reviews MUST check for policy compliance
- Quarterly dependency audits
- Regular migration progress tracking

### Success Metrics
- **Compilation Success Rate**: Track package build success
- **Policy Compliance**: No direct rand/ndarray/num-* violations
- **Pattern Consistency**: Uniform usage across codebase
- **Performance**: Maintain or improve ML algorithm performance

## Current Workspace Integration

### SciRS2 Dependencies (v0.1.1 Stable)
```toml
[workspace.dependencies]
# Essential SciRS2 dependencies for Sklears (SciRS2 Policy compliance) - v0.1.1 from crates.io
scirs2-core = { version = "0.1.1", default-features = false, features = ["random", "linalg"] }

# TRANSITIONAL: Keep scirs2-autograd during migration (will be removed when all crates use scirs2_core::ndarray)
scirs2-autograd = { version = "0.1.1", default-features = false }

# HIGHLY LIKELY REQUIRED SciRS2 crates from crates.io
scirs2-optimize = { version = "0.1.1", default-features = false }
scirs2-linalg = { version = "0.1.1", default-features = false }
scirs2-stats = { version = "0.1.1", default-features = false }
scirs2-cluster = { version = "0.1.1", default-features = false }
scirs2-metrics = { version = "0.1.1", default-features = false }

# CONDITIONALLY REQUIRED SciRS2 crates from crates.io
scirs2-datasets = { version = "0.1.1", default-features = false }
scirs2-sparse = { version = "0.1.1", default-features = false }
scirs2-neural = { version = "0.1.1", default-features = false }

# SPECIALIZED DOMAIN-SPECIFIC SciRS2 crates from crates.io
scirs2-special = { version = "0.1.1", default-features = false }
scirs2-spatial = { version = "0.1.1", default-features = false }
scirs2-signal = { version = "0.1.1", default-features = false }
scirs2-series = { version = "0.1.1", default-features = false }
scirs2-text = { version = "0.1.1", default-features = false }
scirs2-fft = { version = "0.1.1", default-features = false }
scirs2-graph = { version = "0.1.1", default-features = false }

# Serialization (COOLJAPAN Policy)
oxicode = { version = "0.1.1" }
```

## Migration Status & Current State

### Sklears Migration Status (v0.1.0-rc.1)
- **✅ SciRS2 Ecosystem**: 100% POLICY-compliant (All 23 crates)
- **✅ Workspace Dependencies**: Updated to SciRS2 v0.1.3 from crates.io
- **✅ scirs2_core::random**: ALL rand_distr distributions available
- **✅ scirs2_core::ndarray**: Complete ndarray including macros (array!, s!, azip!)
- **✅ Working Crates**: All 36 crates including sklears-core, sklears-compose, sklears-linear, sklears-neural, sklears-svm, sklears-decomposition, sklears-kernel-approximation, sklears-manifold, sklears-covariance, and 27 more
- **✅ Build Status**: 36/36 crates building successfully (100%)
- **✅ Test Status**: 4,409/4,410 tests passing (99.98%)

### API Migration Complete (v0.1.1)
- ✅ Oxicode API: 11 files migrated from bincode
- ✅ SVD API: 100+ locations updated for new return types
- ✅ Solve API: 30+ locations migrated to method calls
- ✅ QR API: 5 locations updated
- ✅ Eigh API: 50+ locations updated with UPLO parameter
- ✅ All trait import conflicts resolved
- ✅ Pure Rust dependencies only (OxiBLAS v0.1.2, no system BLAS required)

### Nalgebra → scirs2-linalg Migrations (v0.1.0-rc.1)
Complete migration to Pure Rust stack:
- **✅ sklears-decomposition**: 100% complete (6 files, 255 tests passing)
  - incremental_pca.rs, kernel_pca.rs, manifold.rs, tensor_decomposition.rs, matrix_completion.rs, pls.rs
- **✅ sklears-linear**: 100% complete (5 files, 137 tests passing)
  - glm.rs, serialization.rs, quantile.rs, constrained_optimization.rs, simd_optimizations.rs
- **✅ sklears-svm**: 100% complete (7 files, 256 tests passing)
  - grid_search.rs, bayesian_optimization.rs, random_search.rs, evolutionary_optimization.rs
  - advanced_optimization.rs, semi_supervised.rs, property_tests.rs

**Total**: 18 files migrated, 648 tests passing, zero system dependencies

### SciRS2 Policy Compliance Status
As of SciRS2 v0.1.1 (Stable Release), **ALL 23 SciRS2 crates are POLICY-compliant (100%)**:
- ✅ scirs2-core provides unified abstractions for all external dependencies
- ✅ `scirs2_core::random` - ALL rand_distr distributions (Beta, Cauchy, StudentT, etc.)
- ✅ `scirs2_core::ndarray` - Complete ndarray including macros (`array!`, `s!`, `azip!`)
- ✅ `scirs2_core::numeric` - All num-traits, num-complex functionality
- ✅ `scirs2_linalg` - Independent pure Rust implementation with OxiBLAS (no ndarray-linalg dependency)

### Known Working Patterns
Based on successful migrations:
1. **sklears-compose** ✅ - Fixed async Send issues with proper mutex scoping
2. **sklears-svm** ✅ - Fixed choose_multiple with `scirs2_core::rand_prelude::IndexedRandom`
3. **sklears-manifold** ✅ - Fixed 102 compilation errors with new linalg API
4. **sklears-covariance** ✅ - Fixed 37 compilation errors with trait consolidation
5. **Workspace builds** ✅ - All 35/35 crates building successfully
6. **array! macro** ✅ - Available through `scirs2_autograd::ndarray::array` (tests only) OR `scirs2_core::ndarray::array`
7. **Pure Rust stack** ✅ - OxiBLAS v0.1.2 + Oxicode v0.1.1 fully integrated

## Future Considerations

### SciRS2 Version Management
- Track SciRS2 release cycle (currently on v0.1.1 stable)
- Test Sklears against SciRS2 releases
- Coordinate breaking change migrations
- Monitor OxiBLAS and Oxicode updates

### Community Alignment
- Coordinate with SciRS2 team on roadmap
- Contribute improvements back to SciRS2
- Maintain architectural consistency with scientific Rust ecosystem
- Follow COOLJAPAN Policy for preferred dependencies (OxiBLAS, Oxicode)

## Recent Enhancements (v0.1.1 Stable)

### Stable Core Abstractions (100% Complete)
As of v0.1.1, `scirs2_core::random` provides:
- ✅ All `rand_distr` distributions (Beta, Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull, etc.)
- ✅ Unified distribution interface with enhanced sampling
- ✅ Full compatibility with ecosystem projects
- ✅ Production-ready stability and performance

### Unified NDArray Module (100% Complete)
As of v0.1.1, `scirs2_core::ndarray` provides:
- ✅ Complete ndarray functionality including all macros (`array!`, `s!`, `azip!`)
- ✅ All array types, views, and operations
- ✅ Single unified import point for all array operations
- ✅ Backward compatibility with existing `ndarray_ext`
- ✅ Enhanced documentation and examples

### Pure Rust Linear Algebra (v0.1.0+)
As of v0.1.0, `scirs2_linalg` provides:
- ✅ Independent implementation (no ndarray-linalg dependency)
- ✅ Pure Rust BLAS/LAPACK via OxiBLAS v0.1.2+
- ✅ Zero system dependencies required
- ✅ Cross-platform compatibility
- ✅ Complete API: SVD, QR, Eigenvalue, Cholesky, Solve, Inverse
- ✅ Trait-based design with `ArrayLinalgExt`

### 🔧 Critical Implementation Note: array! Macro Usage

For **tests and examples ONLY**, when you need the `array!` macro:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // ✅ REQUIRED: Import array! macro for tests
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_example() {
        let X = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];
        // ... test code
    }
}
```

For **production code**, prefer explicit array construction:
```rust
// ✅ PRODUCTION: Use explicit construction
use scirs2_core::ndarray::{Array1, Array2};

let X = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
```

## Conclusion

This policy ensures Sklears properly leverages SciRS2's scientific computing foundation while maintaining scikit-learn API compatibility. **Sklears must use SciRS2 as its computational foundation through scirs2-core abstractions only.**

By following the official SciRS2 Ecosystem Policy v3.0.0, we achieve:
1. **Unified Performance**: All modules benefit from optimizations
2. **Easier Maintenance**: Updates in one place benefit all modules
3. **Consistent Behavior**: Same optimizations across the ecosystem
4. **Better Testing**: Centralized testing of critical operations
5. **Improved Portability**: Platform-specific code is isolated
6. **Reduced Duplication**: No repeated implementation of common operations
7. **Version Control**: Simplified dependency management
8. **Type Safety**: Consistent types across the ecosystem
9. **Pure Rust Stack**: Zero system dependencies with OxiBLAS and Oxicode

---

**Document Version**: 3.0.1 - Aligned with Official SciRS2 Ecosystem Policy v3.0.0
**Last Updated**: 2026-02-05 (Updated for SciRS2 v0.1.3 Stable + Sklears v0.1.0-rc.1)
**Based on**:
- SciRS2 Ecosystem Policy v3.0.0 (v0.1.3 Stable - 100% Complete)
- SciRS2 CLAUDE.md (v0.1.3 Stable)
- SciRS2 Core Module Usage Guidelines
- Sklears successful migration to published crates.io versions
- Complete nalgebra → scirs2-linalg migrations (18 files, 3 crates)
- COOLJAPAN Policy: OxiBLAS v0.1.2+ and Oxicode v0.1.1+
**Next Review**: Q2 2026
**Owner**: Sklears Architecture Team
**Status**: ✅ ACTIVE - All 36 Sklears crates using SciRS2 v0.1.3 from crates.io
