# SciRS2 Integration Policy for Sklears

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**Sklears MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into Sklears, following the official SciRS2 Ecosystem Policy.

## Table of Contents

### Part I: Core Policies
1. [Overview](#overview)
2. [Dependency Abstraction Policy](#dependency-abstraction-policy)
3. [Core Architectural Principles](#core-architectural-principles)
4. [Implementation Guidelines](#implementation-guidelines)

### Part II: Sklears-Specific Integration
5. [Required SciRS2 Crates Analysis](#required-scirs2-crates-analysis)
6. [Sklears Module-Specific Usage](#sklears-module-specific-usage)
7. [Proven Successful Patterns](#proven-successful-patterns)

### Part III: Migration & Enforcement
8. [Standard Resolution Workflow](#standard-resolution-workflow)
9. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
10. [Enforcement & Quality Assurance](#enforcement--quality-assurance)

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
ndarray, num-traits, rand, etc. (Core Rust Scientific Stack)
```

This architecture ensures:
- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in SciRS2-core
- **Version Control**: Only SciRS2-core manages external dependency versions
- **Type Safety**: Prevents mixing external types with SciRS2 types

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
ndarray-linalg = { workspace = true }    # ❌ Use scirs2-core::ndarray
num-traits = { workspace = true }        # ❌ Use scirs2-core::numeric
num-complex = { workspace = true }       # ❌ Use scirs2-core::numeric
num-integer = { workspace = true }       # ❌ Use scirs2-core::numeric
nalgebra = { workspace = true }          # ❌ Use scirs2-core::linalg
```

#### ✅ REQUIRED Core Dependency in Cargo.toml:
```toml
# ✅ REQUIRED in all Sklears crates
[dependencies]
scirs2-core = { workspace = true, features = ["array", "random"] }
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
use num_complex::Complex;
use num_traits::*;
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
// Includes: ndarray-linalg, ndarray-stats, ndarray-npy when array feature enabled

// === Numerical Traits ===
use scirs2_core::numeric::*;          // num-traits, num-complex, num-integer
// Includes: Float, Zero, One, Num, Complex, etc.

// === Advanced Types ===
use scirs2_core::array::*;            // Scientific array types
use scirs2_core::linalg::*;           // Linear algebra (nalgebra when needed)
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
| `ndarray-linalg` | `scirs2_core::ndarray` | Via `array` feature |
| `num-traits` | `scirs2_core::numeric` | All traits |
| `num-complex` | `scirs2_core::numeric` | Complex numbers |
| `num-integer` | `scirs2_core::numeric` | Integer traits |
| `nalgebra` | `scirs2_core::linalg` | When needed |

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

### 4. Technical Policies (from SciRS2 Ecosystem Policy)

#### SIMD Operations Policy
- **ALWAYS use `scirs2-core::simd_ops::SimdUnifiedOps`** for SIMD operations
- **NEVER implement custom SIMD** code
- **ALWAYS provide scalar fallbacks** through the unified trait

#### Parallel Processing Policy
- **ALWAYS use `scirs2-core::parallel_ops`** for all parallel operations
- **NEVER add direct `rayon` dependency** to Cargo.toml
- **NEVER use `rayon::prelude::*` directly**

#### BLAS Operations Policy
- **ALL BLAS operations go through `scirs2-core`**
- **NEVER add direct BLAS dependencies** to individual modules
- Backend selection is handled by SciRS2-core's platform configuration

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
// ❌ Wrong - direct external usage
use rand::thread_rng;
use rand_distr::{Beta, Normal};
use ndarray::{Array2, array, s};
let mut rng = thread_rng();
let arr = array![[1, 2], [3, 4]];
let slice = arr.slice(s![.., 0]);

// ✅ Correct - SciRS2-Core unified abstractions (v0.1.0-beta.4+)
use scirs2_core::random::*;
use scirs2_core::ndarray::*;

let mut rng = thread_rng();  // Now available through scirs2_core
let beta = RandBeta::new(2.0, 5.0)?;  // All distributions available
let arr = array![[1, 2], [3, 4]];  // array! macro works
let slice = arr.slice(s![.., 0]);  // s! macro works
```

---

## Part II: Sklears-Specific Integration

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Use Cases**: Core scientific primitives, random number generation, array operations
- **Sklears Modules**: All modules use core utilities
- **Status**: ✅ REQUIRED - Foundation crate

#### `scirs2` - MAIN INTEGRATION
- **Use Cases**: Core scientific computing, linear algebra integration
- **Sklears Modules**: Core operations throughout
- **Features**: ["standard", "linalg", "stats"]
- **Status**: ✅ REQUIRED - Main integration crate

### **HIGHLY LIKELY REQUIRED**

#### `scirs2-autograd` - AUTOMATIC DIFFERENTIATION
- **Use Cases**: Gradient computation, automatic differentiation for neural networks
- **Sklears Modules**: Gradient-based algorithms (neural networks, gradient descent)
- **Status**: ✅ REQUIRED - Automatic differentiation functionality
- **⚠️ CRITICAL**: NEVER import `scirs2_autograd::ndarray` - ALWAYS use `scirs2_core::ndarray` for arrays

#### `scirs2-optimize` - OPTIMIZATION
- **Use Cases**: Optimization algorithms (SGD, LBFGS, etc.)
- **Sklears Modules**: `sklears-linear`, `sklears-decomposition`, `sklears-neural`
- **Status**: ✅ REQUIRED - Core optimization functionality

#### `scirs2-linalg` - LINEAR ALGEBRA
- **Use Cases**: Matrix operations, decompositions, eigenvalue problems
- **Sklears Modules**: `sklears-linear`, `sklears-decomposition`, `sklears-gaussian-process`
- **Status**: ✅ REQUIRED - Linear algebra operations

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Use Cases**: Statistical functions, distributions, hypothesis testing
- **Sklears Modules**: `sklears-metrics`, `sklears-model-selection`
- **Status**: ✅ REQUIRED - Statistical computing

#### `scirs2-cluster` - CLUSTERING ALGORITHMS
- **Use Cases**: K-means, DBSCAN, hierarchical clustering
- **Sklears Modules**: `sklears-clustering`
- **Status**: ✅ REQUIRED - Clustering implementations

#### `scirs2-metrics` - EVALUATION METRICS
- **Use Cases**: Classification, regression, clustering metrics
- **Sklears Modules**: `sklears-metrics`
- **Status**: ✅ REQUIRED - Comprehensive evaluation

### **CONDITIONALLY REQUIRED**

#### `scirs2-datasets` - DATA HANDLING
- **Use Cases**: Built-in datasets, data generators
- **Sklears Modules**: `sklears-datasets`
- **Status**: ✅ REQUIRED - Data pipeline enhancement

#### `scirs2-sparse` - SPARSE MATRICES
- **Use Cases**: Sparse matrix operations, CSR/CSC formats
- **Sklears Modules**: `sklears-linear`, `sklears-feature-selection`
- **Status**: ✅ REQUIRED - Sparse functionality

#### `scirs2-neural` - NEURAL NETWORKS
- **Use Cases**: Neural network layers, activation functions
- **Sklears Modules**: `sklears-neural`
- **Status**: ✅ REQUIRED - Neural network features

### **SPECIALIZED DOMAIN-SPECIFIC**

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Use Cases**: Gamma, beta, error functions
- **Sklears Modules**: `sklears-stats`
- **Status**: ✅ REQUIRED - Special mathematical functions

#### `scirs2-signal` - SIGNAL PROCESSING
- **Use Cases**: FFT, convolutions, signal processing
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - Signal processing capabilities

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Use Cases**: FFT operations, frequency domain transformations
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - FFT operations

#### `scirs2-series` - TIME SERIES ANALYSIS
- **Use Cases**: Time series decomposition, forecasting
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - Time series capabilities

#### `scirs2-spatial` - SPATIAL DATA PROCESSING
- **Use Cases**: KD-trees, spatial indexing, computational geometry
- **Sklears Modules**: `sklears-neighbors`, `sklears-clustering`
- **Status**: ✅ REQUIRED - Spatial operations

#### `scirs2-text` - NLP PROCESSING
- **Use Cases**: Tokenization, text features
- **Sklears Modules**: `sklears-feature-extraction`
- **Status**: ✅ REQUIRED - NLP capabilities

#### `scirs2-graph` - GRAPH ALGORITHMS
- **Use Cases**: Graph representations, algorithms
- **Sklears Modules**: `sklears-manifold`, `sklears-clustering`
- **Status**: ✅ REQUIRED - Graph neural networks

## Sklears Module-Specific Usage

### Array Import Patterns - **CRITICAL GUIDANCE**

```rust
// ✅ CORRECT: ALL array operations through scirs2-core ONLY
use scirs2_core::ndarray::{Array, Array1, Array2, array, s, Axis};

// ✅ ALSO CORRECT: For extended operations (optional)
use scirs2_core::ndarray_ext::{/* extended utilities */};

// ❌ POLICY VIOLATION: NEVER use ndarray directly
use ndarray::{Array, array};  // FORBIDDEN - Breaks SciRS2 integration

// ❌ POLICY VIOLATION: NEVER use scirs2_autograd::ndarray
use scirs2_autograd::ndarray::*;  // FORBIDDEN - Use scirs2_core::ndarray instead

// 📝 NOTE: scirs2_autograd is for automatic differentiation ONLY
// For any array operations, ALWAYS use scirs2_core::ndarray
```

### Random Number Generation - **WORKING PATTERNS**

```rust
// ✅ WORKING PATTERN: Essential distributions
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::ndarray::distributions::Distribution;

// ✅ WORKING PATTERN: RNG setup
use scirs2_core::random::{CoreRandom, seeded_rng, thread_rng};
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
```

### Module-Specific Patterns

#### sklears-linear
```rust
use scirs2_core::ndarray::{Array, ArrayView};  // Basic arrays
use scirs2_optimize::*;                        // Solver implementations
use scirs2_linalg::*;                          // Advanced linear algebra
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
use scirs2_core::ndarray::{Array, array};  // For all array operations
use scirs2_autograd::*;                    // When actual autograd is needed
```

---

## Part III: Migration & Enforcement

## Proven Successful Patterns

### 🏆 Based on ToRSh 96.7% Success Rate

These patterns have been proven successful in ToRSh migration:

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

## Standard Resolution Workflow

### Proven 5-Step Migration Process

1. **Cargo.toml Cleanup**
   ```toml
   # ❌ REMOVE: Direct dependencies violating SciRS2 POLICY
   # rand = "0.9.2"         # REMOVED: Use scirs2_core::random
   # rand_distr = "0.5.1"   # REMOVED: Use scirs2_core::random
   # ndarray = "0.16"       # REMOVED: Use scirs2_core::ndarray
   # num-traits = "0.2"     # REMOVED: Use scirs2_core::numeric
   # num-complex = "0.4"    # REMOVED: Use scirs2_core::numeric

   # ✅ SciRS2 POLICY COMPLIANT dependencies
   scirs2-core = { workspace = true }
   scirs2-autograd = { workspace = true }  # Only if autograd needed
   ```

2. **Import Path Migration**
   ```rust
   // ❌ OLD PATTERN (violated SciRS2 POLICY)
   use rand::{thread_rng, Rng};
   use ndarray::{Array, Array1, Array2};
   use num_traits::Float;

   // ✅ NEW PATTERN (SciRS2 compliant)
   use scirs2_core::random::{CoreRandom, thread_rng};
   use scirs2_core::ndarray::{Array, Array1, Array2, array};
   use scirs2_core::numeric::Float;
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
use rand::{Rng, thread_rng};
use rand_distr::{Normal, Beta, StudentT};
use num_traits::Float;
use num_complex::Complex;

// ❌ WRONG - Nested type confusion
rng: CoreRandom<scirs2_core::Random<StdRng>>

// ❌ WRONG - Incorrect initialization
Self { rng: Random::seed(42) }

// ❌ WRONG - Using scirs2_autograd for arrays
use scirs2_autograd::ndarray::Array;
```

### ✅ Correct Patterns
```rust
// ✅ CORRECT - SciRS2 policy compliant
use scirs2_core::ndarray::{Array2, array, s};         // All array operations
use scirs2_core::random::{CoreRandom, seeded_rng};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::numeric::{Float, Complex};

// ✅ CORRECT - Clean types
rng: CoreRandom<StdRng>

// ✅ CORRECT - Proper initialization
Self { rng: seeded_rng(42) }
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

### SciRS2 Dependencies (v0.1.0-RC.1)
```toml
[workspace.dependencies]
# Essential SciRS2 dependencies for Sklears (SciRS2 Policy compliance)
# Note: SciRS2 Policy is now 100% COMPLETE as of v0.1.0-RC.1
scirs2 = { version = "0.1.0-rc.1", features = ["standard"], default-features = false }
scirs2-core = { version = "0.1.0-rc.1", default-features = false }

# Automatic differentiation
scirs2-autograd = { version = "0.1.0-rc.1", default-features = false }

# HIGHLY LIKELY REQUIRED SciRS2 crates
scirs2-optimize = { version = "0.1.0-rc.1", default-features = false }
scirs2-linalg = { version = "0.1.0-rc.1", default-features = false }
scirs2-stats = { version = "0.1.0-rc.1", default-features = false }
scirs2-cluster = { version = "0.1.0-rc.1", default-features = false }
scirs2-metrics = { version = "0.1.0-rc.1", default-features = false }

# CONDITIONALLY REQUIRED SciRS2 crates
scirs2-datasets = { version = "0.1.0-rc.1", default-features = false }
scirs2-sparse = { version = "0.1.0-rc.1", default-features = false }
scirs2-neural = { version = "0.1.0-rc.1", default-features = false }

# SPECIALIZED DOMAIN-SPECIFIC SciRS2 crates
scirs2-special = { version = "0.1.0-rc.1", default-features = false }
scirs2-spatial = { version = "0.1.0-rc.1", default-features = false }
scirs2-signal = { version = "0.1.0-rc.1", default-features = false }
scirs2-series = { version = "0.1.0-rc.1", default-features = false }
scirs2-text = { version = "0.1.0-rc.1", default-features = false }
scirs2-fft = { version = "0.1.0-rc.1", default-features = false }
scirs2-graph = { version = "0.1.0-rc.1", default-features = false }
```

## Migration Status & Next Steps

### Current Progress (SciRS2 v0.1.0-RC.1)
- **✅ SciRS2 Ecosystem**: 100% POLICY-compliant (All 23 crates)
- **✅ Workspace Dependencies**: Updated to SciRS2 v0.1.0-RC.1
- **✅ Core Infrastructure**: sklears-compose, sklears-core working
- **🔄 Sklears Migration**: Systematic removal of prohibited dependencies in progress

### SciRS2 Policy Compliance Status
As of SciRS2 v0.1.0-RC.1, **ALL 23 SciRS2 crates are now POLICY-compliant (100%)**:
- ✅ scirs2-core provides unified abstractions for all external dependencies
- ✅ `scirs2_core::random` - ALL rand_distr distributions (Beta, Cauchy, StudentT, etc.)
- ✅ `scirs2_core::ndarray` - Complete ndarray including macros (`array!`, `s!`, `azip!`)
- ✅ `scirs2_core::numeric` - All num-traits, num-complex functionality

### Known Working Patterns
Based on successful migrations:
1. **sklears-compose** ✅ - Fixed async Send issues with proper mutex scoping
2. **sklears-svm** ✅ - Fixed choose_multiple with `scirs2_core::rand_prelude::IndexedRandom`
3. **Workspace builds** ✅ - Dependency resolution working correctly
4. **array! macro** ✅ - Available through `scirs2_autograd::ndarray::array` (tests only) OR `scirs2_core::ndarray::array`

## Future Considerations

### SciRS2 Version Management
- Track SciRS2 release cycle (currently on 0.1.0-beta.3)
- Test Sklears against SciRS2 beta releases
- Coordinate breaking change migrations

### Community Alignment
- Coordinate with SciRS2 team on roadmap
- Contribute improvements back to SciRS2
- Maintain architectural consistency with scientific Rust ecosystem

## Recent Enhancements (v0.1.0-RC.1)

### Stable Core Abstractions (100% Complete)
As of RC.1, `scirs2_core::random` provides:
- ✅ All `rand_distr` distributions (Beta, Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull, etc.)
- ✅ Unified distribution interface with enhanced sampling
- ✅ Full compatibility with ecosystem projects
- ✅ Production-ready stability and performance

### Unified NDArray Module (100% Complete)
As of RC.1, `scirs2_core::ndarray` provides:
- ✅ Complete ndarray functionality including all macros (`array!`, `s!`, `azip!`)
- ✅ All array types, views, and operations
- ✅ Single unified import point for all array operations
- ✅ Backward compatibility with existing `ndarray_ext`
- ✅ Enhanced documentation and examples

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

By following the official SciRS2 Ecosystem Policy, we achieve:
1. **Unified Performance**: All modules benefit from optimizations
2. **Easier Maintenance**: Updates in one place benefit all modules
3. **Consistent Behavior**: Same optimizations across the ecosystem
4. **Better Testing**: Centralized testing of critical operations
5. **Improved Portability**: Platform-specific code is isolated
6. **Reduced Duplication**: No repeated implementation of common operations
7. **Version Control**: Simplified dependency management
8. **Type Safety**: Consistent types across the ecosystem

---

**Document Version**: 2.1 - Aligned with Official SciRS2 Ecosystem Policy v3.0.0
**Last Updated**: 2025-10-09 (Updated for SciRS2 v0.1.0-RC.1)
**Based on**:
- SciRS2 Ecosystem Policy v3.0.0 (v0.1.0-RC.1 - 100% Complete)
- SciRS2 CLAUDE.md (v0.1.0-RC.1)
- SciRS2 Core Module Usage Guidelines
- ToRSh SciRS2 Integration Policy v2.0 (proven patterns)
**Next Review**: Q1 2026
**Owner**: Sklears Architecture Team
