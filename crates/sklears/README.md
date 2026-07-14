# sklears (crate)

[![Crates.io](https://img.shields.io/crates/v/sklears.svg)](https://crates.io/crates/sklears)
[![Documentation](https://docs.rs/sklears/badge.svg)](https://docs.rs/sklears)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

This crate exposes the top-level `sklears` API that bundles all subcrates into a cohesive, scikit-learn compatible experience. It re-exports `sklears-core`/`sklears-utils` at the crate root and exposes each optional algorithm subcrate as a feature-gated module.

## Key Features

- **Per-Module Re-exports**: Each enabled feature exposes its subcrate as a named module (`sklears::linear`, `sklears::tree`, `sklears::neighbors`, ...) — there is no single `prelude` glob import.
- **Feature Flags**: Enable only the modules you need (`linear`, `ensemble`, `gpu`, etc.) to keep builds lightweight.
- **Rust + Python**: Designed to work with both native Rust projects and the `sklears-python` bindings.
- **Documentation Hub**: Acts as the canonical entry point for examples, tutorials, and integration guides.

## Quick Start

```toml
[dependencies]
sklears = { version = "0.2.0", features = ["linear", "ensemble", "gpu"] }
```

```rust
use sklears::tree::RandomForestClassifier;
use sklears::traits::{Fit, Predict};
use scirs2_core::ndarray::{array, Array1};

let x = array![
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
];
let y = Array1::from(vec![0, 1, 1]);

let model = RandomForestClassifier::new()
    .n_estimators(200)
    .fit(&x, &y)?;

let predictions = model.predict(&x)?;
```

Note: this crate is a facade — it has no `prelude` module. Each enabled
feature re-exports its subcrate under its own name (`sklears::linear`,
`sklears::tree`, `sklears::ensemble`, ...); import types from the specific
module for the algorithms you enabled.

## Status

- Serves as the umbrella facade crate, re-exporting `sklears-core`/`sklears-utils` plus one module per optional algorithm-category feature (`linear`, `clustering`, `ensemble`, `tree`, `neighbors`, `naive-bayes`, `multiclass`, `semi-supervised`, and more — see `Cargo.toml`'s `[features]` for the full list).
- GPU acceleration is available behind the `gpu` feature (OxiCUDA/CUDA-backed via `sklears-core::gpu`), forwarding to each enabled subcrate's own `gpu` feature with an honest CPU fallback when no device is detected.
- This crate's own integration/property tests: 32 passing (13 skipped) with `--all-features` for `0.2.0`; each subcrate carries its own, much larger, unit-test suite documented in its own README.
- The `preprocessing` feature and its `sklears-preprocessing` optional dependency (previously commented out workspace-wide) are restored in `0.2.0`, along with the 8 algorithm-showcase examples and the `tree_ensemble_benchmarks` bench target that require it (`linear_models_showcase`, `lasso_regression`, `kmeans_clustering`, `dbscan_clustering`, `hierarchical_clustering`, `mean_shift_clustering`, `spectral_clustering`, `gmm_clustering`, plus `performance_comparison_comprehensive`).
- Re-export map kept in sync with individual module READMEs and documentation.
- Further enhancements (module-level doc consolidation, feature flag audits) tracked in `TODO.md`.
