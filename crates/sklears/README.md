# sklears (crate)

[![Crates.io](https://img.shields.io/crates/v/sklears.svg)](https://crates.io/crates/sklears)
[![Documentation](https://docs.rs/sklears/badge.svg)](https://docs.rs/sklears)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

This crate exposes the top-level `sklears` API that bundles all subcrates into a cohesive, scikit-learn compatible experience. It provides re-exports, prelude shortcuts, and integrated feature flag management.

## Key Features

- **Unified Prelude**: Access estimators, transformers, pipelines, and utilities through a single import.
- **Feature Flags**: Enable only the modules you need (`linear`, `ensemble`, `gpu`, etc.) to keep builds lightweight.
- **Rust + Python**: Designed to work with both native Rust projects and the `sklears-python` bindings.
- **Documentation Hub**: Acts as the canonical entry point for examples, tutorials, and integration guides.

## Quick Start

```toml
[dependencies]
sklears = { version = "0.1.0-beta.1", features = ["linear", "ensemble", "gpu"] }
```

```rust
use sklears::prelude::*;
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

## Status

- Serves as the umbrella crate validated through the 11,292 passing workspace tests for `0.1.0-beta.1`.
- Re-export map kept in sync with individual module READMEs and documentation.
- Further enhancements (module-level doc consolidation, feature flag audits) tracked in `TODO.md`.
