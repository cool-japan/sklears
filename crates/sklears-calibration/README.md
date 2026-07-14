# sklears-calibration

[![Crates.io](https://img.shields.io/crates/v/sklears-calibration.svg)](https://crates.io/crates/sklears-calibration)
[![Documentation](https://docs.rs/sklears-calibration/badge.svg)](https://docs.rs/sklears-calibration)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (July 14, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-calibration` provides probability calibration tools, matching scikit-learn’s calibration module with additional Rust-centric performance improvements.

## Key Features

- **CalibratedClassifierCV**: Platt scaling, isotonic regression, and temperature scaling strategies.
- **Probability Tools**: Reliability diagrams, Brier score decomposition, and calibration curve generation.
- **Integration**: Works with sklears pipelines, model selection, and inspection modules.
- **GPU Support**: Optional oxicuda-backed acceleration (`gpu` feature) for device enumeration, memory queries, and temperature-scaling calibration on large-scale workloads; honestly reports "no device" when the feature is off or no GPU is present. `GpuTemperatureScalingCalibrator`'s device sigmoid path only exists as an f32 kernel (no faithful f64 form in the oxicuda stack), so the device fast path is only taken under an explicit `use_mixed_precision` opt-in; without it — or with no device present or a small batch — prediction runs the exact CPU f64 path instead of risking a precision mismatch.

## Quick Start

```rust
use sklears_calibration::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::traits::{Fit, PredictProba};

let calibrator = CalibratedClassifierCV::new()
    .method(CalibrationMethod::Sigmoid)
    .cv(5);

let fitted = calibrator.fit(&x_train, &y_train)?;
let probas = fitted.predict_proba(&x_test)?;
```

## Status

- Covered by 406 passing tests in `0.2.0` (Stable), verified with `cargo nextest run -p sklears-calibration --all-features`.
- API parity with scikit-learn 1.5, including multi-class calibration.
- Future work (Bayesian calibration, streaming reliability) tracked in this crate’s `TODO.md`.
