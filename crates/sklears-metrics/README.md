# sklears-metrics

[![Crates.io](https://img.shields.io/crates/v/sklears-metrics.svg)](https://crates.io/crates/sklears-metrics)
[![Documentation](https://docs.rs/sklears-metrics/badge.svg)](https://docs.rs/sklears-metrics)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Comprehensive evaluation metrics for machine learning in Rust, with SIMD/GPU-accelerated kernels available behind optional features.

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

## Overview

`sklears-metrics` provides a broad suite of evaluation metrics including:

- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and more
- **Regression Metrics**: MSE, MAE, R², MAPE, Huber, quantile and time-series regression metrics
- **Clustering Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz, V-measure
- **Advanced Features**: GPU acceleration (`gpu` feature), uncertainty quantification, streaming computation
- **Specialized Domains**: Computer vision, NLP, survival analysis, federated learning, multi-objective ranking

> **Note on module paths**: this crate re-exports only a small set of names at the crate root
> (`scoring::*` and a handful of `classification` display types). Almost everything else —
> including `accuracy_score`, `roc_auc_score`, and all of the domain-specific functions below —
> must be imported through its owning module path (e.g. `sklears_metrics::basic_metrics::accuracy_score`),
> not `sklears_metrics::accuracy_score` directly. The examples below use the real, verified paths.

## Quick Start

```rust
use sklears_metrics::basic_metrics::{accuracy_score, precision_score, recall_score, f1_score};
use sklears_metrics::ranking::roc_auc_score;
use scirs2_core::ndarray::array;

// Basic classification metrics
let y_true = array![0, 1, 1, 0, 1, 0];
let y_pred = array![0, 1, 0, 0, 1, 1];

let acc = accuracy_score(&y_true, &y_pred)?;
let precision = precision_score(&y_true, &y_pred, None)?;
let recall = recall_score(&y_true, &y_pred, None)?;
let f1 = f1_score(&y_true, &y_pred, None)?;

let y_score = array![0.1, 0.8, 0.3, 0.2, 0.9, 0.6];
let auc = roc_auc_score(&y_true, &y_score)?;

println!("Accuracy: {:.2}", acc);
println!("Precision: {:.2}, Recall: {:.2}, F1: {:.2}", precision, recall, f1);
println!("ROC-AUC: {:.2}", auc);
```

## Features

### Core Capabilities

- **Broad Coverage**: metrics spanning classification, regression, clustering, ranking, calibration, and more, organized into per-domain modules
- **Type Safety**: numeric-generic functions (`Float` bound) with explicit `MetricsResult<T>` error handling
- **Performance**: SIMD (`simd` feature), parallel processing (`parallel` feature via rayon), GPU acceleration (`gpu` feature)
- **Memory Efficiency**: streaming metrics via `memory_efficiency::StreamingMetrics`
- **Test Coverage**: see [Status](#status) for the current verified test count

### Advanced Features

#### GPU Acceleration (`gpu` feature)
```rust
use sklears_metrics::gpu_acceleration::{GpuMetricsContext, GpuMetricType};

let mut ctx = GpuMetricsContext::new()?;
let accuracy = ctx.compute_metric(GpuMetricType::Accuracy, &y_true.view(), &y_pred.view())?;
```

#### Uncertainty Quantification
```rust
use sklears_metrics::uncertainty::bootstrap_confidence_interval;
use sklears_metrics::basic_metrics::accuracy_score;

let result = bootstrap_confidence_interval(
    &y_true, &y_pred,
    |t, p| accuracy_score(t, p).unwrap_or(0.0),
    0.95,   // confidence_level
    1000,   // n_bootstrap
    42,     // seed
)?;
```

#### Streaming Metrics
```rust
use sklears_metrics::memory_efficiency::StreamingMetrics;

let mut metrics = StreamingMetrics::new(batch_size);
for batch in data_stream {
    metrics.add_samples(&batch.y_true, &batch.y_pred)?;
}
let final_scores = metrics.finalize();
```

## Performance

SIMD (`simd`), parallel (`parallel`), and GPU (`gpu`) features can be combined via the `optimized`
and `full` feature groups; see `Cargo.toml` for the exact feature graph. Default builds stay
Pure-Rust/CPU-only.

## Specialized Domains

### Computer Vision
```rust
use sklears_metrics::computer_vision::{iou_masks, ssim, psnr};

let iou = iou_masks(&pred_masks, &true_masks)?;
let similarity = ssim(&pred_image, &true_image, 7, 0.01, 0.03, 255.0)?;
let peak_snr = psnr(&pred_image, &true_image, 255.0)?;
```

### Natural Language Processing
```rust
use sklears_metrics::nlp::{bleu_score, rouge_n_score, perplexity, SmoothingFunction};

let weights = [0.25, 0.25, 0.25, 0.25];
let bleu = bleu_score(&hypothesis, &references, &weights, SmoothingFunction::Method1)?;
let rouge_1: f64 = rouge_n_score(&summary, &reference_summaries, 1)?;
let ppl = perplexity(&log_probabilities)?;
```

### Time Series (regression module)
```rust
use sklears_metrics::regression::{
    mean_absolute_scaled_error, symmetric_mean_absolute_percentage_error, mean_directional_accuracy,
};

let mase = mean_absolute_scaled_error(&y_true, &y_pred, &y_train, seasonality)?;
let smape = symmetric_mean_absolute_percentage_error(&y_true, &y_pred)?;
let da = mean_directional_accuracy(&y_true, &y_pred)?;
```

## Advanced Usage

### Multi-Objective Optimization
```rust
use sklears_metrics::multi_objective::{pareto_frontier, topsis_ranking};

let frontier = pareto_frontier(&objectives, &maximize_flags)?;
let rankings = topsis_ranking(&objectives, &weights, &maximize_flags)?;
```

### Federated Learning
```rust
use sklears_metrics::federated_learning::{secure_aggregation, privacy_preserving_aggregation};

let global_metric = secure_aggregation(&client_shares, threshold)?;
let private_metric = privacy_preserving_aggregation(&local_metrics, epsilon, sensitivity, seed)?;
```

### Calibration
```rust
use sklears_metrics::probabilistic_metrics::{reliability_diagram, expected_calibration_error};

let (bin_boundaries, bin_accuracies, bin_confidences, bin_counts) =
    reliability_diagram(&y_true, &y_proba, 10)?;
let ece = expected_calibration_error(&y_true, &y_proba, 10)?;
```

## Architecture

The crate is organized into modules (all declared directly in `src/lib.rs`):

```
sklears-metrics/
├── basic_metrics, classification, ranking, multilabel_metrics  # Classification metrics
├── regression/                                                 # Continuous-target metrics (incl. time series)
├── clustering                                                  # Unsupervised evaluation
├── probabilistic_metrics                                       # Calibration, log-loss, divergences
├── uncertainty                                                 # Confidence intervals, conformal-style utilities
├── memory_efficiency, async_streaming (feature = "async")       # Streaming / online metrics
├── gpu_acceleration (feature = "gpu")                           # OxiCUDA-accelerated computations
├── computer_vision, nlp, survival, temporal, fairness_metrics   # Domain-specific metrics
├── federated_learning, multi_objective, distributed_metrics     # Multi-party / multi-objective metrics
├── visualization, automated_reporting, latex_export (feature = "latex")  # Reporting
└── fluent_api, modular_framework, scoring                       # Builder / composition APIs
```

## Fluent API

```rust
use sklears_metrics::fluent_api::MetricsBuilder;

let results = MetricsBuilder::new()
    .accuracy()
    .precision()
    .recall()
    .f1_score()
    .roc_auc()
    .with_confidence_intervals(true, 0.95, 1000)
    .compute(&y_true, &y_pred)?;
```

## Status

- Covered by 426 passing tests (`cargo nextest run -p sklears-metrics --all-features`, verified 2026-07-11).
- Broad scikit-learn-style metric coverage across classification, regression, clustering, ranking, calibration, and specialized domains.
- Only `scoring::*` and select `classification` display types are re-exported at the crate root — everything else is reached through its owning module (see the examples above).
- Future work (additional GPU-accelerated metric kernels, richer conformal prediction, distributed metrics on non-macOS/MPI targets) tracked in this crate’s `TODO.md`.
