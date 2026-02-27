# sklears-metrics

[![Crates.io](https://img.shields.io/crates/v/sklears-metrics.svg)](https://crates.io/crates/sklears-metrics)
[![Documentation](https://docs.rs/sklears-metrics/badge.svg)](https://docs.rs/sklears-metrics)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

Comprehensive, high-performance evaluation metrics for machine learning in Rust, offering 10-50x speedup over scikit-learn with GPU acceleration support.

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-metrics` provides a complete suite of evaluation metrics including:

- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and more
- **Regression Metrics**: MSE, MAE, R², MAPE, Huber, Quantile regression metrics
- **Clustering Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz, V-measure
- **Advanced Features**: GPU acceleration, uncertainty quantification, streaming computation
- **Specialized Domains**: Computer vision, NLP, survival analysis, time series

## Quick Start

```rust
use sklears_metrics::{accuracy_score, precision_recall_fscore, roc_auc_score};
use ndarray::array;

// Basic classification metrics
let y_true = array![0, 1, 1, 0, 1, 0];
let y_pred = array![0, 1, 0, 0, 1, 1];

let acc = accuracy_score(&y_true, &y_pred)?;
let (precision, recall, f1) = precision_recall_fscore(&y_true, &y_pred)?;
let auc = roc_auc_score(&y_true, &y_pred)?;

println!("Accuracy: {:.2}", acc);
println!("Precision: {:.2}, Recall: {:.2}, F1: {:.2}", precision, recall, f1);
println!("ROC-AUC: {:.2}", auc);
```

## Features

### Core Capabilities

- **Comprehensive Coverage**: 100+ metrics across all ML domains
- **Type Safety**: Compile-time validation with phantom types
- **Performance**: SIMD optimizations, GPU acceleration, parallel processing
- **Memory Efficiency**: Streaming metrics, compressed storage, lazy evaluation
- **Production Ready**: 393/393 crate tests passing, plus inclusion in the 11,292 passing workspace checks for 0.1.0-beta.1

### Advanced Features

#### GPU Acceleration
```rust
use sklears_metrics::gpu::{GpuMetricsContext, gpu_accuracy};

let ctx = GpuMetricsContext::new()?;
let accuracy = gpu_accuracy(&ctx, &y_true_gpu, &y_pred_gpu)?;
```

#### Uncertainty Quantification
```rust
use sklears_metrics::uncertainty::{bootstrap_confidence_interval, conformal_prediction};

let (lower, upper) = bootstrap_confidence_interval(&y_true, &y_pred, 0.95)?;
let prediction_sets = conformal_prediction(&calibration_scores, alpha)?;
```

#### Streaming Metrics
```rust
use sklears_metrics::streaming::StreamingMetrics;

let mut metrics = StreamingMetrics::new();
for batch in data_stream {
    metrics.update(&batch.y_true, &batch.y_pred)?;
}
let final_scores = metrics.compute()?;
```

## Performance

Benchmarks show significant improvements:

| Metric | scikit-learn | sklears-metrics | Speedup |
|--------|--------------|-----------------|---------|
| Accuracy | 1.2ms | 0.05ms | 24x |
| ROC-AUC | 8.5ms | 0.3ms | 28x |
| Clustering | 15ms | 0.8ms | 19x |
| GPU Accuracy | N/A | 0.01ms | >100x |

## Specialized Domains

### Computer Vision
```rust
use sklears_metrics::vision::{iou_score, ssim, psnr};

let iou = iou_score(&pred_masks, &true_masks)?;
let similarity = ssim(&pred_image, &true_image)?;
let peak_snr = psnr(&pred_image, &true_image)?;
```

### Natural Language Processing
```rust
use sklears_metrics::nlp::{bleu_score, rouge_scores, perplexity};

let bleu = bleu_score(&hypothesis, &reference)?;
let rouge = rouge_scores(&summary, &reference)?;
let ppl = perplexity(&model_logits, &true_tokens)?;
```

### Time Series
```rust
use sklears_metrics::timeseries::{mase, smape, directional_accuracy};

let mase_score = mase(&y_true, &y_pred, &y_train)?;
let smape_score = smape(&y_true, &y_pred)?;
let da = directional_accuracy(&y_true, &y_pred)?;
```

## Advanced Usage

### Multi-Objective Optimization
```rust
use sklears_metrics::multiobjective::{pareto_frontier, topsis_ranking};

let frontier = pareto_frontier(&objectives)?;
let rankings = topsis_ranking(&alternatives, &weights)?;
```

### Federated Learning
```rust
use sklears_metrics::federated::{secure_aggregation, privacy_preserving_metrics};

let global_metrics = secure_aggregation(&client_metrics, epsilon)?;
let private_accuracy = privacy_preserving_metrics(&local_data, delta)?;
```

### Calibration
```rust
use sklears_metrics::calibration::{calibration_curve, expected_calibration_error};

let (fraction_positive, mean_predicted) = calibration_curve(&y_true, &y_prob)?;
let ece = expected_calibration_error(&y_true, &y_prob)?;
```

## Architecture

The crate is organized into modules:

```
sklears-metrics/
├── classification/     # Binary and multiclass metrics
├── regression/        # Continuous target metrics
├── clustering/        # Unsupervised evaluation
├── ranking/          # Information retrieval metrics
├── uncertainty/      # Confidence and uncertainty
├── streaming/        # Online and incremental metrics
├── gpu/             # CUDA-accelerated computations
├── visualization/   # Plotting and reporting
└── specialized/     # Domain-specific metrics
```

## Fluent API

```rust
use sklears_metrics::MetricsBuilder;

let results = MetricsBuilder::new()
    .accuracy()
    .precision()
    .recall()
    .f1_score()
    .roc_auc()
    .with_confidence_intervals(0.95)
    .with_gpu_acceleration()
    .compute(&y_true, &y_pred)?;
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE)).

## Citation

```bibtex
@software{sklears_metrics,
  title = {sklears-metrics: High-Performance ML Metrics for Rust},
  author = {COOLJAPAN OU (Team KitaSan)},
  year = {2026},
  url = {https://github.com/cool-japan/sklears}
}
```
