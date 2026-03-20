# Sklears Calibration Examples

This directory contains practical examples demonstrating how to use the sklears-calibration crate for probability calibration.

## Running the Examples

You can run any example using:

```bash
cargo run --example <example_name>
```

For example:
```bash
cargo run --example quickstart
cargo run --example multiclass_calibration
cargo run --example performance_comparison
```

## Quick Navigation

| Example | Description | Level | Time |
|---------|-------------|-------|------|
| `quickstart.rs` ⭐ | Quick introduction to calibration | Beginner | < 1 min |
| `basic_calibration.rs` | Comprehensive binary calibration | Beginner | 1-2 min |
| `multiclass_calibration.rs` | Multiclass methods | Intermediate | 2-3 min |
| `bayesian_uncertainty.rs` | Bayesian & uncertainty methods | Advanced | 3-5 min |
| `streaming_calibration.rs` | Online/streaming calibration | Advanced | 2-3 min |
| `performance_comparison.rs` 📊 | Benchmarking tool | All Levels | 5-10 min |

## Available Examples

### `quickstart.rs` ⭐
**Recommended starting point** - A simple introduction showing:
- How to create a calibrated classifier
- Basic API usage with the builder pattern
- Comparing multiple calibration methods
- Quick validation that everything works

**Run with:** `cargo run --example quickstart`

### `basic_calibration.rs`
A comprehensive example demonstrating:
- Various calibration methods (Sigmoid, Isotonic, Temperature Scaling, etc.)
- How to use the builder pattern for configuration
- Working with binary classification
- Basic calibration API workflow

**Run with:** `cargo run --example basic_calibration`

### `multiclass_calibration.rs`
Advanced multiclass calibration demonstration:
- Multiclass-specific methods (Temperature, Matrix Scaling, Dirichlet)
- Performance comparison across methods
- Class distribution analysis
- Method selection guidance for multiclass problems

**Run with:** `cargo run --example multiclass_calibration`

### `bayesian_uncertainty.rs`
Bayesian methods and uncertainty quantification:
- Bayesian Model Averaging
- Variational Inference
- MCMC calibration
- Hierarchical Bayesian methods
- Dirichlet Process
- Conformal prediction
- Gaussian Process calibration
- Method selection guide for uncertainty quantification

**Run with:** `cargo run --example bayesian_uncertainty`

### `streaming_calibration.rs`
Online and streaming calibration methods:
- Online Sigmoid (incremental Platt scaling)
- Adaptive Online (concept drift detection)
- Incremental Update methods
- Batch vs streaming comparison
- Throughput analysis
- Use case recommendations

**Run with:** `cargo run --example streaming_calibration`

### `performance_comparison.rs` 📊
Comprehensive performance benchmarking tool:
- Training time measurements across methods
- Throughput analysis (samples/second)
- Scalability testing with different dataset sizes
- Growth factor analysis
- Performance recommendations

**Run with:** `cargo run --example performance_comparison`

## What is Probability Calibration?

Probability calibration is the process of adjusting the predicted probabilities from a classifier to better match the true likelihood of outcomes. A well-calibrated classifier means that when it predicts a probability of 0.7 for an event, that event should occur approximately 70% of the time.

### Why Calibrate?

1. **Decision Making**: Better calibrated probabilities lead to better decisions
2. **Risk Assessment**: More reliable confidence estimates
3. **Model Comparison**: Fair comparison across different models
4. **Threshold Selection**: More effective classification threshold tuning

### Available Calibration Methods

The sklears-calibration crate provides numerous calibration methods:

#### Basic Methods
- **Sigmoid (Platt Scaling)**: Fits a logistic regression to calibrate probabilities
- **Isotonic Regression**: Non-parametric method preserving monotonicity
- **Temperature Scaling**: Scales logits with a temperature parameter
- **Histogram Binning**: Bins predictions and calibrates per bin
- **Bayesian Binning Quantiles (BBQ)**: Bayesian approach to binning

#### Advanced Methods
- **Beta Calibration**: Uses beta distribution for flexible calibration
- **Kernel Density Estimation (KDE)**: Non-parametric density-based calibration
- **Gaussian Process (GP)**: Probabilistic calibration with uncertainty
- **Bayesian Model Averaging**: Combines multiple calibration models
- **Neural Calibration**: Deep learning-based calibration
- **Conformal Prediction**: Distribution-free uncertainty quantification

#### Specialized Methods
- **Local KNN**: Calibrates based on local neighborhoods
- **Dirichlet/Matrix Scaling**: For multiclass problems
- **Time Series Calibration**: For temporal data
- **Online/Streaming**: For real-time applications

## Usage Pattern

The typical workflow for using calibration in sklears:

```rust
use sklears_calibration::{CalibrationMethod, CalibratedClassifierCV};
use sklears_core::traits::Fit;

// 1. Create calibrator with chosen method
let calibrator = CalibratedClassifierCV::new()
    .method(CalibrationMethod::Sigmoid)  // Choose calibration method
    .cv(5);                              // Set cross-validation folds

// 2. Fit on your data
let trained = calibrator.fit(&features, &labels)?;

// 3. The calibrator is now ready to calibrate probabilities
// (In practice, you'd use it with a classifier's predictions)
```

## Performance Tips

1. **Cross-Validation**: Use more folds (e.g., 5-10) for better calibration but slower training
2. **Method Selection**:
   - Sigmoid works well for many cases
   - Isotonic for non-monotonic relationships
   - Temperature scaling for neural networks
3. **Data Requirements**: Most methods need at least 100-1000 samples for reliable calibration

## Further Reading

- Check the main crate documentation: `cargo doc --open`
- Read the API documentation for detailed parameter explanations
- See the test files in `src/` for more usage examples
- Review the calibration metrics in `metrics.rs` for evaluation tools

## Contributing Examples

If you have a useful calibration example, please consider contributing it!
Examples should be:
- Self-contained and runnable
- Well-commented
- Demonstrate a specific use case or feature
- Include realistic data generation or usage patterns
