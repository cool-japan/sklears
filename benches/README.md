# sklears Benchmarking Suite

This directory contains the comprehensive benchmarking suite for sklears, designed to track performance over time and detect regressions.

## Overview

The benchmarking suite includes:
- **Continuous benchmarks**: Track performance of all major algorithms
- **Regression detection**: Automatic detection of performance regressions
- **CI/CD integration**: GitHub Actions workflow for automated benchmarking
- **Performance reports**: Detailed analysis and visualization of results

## Running Benchmarks

### Local Benchmarking

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark:
```bash
cargo bench continuous_benchmarks
```

Run with specific features:
```bash
cargo bench --features "all-algorithms" continuous_benchmarks
```

### Analyzing Results

Use the benchmark analysis script:
```bash
python scripts/benchmark_analysis.py
```

Compare with baseline:
```bash
python scripts/benchmark_analysis.py --baseline-dir /path/to/baseline/criterion
```

Generate JSON report:
```bash
python scripts/benchmark_analysis.py --format json --output results.json
```

## Benchmark Structure

### Algorithm Categories

1. **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet
2. **Tree Models**: DecisionTree, RandomForest, ExtraTrees
3. **Ensemble Methods**: GradientBoosting, AdaBoost, VotingClassifier
4. **Preprocessing**: StandardScaler, MinMaxScaler, PCA, ICA
5. **Clustering**: KMeans, DBSCAN, Hierarchical, Spectral
6. **Neural Networks**: MLPClassifier, MLPRegressor
7. **Distance Calculations**: Euclidean, Manhattan, Cosine

### Performance Metrics

Each benchmark measures:
- **Mean execution time**: Average time across iterations
- **Standard deviation**: Variability in performance
- **Median time**: Middle value of all measurements
- **Throughput**: Operations per second

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/benchmark.yml`) automatically:
1. Runs benchmarks on every push to main
2. Compares PR performance against baseline
3. Comments on PRs with performance changes
4. Alerts on significant regressions (>200% slower)
5. Stores benchmark history for trend analysis

## Performance Guidelines

### Regression Thresholds

- **Warning**: 10-20% performance degradation
- **Alert**: >20% performance degradation
- **Critical**: >200% performance degradation

### Optimization Targets

Based on comparison with scikit-learn:
- Linear models: 3-5x faster
- Tree models: 10-20x faster  
- Neural networks: 5-10x faster
- Preprocessing: 3-6x faster

## Adding New Benchmarks

To add a new benchmark:

1. Add the algorithm to `continuous_benchmarks.rs`:
```rust
fn benchmark_my_algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("My Algorithm");
    // ... benchmark code
}
```

2. Add to criterion_group:
```rust
criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = 
        // ... existing benchmarks
        benchmark_my_algorithm
}
```

3. Update the analysis script if needed for custom metrics

## Best Practices

1. **Consistent environment**: Run benchmarks on the same hardware
2. **Warm up**: Allow CPU to reach steady state before measuring
3. **Multiple iterations**: Use sufficient samples for statistical significance
4. **Isolated runs**: Close other applications during benchmarking
5. **Profile-guided optimization**: Use benchmark results to guide optimization

## Troubleshooting

### High variance in results
- Increase sample size in `configure_criterion()`
- Check for background processes
- Ensure CPU frequency scaling is disabled

### Missing benchmark results
- Verify all features are enabled
- Check that algorithms are properly imported
- Ensure criterion output directory exists

### CI benchmark failures
- Check workflow logs for compilation errors
- Verify BLAS/LAPACK dependencies are installed
- Review regression thresholds in workflow

## Future Enhancements

1. **Memory profiling**: Track memory usage alongside performance
2. **SIMD benchmarks**: Specific tests for vectorized operations
3. **GPU benchmarks**: Performance tests for CUDA/WebGPU backends
4. **Scalability tests**: Performance vs data size analysis
5. **Real-world datasets**: Benchmarks on actual ML datasets