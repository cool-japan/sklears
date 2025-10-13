#!/bin/bash

# Comprehensive benchmark runner for sklears-svm
# This script runs all performance benchmarks and generates reports

set -e

echo "================================="
echo "sklears-svm Benchmark Suite"
echo "================================="
echo

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -f "benches/libsvm_comparison.rs" ]]; then
    echo "Error: Please run this script from the sklears-svm crate root directory"
    exit 1
fi

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$RESULTS_DIR/benchmark_report_$TIMESTAMP.md"

# Function to run benchmark and capture output
run_benchmark() {
    local bench_name=$1
    local description=$2

    echo "Running $description..."
    echo "## $description" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    # Run benchmark and capture output
    if cargo bench --bench "$bench_name" -- --output-format json | tee "$RESULTS_DIR/${bench_name}_$TIMESTAMP.json"; then
        echo "✅ $description completed successfully"
        echo "Results saved to $RESULTS_DIR/${bench_name}_$TIMESTAMP.json"
    else
        echo "❌ $description failed"
        echo "**Benchmark failed**" >> "$REPORT_FILE"
    fi

    echo "" >> "$REPORT_FILE"
    echo "---" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# Initialize report
cat > "$REPORT_FILE" << EOF
# sklears-svm Performance Benchmark Report

Generated on: $(date)
Rust version: $(rustc --version)
CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs || echo "Unknown")
Memory: $(grep MemTotal /proc/meminfo | awk '{print $2 " " $3}' || echo "Unknown")

## Summary

This report contains comprehensive performance benchmarks comparing sklears-svm
against reference implementations and measuring various performance characteristics.

EOF

echo "Starting benchmark suite..."
echo "Report will be saved to: $REPORT_FILE"
echo

# Run libsvm comparison benchmarks
echo "📊 Running libSVM comparison benchmarks..."
run_benchmark "libsvm_comparison" "libSVM Performance Comparison"

echo
echo "📊 Running scikit-learn comparison benchmarks..."
run_benchmark "sklearn_comparison" "scikit-learn Performance Comparison"

# Run additional benchmark configurations
echo
echo "📊 Running feature-specific benchmarks..."

# SIMD benchmarks (if feature is enabled)
if cargo check --features simd &>/dev/null; then
    echo "Running SIMD-optimized benchmarks..."
    run_benchmark "libsvm_comparison" "SIMD-Optimized Performance" --features simd
fi

# GPU benchmarks (if feature is enabled)
if cargo check --features gpu &>/dev/null; then
    echo "Running GPU-accelerated benchmarks..."
    run_benchmark "libsvm_comparison" "GPU-Accelerated Performance" --features gpu
fi

# Parallel benchmarks
echo "Running parallel processing benchmarks..."
run_benchmark "libsvm_comparison" "Parallel Processing Performance" --features parallel

# Generate summary
echo
echo "📈 Generating performance summary..."

cat >> "$REPORT_FILE" << 'EOF'

## Performance Targets

sklears-svm aims to achieve the following performance improvements over reference implementations:

- **Training Speed**: 2-10x faster than scikit-learn SVMs
- **Prediction Speed**: 3-15x faster than scikit-learn SVMs
- **Memory Usage**: 20-40% reduction compared to scikit-learn
- **Scalability**: Better performance on large datasets (>10k samples)

## Key Optimizations

1. **Efficient SMO Algorithm**: Optimized Sequential Minimal Optimization with advanced working set selection
2. **SIMD Kernels**: Vectorized kernel computations using AVX2/SSE2/NEON
3. **Parallel Processing**: Multi-threaded training and prediction using rayon
4. **Memory Management**: Efficient caching and memory-mapped kernel matrices
5. **GPU Acceleration**: WGPU-based GPU kernels for large-scale problems

## Benchmark Categories

### Training Benchmarks
- SMO algorithm performance across different dataset sizes
- Kernel computation efficiency (Linear, RBF, Polynomial, Sigmoid)
- Memory usage patterns and optimization
- Parallel processing scalability

### Prediction Benchmarks
- Prediction throughput across different model sizes
- Kernel evaluation performance
- Memory efficiency during prediction

### Comparison Benchmarks
- Performance relative to libsvm reference implementation
- Performance relative to scikit-learn patterns
- Accuracy validation against reference implementations

### Optimization Benchmarks
- Different solver algorithms (SMO, Dual Coordinate Ascent, ADMM)
- SIMD vs regular kernel computations
- GPU vs CPU kernel computations
- Parallel vs serial processing

## Usage Instructions

To run these benchmarks yourself:

```bash
# Run all benchmarks
./scripts/run_benchmarks.sh

# Run specific benchmark
cargo bench --bench libsvm_comparison

# Run with specific features
cargo bench --bench libsvm_comparison --features "simd parallel"

# Run with release optimizations
cargo bench --release --bench sklearn_comparison
```

## Interpreting Results

Criterion.rs provides detailed timing information including:
- Mean execution time with confidence intervals
- Throughput measurements (samples/second)
- Performance regression detection
- Statistical significance tests

Look for:
- **Training throughput**: Higher is better (samples/second)
- **Prediction latency**: Lower is better (microseconds)
- **Memory efficiency**: Lower peak memory usage
- **Scalability**: Performance characteristics as dataset size increases

EOF

echo
echo "✅ Benchmark suite completed!"
echo "📊 Results summary:"
echo "   - Full report: $REPORT_FILE"
echo "   - Raw data: $RESULTS_DIR/*_$TIMESTAMP.json"
echo
echo "📈 Performance highlights:"
echo "   - Training speed improvements: Check libsvm_comparison results"
echo "   - Prediction throughput: Check prediction benchmark sections"
echo "   - Memory efficiency: Check memory_usage benchmark results"
echo "   - Scalability: Compare performance across different dataset sizes"
echo

# Optional: Open report if running in a graphical environment
if command -v xdg-open >/dev/null 2>&1; then
    echo "Opening report in default viewer..."
    xdg-open "$REPORT_FILE" &
elif command -v open >/dev/null 2>&1; then
    echo "Opening report in default viewer..."
    open "$REPORT_FILE" &
fi

echo "Benchmark suite complete! 🎉"