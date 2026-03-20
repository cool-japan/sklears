#!/bin/bash

# Profile-Guided Optimization (PGO) script for sklears-simd
# This script demonstrates how to use PGO to optimize the SIMD operations

set -e

echo "Starting Profile-Guided Optimization for sklears-simd"

# Step 1: Build with PGO generation enabled
echo "Step 1: Building with PGO generation..."
export RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data"
cargo build --profile pgo-gen --all-features

# Step 2: Run representative workloads to generate profile data
echo "Step 2: Running benchmarks to generate profile data..."
cargo bench --profile pgo-gen --all-features

# Step 3: Run tests to ensure more code coverage
echo "Step 3: Running tests for additional coverage..."
cargo test --profile pgo-gen --all-features

# Step 4: Build with PGO usage enabled
echo "Step 4: Building with PGO usage..."
export RUSTFLAGS="-Cprofile-use=/tmp/pgo-data -Cllvm-args=-pgo-warn-missing-function"
cargo build --profile pgo-use --all-features

echo "PGO optimization complete!"
echo "The optimized binary is available in target/pgo-use/"
echo ""
echo "To use PGO-optimized version in your application:"
echo "cargo build --profile pgo-use --all-features"