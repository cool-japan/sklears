# sklears-simd

[![Crates.io](https://img.shields.io/crates/v/sklears-simd.svg)](https://crates.io/crates/sklears-simd)
[![Documentation](https://docs.rs/sklears-simd/badge.svg)](https://docs.rs/sklears-simd)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-beta.1` (January 1, 2026). See the [workspace release notes](../../docs/releases/0.1.0-beta.1.md) for highlights and upgrade guidance.

## Overview

`sklears-simd` exposes low-level SIMD, GPU, and hardware acceleration utilities used across the sklears ecosystem. While primarily an internal crate, it is documented for contributors building new high-performance components.

## Key Features

- **Vector Abstractions**: Portable SIMD types (f32x4, f32x8, f32x16) with architecture-specific intrinsics.
- **Alignment & Memory**: Alignment helpers, prefetching hints, cache-aware allocation strategies.
- **GPU Bridges**: CUDA/WebGPU adapters, Tensor Core pathways, and multi-GPU orchestration helpers.
- **Benchmark Harnesses**: Criterion-based benchmarks and profiling utilities for micro-optimizations.

## Quick Peek

```rust
use sklears_simd::vector::F32x4;

let a = F32x4::new(1.0, 2.0, 3.0, 4.0);
let b = F32x4::splat(2.0);
let result = a.mul(b);
assert_eq!(result.horizontal_sum(), 20.0);
```

## Status

- Core building block validated by the 11,292 passing workspace tests for `0.1.0-beta.1`.
- Powers SIMD/GPU acceleration in linear models, neighbors, metrics, and more.
- Contributor roadmap (new architectures, auto-vectorization tooling) maintained in `TODO.md`.
