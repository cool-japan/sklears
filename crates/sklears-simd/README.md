# sklears-simd

[![Crates.io](https://img.shields.io/crates/v/sklears-simd.svg)](https://crates.io/crates/sklears-simd)
[![Documentation](https://docs.rs/sklears-simd/badge.svg)](https://docs.rs/sklears-simd)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-simd` exposes low-level SIMD, GPU, and hardware acceleration utilities used across the sklears ecosystem. While primarily an internal crate, it is documented for contributors building new high-performance components.

## Key Features

- **Vector Abstractions**: Portable SIMD types (F32x4, F32x8, F32x16) with architecture-specific intrinsics (SSE2, AVX2, AVX512, NEON) and scalar fallbacks.
- **Alignment & Memory**: Alignment helpers, prefetching hints, cache-aware allocation strategies.
- **Accelerator Scaffolding**: Experimental FPGA/TPU/quantum/neuromorphic interface abstractions with CPU SIMD fallback (no vendor hardware backend). Real GPU dispatch lives in `sklears-core`'s oxicuda-backed `gpu` module, not in this crate.
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

- Core building block for `0.2.0`; 518 passing crate tests (4 skipped, `--features parallel`).
- Powers CPU SIMD acceleration in linear models, neighbors, metrics, and more; GPU dispatch is handled by `sklears-core`.
- Contributor roadmap (new architectures, auto-vectorization tooling) maintained in `TODO.md`.
