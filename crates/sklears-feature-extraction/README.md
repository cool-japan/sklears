# sklears-feature-extraction

[![Crates.io](https://img.shields.io/crates/v/sklears-feature-extraction.svg)](https://crates.io/crates/sklears-feature-extraction)
[![Documentation](https://docs.rs/sklears-feature-extraction/badge.svg)](https://docs.rs/sklears-feature-extraction)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.1` (Unreleased). See the [workspace release notes](../../docs/releases/0.2.1.md) for highlights and upgrade guidance.

## Overview

`sklears-feature-extraction` contains text, signal, and image feature transformers designed to mirror scikit-learn’s feature extraction API with Rust-first performance.

## Key Features

- **Text Processing**: CountVectorizer, TfidfVectorizer, HashingVectorizer, N-gram analyzers, character models.
- **Image Features**: Patch extraction, HOG descriptors, a full SIFT keypoint pipeline, and SIMD-accelerated pixel/statistical operations; `image::simd_accelerated` genuinely delegates to `scirs2_core::simd_ops::SimdUnifiedOps` for vectorized execution (previously several of its functions were plain scalar loops mislabeled as SIMD). There is no GPU backend in this crate.
- **Signal Features**: Windowed statistics, spectrograms, wavelet transforms, and FFT-based descriptors.
- **Pipeline Support**: Integrates with sklears preprocessing, selection, and model selection crates.

## Quick Start

```rust
use sklears_feature_extraction::text::TfidfVectorizer;

let docs = vec![
    "Rust brings fearless concurrency".to_string(),
    "Machine learning in Rust is fast".to_string(),
];

let mut vectorizer = TfidfVectorizer::new()
    .ngram_range((1, 2))
    .min_df(1)
    .max_features(Some(4096));

let tfidf = vectorizer.fit_transform(&docs)?;
```

## Status

- Extensively tested; 416 passing crate tests in `0.2.0`.
- Offers >99% parity with scikit-learn’s feature extraction module, with SIMD-accelerated hot paths (no GPU backend).
- This session fixed `image::simd_accelerated`: 7 of its 9 public functions were labeled "SIMD-accelerated" (with fabricated speedup figures up to "8.7x" in their own doc comments) despite being plain scalar loops — e.g. `simd_array_subtraction` was literally `a - b`; all 7 now genuinely delegate to `scirs2_core::simd_ops::SimdUnifiedOps`, verified numerically equivalent to the prior behavior to 1e-9 tolerance. This audit was scoped to that one module — other `image::` files (`patch_extraction`, `wavelet_features`, `surf_features`) still carry legacy "SIMD-accelerated"/speedup doc comments that were not verified against their implementations in this pass.
- Additional work (streaming text ingestion, audio-specific transforms) documented in `TODO.md`.
