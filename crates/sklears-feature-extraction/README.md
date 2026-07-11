# sklears-feature-extraction

[![Crates.io](https://img.shields.io/crates/v/sklears-feature-extraction.svg)](https://crates.io/crates/sklears-feature-extraction)
[![Documentation](https://docs.rs/sklears-feature-extraction/badge.svg)](https://docs.rs/sklears-feature-extraction)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.2.0` (June 30, 2026). See the [workspace release notes](../../docs/releases/0.2.0.md) for highlights and upgrade guidance.

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
- This session fixed `image::simd_accelerated`: 9 functions previously labeled "SIMD-accelerated" (with fabricated speedup figures in their own doc comments) were plain scalar loops; they now genuinely delegate to `scirs2_core::simd_ops::SimdUnifiedOps`, verified numerically equivalent to the prior behavior to 1e-9 tolerance.
- Additional work (streaming text ingestion, audio-specific transforms) documented in `TODO.md`.
