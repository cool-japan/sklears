# sklears-feature-extraction

[![Crates.io](https://img.shields.io/crates/v/sklears-feature-extraction.svg)](https://crates.io/crates/sklears-feature-extraction)
[![Documentation](https://docs.rs/sklears-feature-extraction/badge.svg)](https://docs.rs/sklears-feature-extraction)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](../../LICENSE)
[![Minimum Rust Version](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://www.rust-lang.org)

> **Latest release:** `0.1.0-alpha.2` (December 22, 2025). See the [workspace release notes](../../docs/releases/0.1.0-alpha.2.md) for highlights and upgrade guidance.

## Overview

`sklears-feature-extraction` contains text, signal, and image feature transformers designed to mirror scikit-learn’s feature extraction API with Rust-first performance.

## Key Features

- **Text Processing**: CountVectorizer, TfidfVectorizer, HashingVectorizer, N-gram analyzers, character models.
- **Image Features**: Patch extraction, HOG descriptors, SIFT-like outlines, and GPU pipelines.
- **Signal Features**: Windowed statistics, spectrograms, wavelet transforms, and FFT-based descriptors.
- **Pipeline Support**: Integrates with sklears preprocessing, selection, and model selection crates.

## Quick Start

```rust
use sklears_feature_extraction::text::TfidfVectorizer;

let docs = vec![
    "Rust brings fearless concurrency",
    "Machine learning in Rust is fast",
];

let vectorizer = TfidfVectorizer::builder()
    .ngram_range((1, 2))
    .min_df(1)
    .max_features(Some(4096))
    .build();

let tfidf = vectorizer.fit_transform(&docs)?;
```

## Status

- Extensively tested via the 11,292 passing workspace suites shipped in `0.1.0-alpha.2`.
- Offers >99% parity with scikit-learn’s feature extraction module, plus GPU paths.
- Additional work (streaming text ingestion, audio-specific transforms) documented in `TODO.md`.
