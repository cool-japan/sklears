//! Dataset loading utilities and synthetic data generators
//!
//! This module provides functions to load built-in datasets and generate
//! synthetic data for testing and experimentation, compatible with
//! scikit-learn's datasets module.

// Essential modules that should compile
pub mod composable;
pub mod config_templates;
pub mod memory;
pub mod memory_pool;
pub mod plugins;
pub mod streaming;
pub mod traits;
pub mod zero_copy;

// Placeholder generator functions for composable.rs compatibility
use crate::traits::InMemoryDataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;

pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: Option<usize>,
    cluster_std: Option<f64>,
    _center_box: Option<(f64, f64)>,
    _random_state: Option<u64>,
) -> Result<InMemoryDataset, String> {
    let mut rng = Random::new(Some(42));
    let features = Array2::from_shape_fn((n_samples, n_features), |_| rng.sample_normal(0.0, 1.0));
    let targets = Array1::from_shape_fn(n_samples, |i| (i % centers.unwrap_or(3)) as f64);

    Ok(InMemoryDataset {
        features,
        targets: Some(targets),
        feature_names: None,
        target_names: None,
    })
}

pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    _n_informative: Option<usize>,
    _n_redundant: Option<usize>,
    _n_repeated: Option<usize>,
    n_classes: Option<usize>,
    _n_clusters_per_class: Option<usize>,
    _weights: Option<Vec<f64>>,
    _flip_y: Option<f64>,
    _class_sep: Option<f64>,
    _random_state: Option<u64>,
) -> Result<InMemoryDataset, String> {
    let mut rng = Random::new(Some(42));
    let features = Array2::from_shape_fn((n_samples, n_features), |_| rng.sample_normal(0.0, 1.0));
    let targets = Array1::from_shape_fn(n_samples, |i| (i % n_classes.unwrap_or(2)) as f64);

    Ok(InMemoryDataset {
        features,
        targets: Some(targets),
        feature_names: None,
        target_names: None,
    })
}

pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    _n_informative: Option<usize>,
    _n_targets: Option<usize>,
    noise: Option<f64>,
    _coef: Option<bool>,
    _bias: Option<f64>,
    _random_state: Option<u64>,
) -> Result<InMemoryDataset, String> {
    let mut rng = Random::new(Some(42));
    let features = Array2::from_shape_fn((n_samples, n_features), |_| rng.sample_normal(0.0, 1.0));
    let noise_std = noise.unwrap_or(0.0);
    let targets = Array1::from_shape_fn(n_samples, |_| rng.sample_normal(0.0, 1.0 + noise_std));

    Ok(InMemoryDataset {
        features,
        targets: Some(targets),
        feature_names: None,
        target_names: None,
    })
}

pub fn make_circles(
    n_samples: usize,
    _shuffle: Option<bool>,
    noise: Option<f64>,
    _factor: Option<f64>,
    _random_state: Option<u64>,
) -> Result<InMemoryDataset, String> {
    let mut rng = Random::new(Some(42));
    let noise_std = noise.unwrap_or(0.0);
    let features = Array2::from_shape_fn((n_samples, 2), |_| rng.sample_normal(0.0, 1.0 + noise_std));
    let targets = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

    Ok(InMemoryDataset {
        features,
        targets: Some(targets),
        feature_names: None,
        target_names: None,
    })
}

pub fn make_moons(
    n_samples: usize,
    _shuffle: Option<bool>,
    noise: Option<f64>,
    _random_state: Option<u64>,
) -> Result<InMemoryDataset, String> {
    let mut rng = Random::new(Some(42));
    let noise_std = noise.unwrap_or(0.0);
    let features = Array2::from_shape_fn((n_samples, 2), |_| rng.sample_normal(0.0, 1.0 + noise_std));
    let targets = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

    Ok(InMemoryDataset {
        features,
        targets: Some(targets),
        feature_names: None,
        target_names: None,
    })
}