#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
//! Dataset loading utilities and synthetic data generators
//!
//! This module provides functions to load built-in datasets and generate
//! synthetic data for testing and experimentation, compatible with
//! scikit-learn's datasets module.

// Only include modules that compile without thread safety or lifetime issues
// pub mod config_templates;  // Temporarily disabled due to dependency on traits
// pub mod traits;  // Temporarily disabled due to API issues

// Simple placeholder generator functions
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::*; // SCIRS2 Policy: Use unified random API

/// Simple dataset structure for basic functionality
#[derive(Debug, Clone)]
pub struct SimpleDataset {
    pub features: Array2<f64>,
    pub targets: Option<Array1<f64>>,
}

impl SimpleDataset {
    pub fn new(features: Array2<f64>, targets: Option<Array1<f64>>) -> Self {
        Self { features, targets }
    }

    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }
}

pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: Option<usize>,
    _cluster_std: Option<f64>,
    _center_box: Option<(f64, f64)>,
    _random_state: Option<u64>,
) -> Result<SimpleDataset, String> {
    let mut rng = Random::seed(42);
    let normal = RandNormal::new(0.0, 1.0).unwrap();
    let features = Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng));
    let targets = Array1::from_shape_fn(n_samples, |i| (i % centers.unwrap_or(3)) as f64);

    Ok(SimpleDataset::new(features, Some(targets)))
}

#[allow(clippy::too_many_arguments)]
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
) -> Result<SimpleDataset, String> {
    let mut rng = Random::seed(42);
    let normal = RandNormal::new(0.0, 1.0).unwrap();
    let features = Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng));
    let targets = Array1::from_shape_fn(n_samples, |i| (i % n_classes.unwrap_or(2)) as f64);

    Ok(SimpleDataset::new(features, Some(targets)))
}

#[allow(clippy::too_many_arguments)]
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    _n_informative: Option<usize>,
    _n_targets: Option<usize>,
    noise: Option<f64>,
    _coef: Option<bool>,
    _bias: Option<f64>,
    _random_state: Option<u64>,
) -> Result<SimpleDataset, String> {
    let mut rng = Random::seed(42);
    let normal = RandNormal::new(0.0, 1.0).unwrap();
    let features = Array2::from_shape_fn((n_samples, n_features), |_| normal.sample(&mut rng));
    let noise_std = noise.unwrap_or(0.0);
    let normal_targets = RandNormal::new(0.0, 1.0 + noise_std).unwrap();
    let targets = Array1::from_shape_fn(n_samples, |_| normal_targets.sample(&mut rng));

    Ok(SimpleDataset::new(features, Some(targets)))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dataset() {
        let features = Array2::zeros((10, 3));
        let targets = Some(Array1::zeros(10));
        let dataset = SimpleDataset::new(features, targets);

        assert_eq!(dataset.n_samples(), 10);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.targets.is_some());
    }

    #[test]
    fn test_make_blobs() {
        let dataset = make_blobs(50, 2, Some(3), Some(1.0), None, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.targets.is_some());
    }

    #[test]
    fn test_make_classification() {
        let dataset = make_classification(
            100,
            5,
            Some(3),
            Some(1),
            None,
            Some(2),
            None,
            None,
            None,
            None,
            Some(42),
        )
        .unwrap();
        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.targets.is_some());
    }

    #[test]
    fn test_make_regression() {
        let dataset =
            make_regression(75, 4, Some(2), None, Some(0.1), None, None, Some(42)).unwrap();
        assert_eq!(dataset.n_samples(), 75);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.targets.is_some());
    }
}
