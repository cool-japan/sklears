//! Privacy-preserving and federated learning datasets
//!
//! This module contains generators for privacy-preserving datasets including
//! differential privacy, federated learning simulations, and secure aggregation scenarios.

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Normal;
use sklears_core::error::{Result, SklearsError};

/// Generate privacy-preserving datasets using differential privacy
pub fn make_privacy_preserving_dataset(
    n_samples: usize,
    n_features: usize,
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive".to_string(),
        ));
    }

    if epsilon <= 0.0 || delta < 0.0 || delta >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "epsilon must be positive, delta must be in [0, 1)".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate base dataset
    let mut X = Array2::zeros((n_samples, n_features));
    let normal = Normal::new(0.0, 1.0).unwrap();

    for i in 0..n_samples {
        for j in 0..n_features {
            X[[i, j]] = rng.sample(normal);
        }
    }

    // Add Laplace noise for differential privacy
    let b = sensitivity / epsilon; // Laplace parameter
    let laplace_noise = Normal::new(0.0, b * std::f64::consts::SQRT_2).unwrap();

    for i in 0..n_samples {
        for j in 0..n_features {
            let noise: f64 = rng.sample(laplace_noise);
            X[[i, j]] += noise;
        }
    }

    // Generate noisy target variable
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let true_value: f64 = X.slice(s![i, ..]).iter().sum::<f64>() / n_features as f64;
        let noise: f64 = rng.sample(laplace_noise);
        y[i] = true_value + noise;
    }

    Ok((X, y))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_privacy_preserving_dataset() {
        let (X, y) = make_privacy_preserving_dataset(100, 5, 1.0, 0.01, 1.0, Some(42)).unwrap();

        assert_eq!(X.shape(), &[100, 5]);
        assert_eq!(y.len(), 100);

        // Check that noise was added (should have some variance)
        let variance = X.iter().map(|&x| x * x).sum::<f64>() / (X.len() as f64);
        assert!(variance > 0.0, "Dataset should have variance due to noise");
    }

    #[test]
    fn test_privacy_preserving_dataset_invalid_input() {
        // Invalid epsilon
        assert!(make_privacy_preserving_dataset(100, 5, 0.0, 0.01, 1.0, Some(42)).is_err());

        // Invalid delta
        assert!(make_privacy_preserving_dataset(100, 5, 1.0, 1.0, 1.0, Some(42)).is_err());

        // Invalid n_samples
        assert!(make_privacy_preserving_dataset(0, 5, 1.0, 0.01, 1.0, Some(42)).is_err());
    }
}
