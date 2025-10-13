//! Experimental design and A/B testing simulation
//!
//! This module contains generators for experimental design datasets including
//! A/B testing simulations, factorial designs, and causal inference experiments.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Normal;
use sklears_core::error::{Result, SklearsError};

/// A/B test configuration
#[derive(Debug, Clone)]
pub struct ABTestConfig {
    pub control_rate: f64,
    pub treatment_effect: f64,
    pub significance_level: f64,
    pub power: f64,
    pub minimum_detectable_effect: f64,
}

/// Generate A/B testing simulation data
pub fn make_ab_testing_simulation(
    config: ABTestConfig,
    n_samples: usize,
    n_features: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>, Array1<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive".to_string(),
        ));
    }

    if config.control_rate < 0.0 || config.control_rate > 1.0 {
        return Err(SklearsError::InvalidInput(
            "control_rate must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate user features
    let mut X = Array2::zeros((n_samples, n_features));
    let normal = Normal::new(0.0, 1.0).unwrap();

    for i in 0..n_samples {
        for j in 0..n_features {
            X[[i, j]] = rng.sample(normal);
        }
    }

    // Assign users to control (0) or treatment (1) groups
    let mut group_assignment = Array1::zeros(n_samples);
    let n_treatment = (n_samples as f64 * (1.0 - config.control_rate)) as usize;

    // Random assignment with stratification based on features
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    for (i, &idx) in indices.iter().enumerate() {
        group_assignment[idx] = if i < n_treatment { 1 } else { 0 };
    }

    // Generate outcomes based on group assignment and user features
    let mut outcomes = Array1::zeros(n_samples);
    let mut conversion = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let user_propensity = X.slice(ndarray::s![i, ..]).iter().sum::<f64>() / n_features as f64;
        let base_rate = 0.1 + 0.05 * user_propensity.tanh(); // Base conversion rate

        let conversion_probability = if group_assignment[i] == 1 {
            // Treatment group gets the effect
            base_rate + config.treatment_effect
        } else {
            // Control group
            base_rate
        };

        let converted = rng.gen::<f64>() < conversion_probability.max(0.0).min(1.0);
        conversion[i] = if converted { 1 } else { 0 };

        // Outcome value (e.g., revenue) - higher for conversions
        outcomes[i] = if converted {
            let baseline_value = 10.0 + 5.0 * user_propensity;
            let treatment_bonus = if group_assignment[i] == 1 {
                config.treatment_effect * 20.0
            } else {
                0.0
            };
            baseline_value + treatment_bonus + rng.sample(Normal::new(0.0, 2.0).unwrap())
        } else {
            0.0
        };
    }

    Ok((X, group_assignment, outcomes, conversion))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_ab_testing_simulation() {
        let config = ABTestConfig {
            control_rate: 0.5,
            treatment_effect: 0.1,
            significance_level: 0.05,
            power: 0.8,
            minimum_detectable_effect: 0.05,
        };

        let (X, groups, outcomes, conversions) =
            make_ab_testing_simulation(config, 200, 3, Some(42)).unwrap();

        assert_eq!(X.shape(), &[200, 3]);
        assert_eq!(groups.len(), 200);
        assert_eq!(outcomes.len(), 200);
        assert_eq!(conversions.len(), 200);

        // Check group assignments are 0 or 1
        for &group in groups.iter() {
            assert!(group == 0 || group == 1, "Groups should be 0 or 1");
        }

        // Check conversions are 0 or 1
        for &conv in conversions.iter() {
            assert!(conv == 0 || conv == 1, "Conversions should be 0 or 1");
        }

        // Should have roughly balanced groups
        let treatment_count = groups.iter().filter(|&&g| g == 1).count();
        assert!(
            treatment_count > 50 && treatment_count < 150,
            "Groups should be roughly balanced"
        );
    }

    #[test]
    fn test_ab_testing_simulation_invalid_input() {
        let config = ABTestConfig {
            control_rate: 1.5, // Invalid
            treatment_effect: 0.1,
            significance_level: 0.05,
            power: 0.8,
            minimum_detectable_effect: 0.05,
        };

        assert!(make_ab_testing_simulation(config, 200, 3, Some(42)).is_err());
    }
}
