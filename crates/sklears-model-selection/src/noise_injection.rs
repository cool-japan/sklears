//! Noise injection for robustness testing
//!
//! This module provides various noise injection strategies for testing model
//! robustness and evaluating performance under different perturbation scenarios.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Distribution;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::prelude::*;

fn noise_error(msg: &str) -> SklearsError {
    SklearsError::InvalidInput(msg.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    /// Gaussian
    Gaussian,
    /// Uniform
    Uniform,
    /// SaltAndPepper
    SaltAndPepper,
    /// Dropout
    Dropout,
    /// Multiplicative
    Multiplicative,
    /// Adversarial
    Adversarial,
    /// OutlierInjection
    OutlierInjection,
    /// LabelNoise
    LabelNoise,
    /// FeatureSwap
    FeatureSwap,
    /// MixedNoise
    MixedNoise,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdversarialMethod {
    /// FGSM
    FGSM,
    /// PGD
    PGD,
    /// RandomNoise
    RandomNoise,
    /// BoundaryAttack
    BoundaryAttack,
}

#[derive(Debug, Clone)]
pub struct NoiseConfig {
    pub noise_type: NoiseType,
    pub intensity: f64,
    pub probability: f64,
    pub random_state: Option<u64>,
    pub adaptive: bool,
    pub preserve_statistics: bool,
    pub adversarial_method: Option<AdversarialMethod>,
    pub outlier_factor: f64,
    pub label_flip_rate: f64,
    pub feature_swap_rate: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            noise_type: NoiseType::Gaussian,
            intensity: 0.1,
            probability: 1.0,
            random_state: None,
            adaptive: false,
            preserve_statistics: false,
            adversarial_method: None,
            outlier_factor: 3.0,
            label_flip_rate: 0.1,
            feature_swap_rate: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RobustnessTestResult {
    pub original_performance: f64,
    pub noisy_performance: f64,
    pub performance_degradation: f64,
    pub noise_sensitivity: f64,
    pub robustness_score: f64,
    pub noise_statistics: NoiseStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseStatistics {
    pub noise_type: String,
    pub intensity: f64,
    pub affected_samples: usize,
    pub affected_features: usize,
    pub signal_to_noise_ratio: f64,
    pub perturbation_magnitude: f64,
}

pub struct NoiseInjector {
    config: NoiseConfig,
    rng: StdRng,
}

impl NoiseInjector {
    pub fn new(config: NoiseConfig) -> Self {
        let rng = if let Some(seed) = config.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        };

        Self { config, rng }
    }

    pub fn inject_feature_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self.config.noise_type {
            NoiseType::Gaussian => self.inject_gaussian_noise(x),
            NoiseType::Uniform => self.inject_uniform_noise(x),
            NoiseType::SaltAndPepper => self.inject_salt_pepper_noise(x),
            NoiseType::Dropout => self.inject_dropout_noise(x),
            NoiseType::Multiplicative => self.inject_multiplicative_noise(x),
            NoiseType::Adversarial => self.inject_adversarial_noise(x),
            NoiseType::OutlierInjection => self.inject_outlier_noise(x),
            NoiseType::FeatureSwap => self.inject_feature_swap_noise(x),
            NoiseType::MixedNoise => self.inject_mixed_noise(x),
            _ => Err(noise_error("Unsupported noise type for data type")),
        }
    }

    pub fn inject_label_noise(&mut self, y: &ArrayView1<i32>) -> Result<Array1<i32>> {
        if self.config.noise_type != NoiseType::LabelNoise {
            return Err(noise_error("Unsupported noise type for data type"));
        }

        let mut noisy_y = y.to_owned();
        let unique_labels: Vec<i32> = {
            let mut labels: Vec<i32> = y.iter().cloned().collect();
            labels.sort_unstable();
            labels.dedup();
            labels
        };

        if unique_labels.len() < 2 {
            return Ok(noisy_y);
        }

        let flip_dist = Bernoulli::new(self.config.label_flip_rate)
            .map_err(|_| noise_error("Invalid label flip rate"))?;

        for i in 0..noisy_y.len() {
            if self.rng.sample(flip_dist) {
                let current_label = noisy_y[i];
                let available_labels: Vec<i32> = unique_labels
                    .iter()
                    .filter(|&&label| label != current_label)
                    .cloned()
                    .collect();

                if !available_labels.is_empty() {
                    let new_label_idx = self.rng.gen_range(0..available_labels.len());
                    noisy_y[i] = available_labels[new_label_idx];
                }
            }
        }

        Ok(noisy_y)
    }

    fn inject_gaussian_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.rng.gen::<f64>() < self.config.probability {
                    let noise_std = if self.config.adaptive {
                        self.config.intensity * x[[i, j]].abs()
                    } else {
                        self.config.intensity
                    };

                    let normal = RandNormal::new(0.0, noise_std)
                        .map_err(|_| noise_error("Random number generation failed"))?;

                    let noise = self.rng.sample(normal);
                    noisy_x[[i, j]] += noise;
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_uniform_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.rng.gen::<f64>() < self.config.probability {
                    let noise_range = if self.config.adaptive {
                        self.config.intensity * x[[i, j]].abs()
                    } else {
                        self.config.intensity
                    };

                    let uniform = Uniform::new(-noise_range, noise_range).unwrap();
                    let noise = self.rng.sample(uniform);
                    noisy_x[[i, j]] += noise;
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_salt_pepper_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.rng.gen::<f64>() < self.config.probability {
                    if self.rng.gen::<f64>() < 0.5 {
                        noisy_x[[i, j]] = min_val;
                    } else {
                        noisy_x[[i, j]] = max_val;
                    }
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_dropout_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.rng.gen::<f64>() < self.config.intensity {
                    noisy_x[[i, j]] = 0.0;
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_multiplicative_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.rng.gen::<f64>() < self.config.probability {
                    let noise_factor = if self.config.intensity > 0.0 {
                        let gamma = Gamma::new(1.0 / self.config.intensity, self.config.intensity)
                            .map_err(|_| noise_error("Random number generation failed"))?;
                        self.rng.sample(gamma)
                    } else {
                        1.0
                    };

                    noisy_x[[i, j]] *= noise_factor;
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_adversarial_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        match self
            .config
            .adversarial_method
            .unwrap_or(AdversarialMethod::RandomNoise)
        {
            AdversarialMethod::FGSM => self.inject_fgsm_noise(x),
            AdversarialMethod::PGD => self.inject_pgd_noise(x),
            AdversarialMethod::RandomNoise => self.inject_random_adversarial_noise(x),
            AdversarialMethod::BoundaryAttack => self.inject_boundary_attack_noise(x),
        }
    }

    fn inject_fgsm_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                let gradient_sign = if self.rng.gen::<f64>() < 0.5 {
                    -1.0
                } else {
                    1.0
                };
                let perturbation = self.config.intensity * gradient_sign;
                noisy_x[[i, j]] += perturbation;
            }
        }

        Ok(noisy_x)
    }

    fn inject_pgd_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();
        let step_size = self.config.intensity * 0.1;
        let num_steps = 10;

        for _ in 0..num_steps {
            for i in 0..n_samples {
                for j in 0..n_features {
                    let gradient_sign = if self.rng.gen::<f64>() < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    let perturbation = step_size * gradient_sign;
                    noisy_x[[i, j]] += perturbation;

                    let max_perturbation = self.config.intensity;
                    noisy_x[[i, j]] = (noisy_x[[i, j]] - x[[i, j]])
                        .max(-max_perturbation)
                        .min(max_perturbation)
                        + x[[i, j]];
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_random_adversarial_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            let mut perturbation_norm: f64 = 0.0;
            let mut perturbations: Vec<f64> = vec![0.0; n_features];

            for j in 0..n_features {
                perturbations[j] = self.rng.gen_range(-1.0..1.0);
                perturbation_norm += perturbations[j].powi(2);
            }

            perturbation_norm = perturbation_norm.sqrt();
            if perturbation_norm > 0.0 {
                for j in 0..n_features {
                    perturbations[j] =
                        (perturbations[j] / perturbation_norm) * self.config.intensity;
                    noisy_x[[i, j]] += perturbations[j];
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_boundary_attack_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                let direction = if self.rng.gen::<f64>() < 0.5 {
                    -1.0
                } else {
                    1.0
                };
                let magnitude = self.rng.gen::<f64>() * self.config.intensity;
                noisy_x[[i, j]] += direction * magnitude;
            }
        }

        Ok(noisy_x)
    }

    fn inject_outlier_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for j in 0..n_features {
            let feature_values: Vec<f64> = (0..n_samples).map(|i| x[[i, j]]).collect();
            let mean = feature_values.iter().sum::<f64>() / n_samples as f64;
            let variance = feature_values
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            let std_dev = variance.sqrt();

            for i in 0..n_samples {
                if self.rng.gen::<f64>() < self.config.probability {
                    let outlier_direction = if self.rng.gen::<f64>() < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    let outlier_magnitude = self.config.outlier_factor * std_dev;
                    noisy_x[[i, j]] = mean + outlier_direction * outlier_magnitude;
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_feature_swap_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let (n_samples, n_features) = x.dim();

        if n_features < 2 {
            return Ok(noisy_x);
        }

        for i in 0..n_samples {
            if self.rng.gen::<f64>() < self.config.feature_swap_rate {
                let feature1 = self.rng.gen_range(0..n_features);
                let feature2 = self.rng.gen_range(0..n_features);

                if feature1 != feature2 {
                    let temp = noisy_x[[i, feature1]];
                    noisy_x[[i, feature1]] = noisy_x[[i, feature2]];
                    noisy_x[[i, feature2]] = temp;
                }
            }
        }

        Ok(noisy_x)
    }

    fn inject_mixed_noise(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut noisy_x = x.to_owned();
        let noise_types = [
            NoiseType::Gaussian,
            NoiseType::Uniform,
            NoiseType::SaltAndPepper,
            NoiseType::Multiplicative,
        ];

        let original_noise_type = self.config.noise_type;
        let original_intensity = self.config.intensity;

        for &noise_type in &noise_types {
            self.config.noise_type = noise_type;
            self.config.intensity = original_intensity / noise_types.len() as f64;

            noisy_x = match noise_type {
                NoiseType::Gaussian => self.inject_gaussian_noise(&noisy_x.view())?,
                NoiseType::Uniform => self.inject_uniform_noise(&noisy_x.view())?,
                NoiseType::SaltAndPepper => self.inject_salt_pepper_noise(&noisy_x.view())?,
                NoiseType::Multiplicative => self.inject_multiplicative_noise(&noisy_x.view())?,
                _ => noisy_x,
            };
        }

        self.config.noise_type = original_noise_type;
        self.config.intensity = original_intensity;

        Ok(noisy_x)
    }

    pub fn compute_noise_statistics(
        &self,
        original: &ArrayView2<f64>,
        noisy: &ArrayView2<f64>,
    ) -> NoiseStatistics {
        let (n_samples, n_features) = original.dim();
        let mut affected_samples = 0;
        let mut affected_features = 0;
        let mut total_perturbation = 0.0;

        for i in 0..n_samples {
            let mut sample_affected = false;
            for j in 0..n_features {
                let perturbation = (noisy[[i, j]] - original[[i, j]]).abs();
                if perturbation > 1e-10 {
                    if !sample_affected {
                        affected_samples += 1;
                        sample_affected = true;
                    }
                    total_perturbation += perturbation;
                }
            }
        }

        for j in 0..n_features {
            let mut feature_affected = false;
            for i in 0..n_samples {
                if (noisy[[i, j]] - original[[i, j]]).abs() > 1e-10 {
                    feature_affected = true;
                    break;
                }
            }
            if feature_affected {
                affected_features += 1;
            }
        }

        let signal_power: f64 = original.iter().map(|&x| x.powi(2)).sum();
        let noise_power: f64 = original
            .iter()
            .zip(noisy.iter())
            .map(|(&orig, &noise)| (noise - orig).powi(2))
            .sum();

        let snr = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        let avg_perturbation = total_perturbation / (n_samples * n_features) as f64;

        NoiseStatistics {
            noise_type: format!("{:?}", self.config.noise_type),
            intensity: self.config.intensity,
            affected_samples,
            affected_features,
            signal_to_noise_ratio: snr,
            perturbation_magnitude: avg_perturbation,
        }
    }
}

pub fn robustness_test<M, F>(
    model: &M,
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
    noise_configs: Vec<NoiseConfig>,
    eval_fn: F,
) -> Result<Vec<RobustnessTestResult>>
where
    M: Clone,
    F: Fn(&M, &ArrayView2<f64>, &ArrayView1<f64>) -> f64 + Copy,
{
    let original_performance = eval_fn(model, x, y);
    let mut results = Vec::new();

    for config in noise_configs {
        let mut injector = NoiseInjector::new(config.clone());
        let noisy_x = injector.inject_feature_noise(x)?;
        let noisy_performance = eval_fn(model, &noisy_x.view(), y);

        let performance_degradation = original_performance - noisy_performance;
        let noise_sensitivity = performance_degradation / config.intensity.max(1e-10);
        let robustness_score =
            1.0 - (performance_degradation / original_performance.max(1e-10)).abs();

        let noise_statistics = injector.compute_noise_statistics(x, &noisy_x.view());

        results.push(RobustnessTestResult {
            original_performance,
            noisy_performance,
            performance_degradation,
            noise_sensitivity,
            robustness_score: robustness_score.max(0.0),
            noise_statistics,
        });
    }

    Ok(results)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2, Array2};

    fn create_test_data() -> Array2<f64> {
        arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ])
    }

    #[test]
    fn test_gaussian_noise() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::Gaussian,
            intensity: 0.1,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
        assert!(noisy_x != x);
    }

    #[test]
    fn test_uniform_noise() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::Uniform,
            intensity: 0.2,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
    }

    #[test]
    fn test_dropout_noise() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::Dropout,
            intensity: 0.3,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        let zero_count = noisy_x.iter().filter(|&&val| val == 0.0).count();
        assert!(zero_count > 0);
    }

    #[test]
    fn test_label_noise() {
        let y = arr1(&[0, 1, 2, 0, 1, 2]);
        let config = NoiseConfig {
            noise_type: NoiseType::LabelNoise,
            label_flip_rate: 0.5,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_y = injector.inject_label_noise(&y.view()).unwrap();

        assert_eq!(noisy_y.len(), y.len());
        assert!(noisy_y != y);
    }

    #[test]
    fn test_adversarial_noise() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::Adversarial,
            intensity: 0.1,
            adversarial_method: Some(AdversarialMethod::FGSM),
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
    }

    #[test]
    fn test_outlier_injection() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::OutlierInjection,
            probability: 0.1,
            outlier_factor: 3.0,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
    }

    #[test]
    fn test_mixed_noise() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::MixedNoise,
            intensity: 0.1,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
    }

    #[test]
    fn test_noise_statistics() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::Gaussian,
            intensity: 0.1,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();
        let stats = injector.compute_noise_statistics(&x.view(), &noisy_x.view());

        assert!(stats.affected_samples > 0);
        assert!(stats.affected_features > 0);
        assert!(stats.signal_to_noise_ratio.is_finite());
        assert!(stats.perturbation_magnitude >= 0.0);
    }

    #[test]
    fn test_adaptive_noise() {
        let x = create_test_data();
        let config = NoiseConfig {
            noise_type: NoiseType::Gaussian,
            intensity: 0.1,
            adaptive: true,
            random_state: Some(42),
            ..Default::default()
        };

        let mut injector = NoiseInjector::new(config);
        let noisy_x = injector.inject_feature_noise(&x.view()).unwrap();

        assert_eq!(noisy_x.dim(), x.dim());
    }
}
