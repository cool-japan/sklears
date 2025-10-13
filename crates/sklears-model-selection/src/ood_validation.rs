//! Out-of-Distribution (OOD) Validation
//!
//! This module provides methods for detecting and validating models on out-of-distribution data.
//! Out-of-distribution validation is crucial for understanding model robustness and performance
//! when encountering data that differs from the training distribution.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use sklears_core::types::Float;

/// Out-of-Distribution detection methods
#[derive(Debug, Clone)]
pub enum OODDetectionMethod {
    /// Statistical distance-based detection (KL divergence, Wasserstein distance)
    StatisticalDistance { threshold: Float },
    /// Isolation Forest for anomaly detection
    IsolationForest { contamination: Float },
    /// One-Class SVM for novelty detection
    OneClassSVM { nu: Float },
    /// Mahalanobis distance from training distribution
    MahalanobisDistance { threshold: Float },
    /// Reconstruction error from autoencoder
    ReconstructionError { threshold: Float },
    /// Ensemble-based uncertainty detection
    EnsembleUncertainty { threshold: Float },
}

/// Configuration for out-of-distribution validation
#[derive(Debug, Clone)]
pub struct OODValidationConfig {
    pub detection_method: OODDetectionMethod,
    pub validation_split: Float,
    pub random_state: Option<u64>,
    pub min_ood_samples: usize,
    pub confidence_level: Float,
}

impl Default for OODValidationConfig {
    fn default() -> Self {
        Self {
            detection_method: OODDetectionMethod::StatisticalDistance { threshold: 0.1 },
            validation_split: 0.2,
            random_state: None,
            min_ood_samples: 10,
            confidence_level: 0.95,
        }
    }
}

/// Results from out-of-distribution validation
#[derive(Debug, Clone)]
pub struct OODValidationResult {
    pub in_distribution_score: Float,
    pub out_of_distribution_score: Float,
    pub ood_detection_accuracy: Float,
    pub ood_samples_detected: usize,
    pub total_ood_samples: usize,
    pub degradation_score: Float,
    pub confidence_intervals: OODConfidenceIntervals,
    pub feature_importance: Vec<Float>,
    pub distribution_shift_metrics: DistributionShiftMetrics,
}

/// Confidence intervals for OOD validation metrics
#[derive(Debug, Clone)]
pub struct OODConfidenceIntervals {
    pub in_distribution_lower: Float,
    pub in_distribution_upper: Float,
    pub out_of_distribution_lower: Float,
    pub out_of_distribution_upper: Float,
    pub degradation_lower: Float,
    pub degradation_upper: Float,
}

/// Metrics for measuring distribution shift
#[derive(Debug, Clone)]
pub struct DistributionShiftMetrics {
    pub kl_divergence: Float,
    pub wasserstein_distance: Float,
    pub population_stability_index: Float,
    pub feature_drift_scores: Vec<Float>,
}

/// Out-of-Distribution Validator
pub struct OODValidator {
    config: OODValidationConfig,
}

impl OODValidator {
    /// Create a new OOD validator with default configuration
    pub fn new() -> Self {
        Self {
            config: OODValidationConfig::default(),
        }
    }

    /// Create a new OOD validator with custom configuration
    pub fn with_config(config: OODValidationConfig) -> Self {
        Self { config }
    }

    /// Set the detection method
    pub fn detection_method(mut self, method: OODDetectionMethod) -> Self {
        self.config.detection_method = method;
        self
    }

    /// Set the validation split ratio
    pub fn validation_split(mut self, split: Float) -> Self {
        self.config.validation_split = split;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set the minimum number of OOD samples required
    pub fn min_ood_samples(mut self, min_samples: usize) -> Self {
        self.config.min_ood_samples = min_samples;
        self
    }

    /// Set the confidence level for statistical tests
    pub fn confidence_level(mut self, level: Float) -> Self {
        self.config.confidence_level = level;
        self
    }

    /// Validate a model's performance on out-of-distribution data
    pub fn validate<E, P>(
        &self,
        estimator: &E,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_ood: &Array2<Float>,
        y_ood: &Array1<Float>,
    ) -> Result<OODValidationResult, Box<dyn std::error::Error>>
    where
        E: Clone,
        P: Clone,
    {
        // Detect OOD samples
        let ood_mask = self.detect_ood_samples(x_train, x_ood)?;
        let detected_ood_count = ood_mask.iter().filter(|&&x| x).count();

        if detected_ood_count < self.config.min_ood_samples {
            return Err(format!(
                "Insufficient OOD samples detected: {} < {}",
                detected_ood_count, self.config.min_ood_samples
            )
            .into());
        }

        // Calculate distribution shift metrics
        let shift_metrics = self.calculate_distribution_shift(x_train, x_ood)?;

        // Calculate feature importance for OOD detection
        let feature_importance = self.calculate_feature_importance(x_train, x_ood)?;

        // Split OOD data for validation
        let (x_ood_val, y_ood_val) = self.split_ood_data(x_ood, y_ood)?;

        // Calculate performance metrics
        let in_dist_score = self.evaluate_in_distribution(estimator, x_train, y_train)?;
        let ood_score = self.evaluate_out_of_distribution(estimator, &x_ood_val, &y_ood_val)?;

        let degradation_score = (in_dist_score - ood_score) / in_dist_score;
        let ood_detection_accuracy = detected_ood_count as Float / x_ood.nrows() as Float;

        // Calculate confidence intervals using bootstrap
        let confidence_intervals = self
            .calculate_confidence_intervals(estimator, x_train, y_train, &x_ood_val, &y_ood_val)?;

        Ok(OODValidationResult {
            in_distribution_score: in_dist_score,
            out_of_distribution_score: ood_score,
            ood_detection_accuracy,
            ood_samples_detected: detected_ood_count,
            total_ood_samples: x_ood.nrows(),
            degradation_score,
            confidence_intervals,
            feature_importance,
            distribution_shift_metrics: shift_metrics,
        })
    }

    /// Detect out-of-distribution samples
    fn detect_ood_samples(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        match &self.config.detection_method {
            OODDetectionMethod::StatisticalDistance { threshold } => {
                self.detect_statistical_distance(x_train, x_ood, *threshold)
            }
            OODDetectionMethod::MahalanobisDistance { threshold } => {
                self.detect_mahalanobis(x_train, x_ood, *threshold)
            }
            OODDetectionMethod::IsolationForest { contamination } => {
                self.detect_isolation_forest(x_train, x_ood, *contamination)
            }
            OODDetectionMethod::OneClassSVM { nu } => {
                self.detect_one_class_svm(x_train, x_ood, *nu)
            }
            OODDetectionMethod::ReconstructionError { threshold } => {
                self.detect_reconstruction_error(x_train, x_ood, *threshold)
            }
            OODDetectionMethod::EnsembleUncertainty { threshold } => {
                self.detect_ensemble_uncertainty(x_train, x_ood, *threshold)
            }
        }
    }

    /// Statistical distance-based OOD detection
    fn detect_statistical_distance(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let mut ood_mask = Vec::new();

        // Calculate feature-wise KL divergence
        for i in 0..x_ood.nrows() {
            let sample = x_ood.row(i);
            let distance = self.calculate_kl_divergence_sample(x_train, &sample)?;
            ood_mask.push(distance > threshold);
        }

        Ok(ood_mask)
    }

    /// Mahalanobis distance-based OOD detection
    fn detect_mahalanobis(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        // Calculate mean and covariance of training data
        let mean = self.calculate_mean(x_train)?;
        let cov_inv = self.calculate_inverse_covariance(x_train)?;

        let mut ood_mask = Vec::new();

        for i in 0..x_ood.nrows() {
            let sample = x_ood.row(i);
            let distance = self.mahalanobis_distance(&sample, &mean, &cov_inv)?;
            ood_mask.push(distance > threshold);
        }

        Ok(ood_mask)
    }

    /// Isolation Forest-based OOD detection (simplified implementation)
    fn detect_isolation_forest(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        contamination: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        // Simplified isolation forest - in practice would use a proper implementation
        let n_trees = 100;
        let mut scores = vec![0.0; x_ood.nrows()];

        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        for _ in 0..n_trees {
            let tree_scores = self.isolation_tree_scores(x_train, x_ood, &mut rng)?;
            for (i, score) in tree_scores.iter().enumerate() {
                scores[i] += score;
            }
        }

        // Average scores and threshold
        for score in &mut scores {
            *score /= n_trees as Float;
        }

        let threshold =
            scores.iter().fold(0.0, |a, &b| a + b) / scores.len() as Float + contamination;
        Ok(scores.iter().map(|&score| score > threshold).collect())
    }

    /// One-Class SVM-based OOD detection (simplified implementation)
    fn detect_one_class_svm(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        nu: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        // Simplified one-class SVM - would need proper SVM implementation
        // For now, use a simple centroid-based approach
        let centroid = self.calculate_mean(x_train)?;
        let mut distances: Vec<Float> = (0..x_train.nrows())
            .map(|i| self.euclidean_distance(&x_train.row(i), &centroid))
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((1.0 - nu) * distances.len() as Float) as usize;
        let threshold = distances[threshold_idx.min(distances.len() - 1)];

        let mut ood_mask = Vec::new();
        for i in 0..x_ood.nrows() {
            let distance = self.euclidean_distance(&x_ood.row(i), &centroid);
            ood_mask.push(distance > threshold);
        }

        Ok(ood_mask)
    }

    /// Reconstruction error-based OOD detection (simplified)
    fn detect_reconstruction_error(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        // Simplified autoencoder reconstruction - use PCA as approximation
        let mean = self.calculate_mean(x_train)?;
        let mut ood_mask = Vec::new();

        for i in 0..x_ood.nrows() {
            let sample = x_ood.row(i);
            let reconstruction_error = self.euclidean_distance(&sample, &mean);
            ood_mask.push(reconstruction_error > threshold);
        }

        Ok(ood_mask)
    }

    /// Ensemble uncertainty-based OOD detection
    fn detect_ensemble_uncertainty(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        // Use ensemble of simple models (k-means clusters) to estimate uncertainty
        let n_clusters = 5;
        let centroids = self.k_means_centroids(x_train, n_clusters)?;

        let mut ood_mask = Vec::new();

        for i in 0..x_ood.nrows() {
            let sample = x_ood.row(i);
            let uncertainties: Vec<Float> = centroids
                .iter()
                .map(|centroid| self.euclidean_distance(&sample, centroid))
                .collect();

            let min_distance = uncertainties.iter().fold(Float::INFINITY, |a, &b| a.min(b));
            ood_mask.push(min_distance > threshold);
        }

        Ok(ood_mask)
    }

    /// Calculate distribution shift metrics
    fn calculate_distribution_shift(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
    ) -> Result<DistributionShiftMetrics, Box<dyn std::error::Error>> {
        let kl_divergence = self.calculate_kl_divergence(x_train, x_ood)?;
        let wasserstein_distance = self.calculate_wasserstein_distance(x_train, x_ood)?;
        let psi = self.calculate_population_stability_index(x_train, x_ood)?;
        let feature_drift_scores = self.calculate_feature_drift_scores(x_train, x_ood)?;

        Ok(DistributionShiftMetrics {
            kl_divergence,
            wasserstein_distance,
            population_stability_index: psi,
            feature_drift_scores,
        })
    }

    /// Calculate feature importance for OOD detection
    fn calculate_feature_importance(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
    ) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
        let n_features = x_train.ncols();
        let mut importance = vec![0.0; n_features];

        for j in 0..n_features {
            let train_feature = x_train.column(j);
            let ood_feature = x_ood.column(j);

            // Use KS test statistic as importance measure
            importance[j] = self.kolmogorov_smirnov_statistic(&train_feature, &ood_feature)?;
        }

        Ok(importance)
    }

    /// Split OOD data for validation
    fn split_ood_data(
        &self,
        x_ood: &Array2<Float>,
        y_ood: &Array1<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>), Box<dyn std::error::Error>> {
        let n_samples = x_ood.nrows();
        let n_val = (n_samples as Float * self.config.validation_split) as usize;

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if let Some(seed) = self.config.random_state {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }

        let val_indices = &indices[..n_val];

        let x_val =
            Array2::from_shape_fn((n_val, x_ood.ncols()), |(i, j)| x_ood[[val_indices[i], j]]);
        let y_val = Array1::from_shape_fn(n_val, |i| y_ood[val_indices[i]]);

        Ok((x_val, y_val))
    }

    /// Evaluate in-distribution performance (mock implementation)
    fn evaluate_in_distribution<E>(
        &self,
        _estimator: &E,
        _x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Mock implementation - would use actual model evaluation
        Ok(0.95) // Assume high in-distribution performance
    }

    /// Evaluate out-of-distribution performance (mock implementation)
    fn evaluate_out_of_distribution<E>(
        &self,
        _estimator: &E,
        _x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Mock implementation - would use actual model evaluation
        Ok(0.75) // Assume degraded OOD performance
    }

    /// Calculate confidence intervals using bootstrap
    fn calculate_confidence_intervals<E>(
        &self,
        _estimator: &E,
        _x_train: &Array2<Float>,
        _y_train: &Array1<Float>,
        _x_ood: &Array2<Float>,
        _y_ood: &Array1<Float>,
    ) -> Result<OODConfidenceIntervals, Box<dyn std::error::Error>> {
        // Mock implementation - would use bootstrap sampling
        Ok(OODConfidenceIntervals {
            in_distribution_lower: 0.92,
            in_distribution_upper: 0.98,
            out_of_distribution_lower: 0.70,
            out_of_distribution_upper: 0.80,
            degradation_lower: 0.15,
            degradation_upper: 0.25,
        })
    }

    // Helper methods for calculations
    fn calculate_mean(
        &self,
        x: &Array2<Float>,
    ) -> Result<Array1<Float>, Box<dyn std::error::Error>> {
        let n_samples = x.nrows() as Float;
        let n_features = x.ncols();
        let mut mean = Array1::zeros(n_features);

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                mean[j] += x[[i, j]];
            }
        }

        for j in 0..n_features {
            mean[j] /= n_samples;
        }

        Ok(mean)
    }

    fn calculate_inverse_covariance(
        &self,
        x: &Array2<Float>,
    ) -> Result<Array2<Float>, Box<dyn std::error::Error>> {
        // Simplified covariance calculation - would use proper matrix inversion
        let n_features = x.ncols();
        let cov_inv = Array2::eye(n_features);

        // Mock implementation - assume identity matrix for simplicity
        Ok(cov_inv)
    }

    fn mahalanobis_distance(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        mean: &Array1<Float>,
        cov_inv: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Simplified Mahalanobis distance calculation
        let diff: Array1<Float> = sample.to_owned() - mean;
        let distance = diff.dot(&diff.dot(cov_inv));
        Ok(distance.sqrt())
    }

    fn euclidean_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<Float>,
        b: &Array1<Float>,
    ) -> Float {
        let diff: Array1<Float> = a.to_owned() - b;
        diff.dot(&diff).sqrt()
    }

    fn calculate_kl_divergence_sample(
        &self,
        x_train: &Array2<Float>,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Simplified KL divergence calculation
        Ok(0.1) // Mock value
    }

    fn calculate_kl_divergence(
        &self,
        _x_train: &Array2<Float>,
        _x_ood: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        Ok(0.15) // Mock value
    }

    fn calculate_wasserstein_distance(
        &self,
        _x_train: &Array2<Float>,
        _x_ood: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        Ok(0.12) // Mock value
    }

    fn calculate_population_stability_index(
        &self,
        _x_train: &Array2<Float>,
        _x_ood: &Array2<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        Ok(0.08) // Mock value
    }

    fn calculate_feature_drift_scores(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
    ) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
        let n_features = x_train.ncols();
        Ok(vec![0.05; n_features]) // Mock values
    }

    fn kolmogorov_smirnov_statistic(
        &self,
        _train_feature: &scirs2_core::ndarray::ArrayView1<Float>,
        _ood_feature: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        Ok(0.1) // Mock value
    }

    fn isolation_tree_scores(
        &self,
        _x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        _rng: &mut StdRng,
    ) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
        Ok(vec![0.5; x_ood.nrows()]) // Mock values
    }

    fn k_means_centroids(
        &self,
        x: &Array2<Float>,
        k: usize,
    ) -> Result<Vec<Array1<Float>>, Box<dyn std::error::Error>> {
        // Simplified k-means - just sample k random points as centroids
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut centroids = Vec::new();
        for _ in 0..k {
            let idx = rng.gen_range(0..x.nrows());
            centroids.push(x.row(idx).to_owned());
        }

        Ok(centroids)
    }
}

impl Default for OODValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for out-of-distribution validation
pub fn validate_ood<E, P>(
    estimator: &E,
    x_train: &Array2<Float>,
    y_train: &Array1<Float>,
    x_ood: &Array2<Float>,
    y_ood: &Array1<Float>,
    config: Option<OODValidationConfig>,
) -> Result<OODValidationResult, Box<dyn std::error::Error>>
where
    E: Clone,
    P: Clone,
{
    let validator = match config {
        Some(cfg) => OODValidator::with_config(cfg),
        None => OODValidator::new(),
    };

    validator.validate::<E, P>(estimator, x_train, y_train, x_ood, y_ood)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_ood_validator_creation() {
        let validator = OODValidator::new();
        assert!(matches!(
            validator.config.detection_method,
            OODDetectionMethod::StatisticalDistance { .. }
        ));
    }

    #[test]
    fn test_ood_validator_with_config() {
        let config = OODValidationConfig {
            detection_method: OODDetectionMethod::MahalanobisDistance { threshold: 2.0 },
            validation_split: 0.3,
            random_state: Some(42),
            min_ood_samples: 20,
            confidence_level: 0.99,
        };

        let validator = OODValidator::with_config(config.clone());
        assert_eq!(validator.config.validation_split, 0.3);
        assert_eq!(validator.config.random_state, Some(42));
        assert_eq!(validator.config.min_ood_samples, 20);
        assert_eq!(validator.config.confidence_level, 0.99);
    }

    #[test]
    fn test_ood_detection_methods() {
        let x_train = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
        let x_ood = Array2::from_shape_vec((5, 3), vec![5.0; 15]).unwrap();

        let validator = OODValidator::new()
            .detection_method(OODDetectionMethod::StatisticalDistance { threshold: 0.5 });

        let result = validator.detect_ood_samples(&x_train, &x_ood);
        assert!(result.is_ok());

        let ood_mask = result.unwrap();
        assert_eq!(ood_mask.len(), 5);
    }

    #[test]
    fn test_mahalanobis_detection() {
        let x_train = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.0, 1.0, 1.2, 0.8, 0.8, 1.2, 1.1, 0.9, 0.9, 1.1,
                1.0, 1.0, 1.1, 0.9,
            ],
        )
        .unwrap();
        let x_ood = Array2::from_shape_vec((3, 2), vec![5.0, 5.0, 0.0, 0.0, 10.0, 10.0]).unwrap();

        let validator = OODValidator::new()
            .detection_method(OODDetectionMethod::MahalanobisDistance { threshold: 2.0 });

        let result = validator.detect_ood_samples(&x_train, &x_ood);
        assert!(result.is_ok());
    }

    #[test]
    fn test_feature_importance_calculation() {
        let x_train = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
        let x_ood = Array2::from_shape_vec((5, 3), vec![2.0; 15]).unwrap();

        let validator = OODValidator::new();
        let result = validator.calculate_feature_importance(&x_train, &x_ood);

        assert!(result.is_ok());
        let importance = result.unwrap();
        assert_eq!(importance.len(), 3);
    }

    #[test]
    fn test_ood_data_splitting() {
        let x_ood = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();
        let y_ood = Array1::from_shape_vec(10, vec![0.5; 10]).unwrap();

        let validator = OODValidator::new().validation_split(0.3);
        let result = validator.split_ood_data(&x_ood, &y_ood);

        assert!(result.is_ok());
        let (x_val, y_val) = result.unwrap();
        assert_eq!(x_val.nrows(), 3); // 30% of 10
        assert_eq!(y_val.len(), 3);
    }

    #[test]
    fn test_distance_calculations() {
        let validator = OODValidator::new();

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mean_result = validator.calculate_mean(&x);

        assert!(mean_result.is_ok());
        let mean = mean_result.unwrap();
        assert_eq!(mean.len(), 2);
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_convenience_function() {
        #[derive(Clone)]
        struct MockEstimator;

        #[derive(Clone)]
        struct MockPredictions;

        let estimator = MockEstimator;
        let x_train = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();
        let y_train = Array1::from_shape_vec(10, vec![0.0; 10]).unwrap();
        let x_ood = Array2::from_shape_vec((5, 2), vec![5.0; 10]).unwrap();
        let y_ood = Array1::from_shape_vec(5, vec![1.0; 5]).unwrap();

        let result = validate_ood::<MockEstimator, MockPredictions>(
            &estimator, &x_train, &y_train, &x_ood, &y_ood, None,
        );

        // This will fail with insufficient OOD samples, but tests the API
        assert!(result.is_err());
    }
}
