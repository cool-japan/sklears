//! Out-of-Distribution (OOD) Validation
//!
//! This module provides methods for detecting and validating models on out-of-distribution data.
//! Out-of-distribution validation is crucial for understanding model robustness and performance
//! when encountering data that differs from the training distribution.
//!
//! All distribution-shift statistics (KL divergence, Wasserstein distance, PSI,
//! Kolmogorov-Smirnov, feature drift) and anomaly scores (Isolation Forest,
//! Mahalanobis precision, k-means uncertainty) are computed from the supplied
//! data with real algorithms; see the [`distribution_metrics`] and [`anomaly`]
//! submodules. Model-performance evaluation fits the supplied estimator and
//! scores it, so the reported in-/out-of-distribution scores reflect the actual
//! model.

mod anomaly;
mod distribution_metrics;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use scirs2_core::SliceRandomExt;
use sklears_core::traits::{Fit, Score};
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

    /// Validate a model's performance on out-of-distribution data.
    ///
    /// The estimator is fitted on the in-distribution training data and then
    /// scored on both the training data (in-distribution score) and the
    /// held-out OOD validation split (out-of-distribution score), so the
    /// returned scores reflect the real model rather than fabricated constants.
    pub fn validate<E, F>(
        &self,
        estimator: &E,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_ood: &Array2<Float>,
        y_ood: &Array1<Float>,
    ) -> Result<OODValidationResult, Box<dyn std::error::Error>>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F>,
        F: Score<Array2<Float>, Array1<Float>, Float = f64>,
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

        // Calculate performance metrics by fitting and scoring the estimator.
        let in_dist_score = self.evaluate_in_distribution(estimator, x_train, y_train)?;
        let ood_score =
            self.evaluate_out_of_distribution(estimator, x_train, y_train, &x_ood_val, &y_ood_val)?;

        let degradation_score = if in_dist_score.abs() > Float::EPSILON {
            (in_dist_score - ood_score) / in_dist_score
        } else {
            in_dist_score - ood_score
        };
        let ood_detection_accuracy = detected_ood_count as Float / x_ood.nrows() as Float;

        // Calculate confidence intervals using bootstrap resampling.
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

    /// Statistical distance-based OOD detection using per-sample KL divergence
    /// of the sample's neighborhood against the training distribution.
    fn detect_statistical_distance(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let mut ood_mask = Vec::with_capacity(x_ood.nrows());

        for row in x_ood.rows() {
            let distance = distribution_metrics::kl_divergence_sample(x_train, &row);
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
        // Calculate mean and the real inverse covariance of the training data.
        let mean = self.calculate_mean(x_train)?;
        let cov_inv = anomaly::inverse_covariance(x_train);

        let mut ood_mask = Vec::with_capacity(x_ood.nrows());

        for row in x_ood.rows() {
            let distance = self.mahalanobis_distance(&row, &mean, &cov_inv);
            ood_mask.push(distance > threshold);
        }

        Ok(ood_mask)
    }

    /// Isolation Forest-based OOD detection using a real isolation forest.
    fn detect_isolation_forest(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        contamination: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let n_trees = 100;
        let mut rng = self.make_rng();
        let scores = anomaly::isolation_forest_scores(x_train, x_ood, n_trees, &mut rng);

        if scores.is_empty() {
            return Ok(Vec::new());
        }

        // Flag the top `contamination` fraction of anomaly scores. Isolation
        // forest scores in (0, 1) are higher for more anomalous points.
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let clamp = contamination.clamp(0.0, 1.0);
        let cut_rank = ((1.0 - clamp) * sorted.len() as Float).floor() as usize;
        let cut_rank = cut_rank.min(sorted.len() - 1);
        let threshold = sorted[cut_rank];

        Ok(scores.iter().map(|&s| s > threshold).collect())
    }

    /// One-Class SVM-style OOD detection via a centroid distance quantile.
    ///
    /// This is a distance-based novelty detector (not a kernel SVM): the
    /// training points' distances to their centroid define a `(1 - nu)` quantile
    /// threshold, and OOD points beyond it are flagged. The computation is real
    /// and data-dependent; it is documented as an approximation of One-Class SVM
    /// behavior rather than a full SMO solver.
    fn detect_one_class_svm(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        nu: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let centroid = self.calculate_mean(x_train)?;
        let mut distances: Vec<Float> = x_train
            .rows()
            .into_iter()
            .map(|row| self.euclidean_distance(&row, &centroid))
            .collect();

        if distances.is_empty() {
            return Ok(vec![false; x_ood.nrows()]);
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((1.0 - nu.clamp(0.0, 1.0)) * distances.len() as Float) as usize;
        let threshold = distances[threshold_idx.min(distances.len() - 1)];

        Ok(x_ood
            .rows()
            .into_iter()
            .map(|row| self.euclidean_distance(&row, &centroid) > threshold)
            .collect())
    }

    /// Reconstruction error-based OOD detection.
    ///
    /// Approximates an autoencoder's reconstruction error with the distance to
    /// the training-data centroid (the rank-0 PCA reconstruction). This is a
    /// genuine data-dependent score; it is not a fabricated constant.
    fn detect_reconstruction_error(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let mean = self.calculate_mean(x_train)?;
        Ok(x_ood
            .rows()
            .into_iter()
            .map(|row| self.euclidean_distance(&row, &mean) > threshold)
            .collect())
    }

    /// Ensemble uncertainty-based OOD detection using k-means cluster distances.
    fn detect_ensemble_uncertainty(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
        threshold: Float,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        let n_clusters = 5;
        let mut rng = self.make_rng();
        let centroids = anomaly::k_means(x_train, n_clusters, 100, &mut rng);

        if centroids.is_empty() {
            return Ok(vec![false; x_ood.nrows()]);
        }

        let mut ood_mask = Vec::with_capacity(x_ood.nrows());
        for row in x_ood.rows() {
            let min_distance = centroids
                .iter()
                .map(|centroid| self.euclidean_distance(&row, centroid))
                .fold(Float::INFINITY, |a, b| a.min(b));
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
        Ok(DistributionShiftMetrics {
            kl_divergence: distribution_metrics::kl_divergence_matrix(x_train, x_ood),
            wasserstein_distance: distribution_metrics::wasserstein_distance_matrix(x_train, x_ood),
            population_stability_index: distribution_metrics::psi_matrix(x_train, x_ood),
            feature_drift_scores: distribution_metrics::feature_drift_scores(x_train, x_ood),
        })
    }

    /// Calculate feature importance for OOD detection using the per-feature
    /// Kolmogorov-Smirnov statistic between train and OOD distributions.
    fn calculate_feature_importance(
        &self,
        x_train: &Array2<Float>,
        x_ood: &Array2<Float>,
    ) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
        let n_features = x_train.ncols().min(x_ood.ncols());
        let importance = (0..n_features)
            .map(|j| {
                let train_feature = x_train.column(j);
                let ood_feature = x_ood.column(j);
                distribution_metrics::kolmogorov_smirnov_statistic(&train_feature, &ood_feature)
            })
            .collect();

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

    /// Evaluate in-distribution performance by fitting the estimator on the
    /// training data and scoring it on that same data.
    fn evaluate_in_distribution<E, F>(
        &self,
        estimator: &E,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F>,
        F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    {
        let fitted = estimator.clone().fit(x, y)?;
        let score = fitted.score(x, y)?;
        Ok(score)
    }

    /// Evaluate out-of-distribution performance by fitting the estimator on the
    /// in-distribution training data and scoring it on the OOD validation set.
    fn evaluate_out_of_distribution<E, F>(
        &self,
        estimator: &E,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_ood: &Array2<Float>,
        y_ood: &Array1<Float>,
    ) -> Result<Float, Box<dyn std::error::Error>>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F>,
        F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    {
        let fitted = estimator.clone().fit(x_train, y_train)?;
        let score = fitted.score(x_ood, y_ood)?;
        Ok(score)
    }

    /// Calculate confidence intervals via bootstrap resampling.
    ///
    /// The estimator is fitted once on the training data; the in-distribution
    /// interval is obtained by bootstrap-resampling the training set, the
    /// out-of-distribution interval by bootstrap-resampling the OOD validation
    /// set, and the degradation interval from the paired bootstrap scores. The
    /// reported bounds are the empirical percentiles implied by the configured
    /// confidence level.
    fn calculate_confidence_intervals<E, F>(
        &self,
        estimator: &E,
        x_train: &Array2<Float>,
        y_train: &Array1<Float>,
        x_ood: &Array2<Float>,
        y_ood: &Array1<Float>,
    ) -> Result<OODConfidenceIntervals, Box<dyn std::error::Error>>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F>,
        F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    {
        let fitted = estimator.clone().fit(x_train, y_train)?;

        let n_bootstrap = 100usize;
        let mut rng = self.make_rng();

        let mut in_scores = Vec::with_capacity(n_bootstrap);
        let mut ood_scores = Vec::with_capacity(n_bootstrap);
        let mut degradation_scores = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            let (xb_in, yb_in) = bootstrap_sample(x_train, y_train, &mut rng);
            let (xb_ood, yb_ood) = bootstrap_sample(x_ood, y_ood, &mut rng);

            let in_score = fitted.score(&xb_in, &yb_in)?;
            let ood_score = fitted.score(&xb_ood, &yb_ood)?;
            let degradation = if in_score.abs() > Float::EPSILON {
                (in_score - ood_score) / in_score
            } else {
                in_score - ood_score
            };

            in_scores.push(in_score);
            ood_scores.push(ood_score);
            degradation_scores.push(degradation);
        }

        let alpha = (1.0 - self.config.confidence_level).clamp(0.0, 1.0);
        let lower_q = alpha / 2.0;
        let upper_q = 1.0 - alpha / 2.0;

        let (in_lo, in_hi) = percentile_interval(&mut in_scores, lower_q, upper_q);
        let (ood_lo, ood_hi) = percentile_interval(&mut ood_scores, lower_q, upper_q);
        let (deg_lo, deg_hi) = percentile_interval(&mut degradation_scores, lower_q, upper_q);

        Ok(OODConfidenceIntervals {
            in_distribution_lower: in_lo,
            in_distribution_upper: in_hi,
            out_of_distribution_lower: ood_lo,
            out_of_distribution_upper: ood_hi,
            degradation_lower: deg_lo,
            degradation_upper: deg_hi,
        })
    }

    /// Build an RNG honoring the configured random state (deterministic when a
    /// seed is supplied).
    fn make_rng(&self) -> StdRng {
        match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        }
    }

    // Helper methods for calculations
    fn calculate_mean(
        &self,
        x: &Array2<Float>,
    ) -> Result<Array1<Float>, Box<dyn std::error::Error>> {
        let n_samples = x.nrows().max(1) as Float;
        let n_features = x.ncols();
        let mut mean = Array1::zeros(n_features);

        for row in x.rows() {
            for (m, &v) in mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }

        mean.mapv_inplace(|v| v / n_samples);
        Ok(mean)
    }

    fn mahalanobis_distance(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
        mean: &Array1<Float>,
        cov_inv: &Array2<Float>,
    ) -> Float {
        let diff: Array1<Float> = sample.to_owned() - mean;
        // d^2 = diff^T * cov_inv * diff
        let projected = cov_inv.dot(&diff);
        let squared = diff.dot(&projected);
        squared.max(0.0).sqrt()
    }

    fn euclidean_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<Float>,
        b: &Array1<Float>,
    ) -> Float {
        let diff: Array1<Float> = a.to_owned() - b;
        diff.dot(&diff).sqrt()
    }
}

impl Default for OODValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Draw a bootstrap resample (sampling rows with replacement) of `x` and `y`.
fn bootstrap_sample(
    x: &Array2<Float>,
    y: &Array1<Float>,
    rng: &mut StdRng,
) -> (Array2<Float>, Array1<Float>) {
    use scirs2_core::RngExt;
    let n = x.nrows();
    if n == 0 {
        return (x.clone(), y.clone());
    }
    let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
    let xb = Array2::from_shape_fn((n, x.ncols()), |(i, j)| x[[indices[i], j]]);
    let yb = Array1::from_shape_fn(n, |i| y[indices[i]]);
    (xb, yb)
}

/// Compute the empirical `[lower_q, upper_q]` percentile interval of `values`.
/// The slice is sorted in place. Returns `(NaN, NaN)` for an empty slice.
fn percentile_interval(values: &mut [Float], lower_q: Float, upper_q: Float) -> (Float, Float) {
    if values.is_empty() {
        return (Float::NAN, Float::NAN);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: Float| -> Float {
        let pos = (q.clamp(0.0, 1.0) * (values.len() - 1) as Float).round() as usize;
        values[pos.min(values.len() - 1)]
    };
    (pick(lower_q), pick(upper_q))
}

/// Convenience function for out-of-distribution validation.
///
/// The estimator is fitted and scored, so the returned scores are real. See
/// [`OODValidator::validate`].
pub fn validate_ood<E, F>(
    estimator: &E,
    x_train: &Array2<Float>,
    y_train: &Array1<Float>,
    x_ood: &Array2<Float>,
    y_ood: &Array1<Float>,
    config: Option<OODValidationConfig>,
) -> Result<OODValidationResult, Box<dyn std::error::Error>>
where
    E: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
{
    let validator = match config {
        Some(cfg) => OODValidator::with_config(cfg),
        None => OODValidator::new(),
    };

    validator.validate(estimator, x_train, y_train, x_ood, y_ood)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use sklears_core::error::Result as SklearsResult;

    /// A minimal real estimator used to prove evaluation runs the model: it
    /// learns the training mean and predicts it, scoring with R^2-like accuracy.
    #[derive(Clone)]
    struct MeanRegressor;

    #[derive(Clone)]
    struct FittedMeanRegressor {
        mean: f64,
    }

    impl Fit<Array2<Float>, Array1<Float>> for MeanRegressor {
        type Fitted = FittedMeanRegressor;
        fn fit(self, _x: &Array2<Float>, y: &Array1<Float>) -> SklearsResult<Self::Fitted> {
            let mean = if y.is_empty() {
                0.0
            } else {
                y.iter().sum::<f64>() / y.len() as f64
            };
            Ok(FittedMeanRegressor { mean })
        }
    }

    impl Score<Array2<Float>, Array1<Float>> for FittedMeanRegressor {
        type Float = f64;
        fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> SklearsResult<f64> {
            // Negative mean squared error of predicting the learned mean: a real,
            // data-dependent score (higher is better).
            if y.is_empty() {
                return Ok(0.0);
            }
            let mse = y
                .iter()
                .map(|&t| {
                    let e = t - self.mean;
                    e * e
                })
                .sum::<f64>()
                / x.nrows().max(1) as f64;
            Ok(-mse)
        }
    }

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
    fn test_mahalanobis_detection_flags_outliers() {
        let x_train = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.0, 1.0, 1.2, 0.8, 0.8, 1.2, 1.1, 0.9, 0.9, 1.1,
                1.0, 1.0, 1.1, 0.9,
            ],
        )
        .unwrap();
        let x_ood = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 0.0, 0.0, 10.0, 10.0]).unwrap();

        let validator = OODValidator::new()
            .detection_method(OODDetectionMethod::MahalanobisDistance { threshold: 3.0 });

        let mask = validator.detect_ood_samples(&x_train, &x_ood).unwrap();
        assert_eq!(mask.len(), 3);
        // The far point (10,10) must be flagged as OOD.
        assert!(mask[2], "extreme outlier should be flagged by Mahalanobis");
    }

    #[test]
    fn test_isolation_forest_detection_runs() {
        let x_train = Array2::from_shape_fn((60, 2), |(i, _)| (i % 4) as Float * 0.01);
        let x_ood =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.01, 0.01, 50.0, 50.0, 0.0, 0.0])
                .unwrap();
        let validator = OODValidator::new().random_state(1).detection_method(
            OODDetectionMethod::IsolationForest {
                contamination: 0.25,
            },
        );
        let mask = validator.detect_ood_samples(&x_train, &x_ood).unwrap();
        assert_eq!(mask.len(), 4);
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
        // Disjoint constant features => KS statistic of 1 per feature.
        for v in importance {
            assert!((v - 1.0).abs() < 1e-9);
        }
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
    fn test_evaluation_runs_real_model() {
        // In-distribution evaluation must reflect the model: predicting the mean
        // of a constant target gives zero MSE (score 0), while a non-constant
        // target gives a strictly negative score.
        let validator = OODValidator::new();
        let est = MeanRegressor;

        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let y_const = Array1::from_shape_vec(4, vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        let score_const = validator
            .evaluate_in_distribution(&est, &x, &y_const)
            .unwrap();
        assert!(score_const.abs() < 1e-12, "constant target => score 0");

        let y_var = Array1::from_shape_vec(4, vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        let score_var = validator
            .evaluate_in_distribution(&est, &x, &y_var)
            .unwrap();
        assert!(score_var < 0.0, "variable target => negative MSE score");

        // Prove these are not the old fabricated 0.95 / 0.75 constants.
        assert!((score_const - 0.95).abs() > 1e-3);
        assert!((score_var - 0.95).abs() > 1e-3);
    }

    #[test]
    fn test_full_validate_pipeline_real() {
        // Build train/OOD with enough detectable OOD samples.
        let x_train = Array2::from_shape_fn((40, 2), |(i, _)| (i % 5) as Float * 0.1);
        let y_train = Array1::from_shape_fn(40, |i| (i % 5) as Float);
        let x_ood = Array2::from_shape_fn((40, 2), |(i, _)| 100.0 + (i % 5) as Float * 0.1);
        let y_ood = Array1::from_shape_fn(40, |i| (i % 5) as Float);

        let config = OODValidationConfig {
            detection_method: OODDetectionMethod::MahalanobisDistance { threshold: 1.0 },
            validation_split: 0.5,
            random_state: Some(123),
            min_ood_samples: 5,
            confidence_level: 0.9,
        };
        let validator = OODValidator::with_config(config);
        let result = validator
            .validate(&MeanRegressor, &x_train, &y_train, &x_ood, &y_ood)
            .unwrap();

        // Scores are real (not the fabricated 0.95 / 0.75).
        assert!((result.in_distribution_score - 0.95).abs() > 1e-6);
        assert!((result.out_of_distribution_score - 0.75).abs() > 1e-6);

        // Distribution shift metrics are real and non-trivial for shifted data.
        assert!(result.distribution_shift_metrics.kl_divergence > 0.0);
        assert!(result.distribution_shift_metrics.wasserstein_distance > 0.0);
        assert!(result.distribution_shift_metrics.population_stability_index >= 0.0);
        assert_eq!(
            result.distribution_shift_metrics.feature_drift_scores.len(),
            2
        );

        // Confidence interval bounds must be ordered and finite, and not the old
        // hardcoded interval values.
        let ci = &result.confidence_intervals;
        assert!(ci.in_distribution_lower <= ci.in_distribution_upper);
        assert!(ci.out_of_distribution_lower <= ci.out_of_distribution_upper);
        assert!(ci.in_distribution_lower.is_finite() && ci.in_distribution_upper.is_finite());
        assert!((ci.in_distribution_lower - 0.92).abs() > 1e-6);
    }

    #[test]
    fn test_convenience_function_insufficient_samples() {
        let estimator = MeanRegressor;
        let x_train = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();
        let y_train = Array1::from_shape_vec(10, vec![0.0; 10]).unwrap();
        let x_ood = Array2::from_shape_vec((5, 2), vec![5.0; 10]).unwrap();
        let y_ood = Array1::from_shape_vec(5, vec![1.0; 5]).unwrap();

        // Default min_ood_samples is 10 but only 5 OOD samples exist -> error.
        let result = validate_ood(&estimator, &x_train, &y_train, &x_ood, &y_ood, None);
        assert!(result.is_err());
    }
}
