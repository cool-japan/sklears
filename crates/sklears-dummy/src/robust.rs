//! Robust dummy estimators resistant to outliers and anomalies
//!
//! This module provides dummy estimators that are robust to outliers and provide
//! reliable baseline predictions even in the presence of data quality issues.

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{
    essentials::Normal, prelude::*, rngs::StdRng, Distribution, Rng, SeedableRng,
};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict};
use sklears_core::types::{Features, Float};

/// Strategy for robust predictions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum RobustStrategy {
    /// Outlier-resistant methods using robust statistics
    OutlierResistant {
        /// Contamination rate (proportion of outliers expected)
        contamination: Float,
        /// Method for outlier detection
        detection_method: OutlierDetectionMethod,
    },
    /// Trimmed mean baselines removing extreme values
    TrimmedMean {
        /// Proportion of values to trim from each tail
        trim_proportion: Float,
    },
    /// Robust scale estimation using various estimators
    RobustScale {
        /// Scale estimator to use
        scale_estimator: ScaleEstimator,
        /// Location estimator to use
        location_estimator: LocationEstimator,
    },
    /// Breakdown point analysis with different robustness levels
    BreakdownPoint { breakdown_point: Float },
    /// Influence-resistant methods using M-estimators
    InfluenceResistant {
        /// Huber parameter for loss function
        huber_delta: Float,
        /// Maximum number of iterations
        max_iter: usize,
        /// Convergence tolerance
        tolerance: Float,
    },
}

/// Methods for outlier detection
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum OutlierDetectionMethod {
    /// Interquartile Range (IQR) method
    IQR { multiplier: Float },
    /// Z-score method using median absolute deviation
    ModifiedZScore { threshold: Float },
    /// Isolation based on distance from median
    MedianDistance { threshold: Float },
}

/// Scale estimation methods
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ScaleEstimator {
    /// Median Absolute Deviation
    MAD,
    /// Qn estimator (more efficient than MAD)
    Qn,
    /// Inter-Quartile Range
    IQR,
    /// Rousseeuw-Croux Sn estimator
    Sn,
}

/// Location estimation methods
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum LocationEstimator {
    /// Median (most robust)
    Median,
    /// Trimmed mean
    TrimmedMean { trim_proportion: Float },
    /// Huber M-estimator
    Huber { delta: Float },
    /// Biweight midvariance
    Biweight,
}

/// Robust dummy regressor
#[derive(Debug, Clone)]
pub struct RobustDummyRegressor<State = sklears_core::traits::Untrained> {
    /// Strategy for robust predictions
    pub strategy: RobustStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,

    // Fitted parameters
    /// Robust location estimate
    pub(crate) robust_location_: Option<Float>,
    /// Robust scale estimate
    pub(crate) robust_scale_: Option<Float>,
    /// Outlier mask (true for outliers)
    pub(crate) outlier_mask_: Option<Array1<bool>>,
    /// Clean data after outlier removal
    pub(crate) clean_data_: Option<Array1<Float>>,
    /// Breakdown point achieved
    pub(crate) breakdown_point_: Option<Float>,
    /// M-estimator weights
    pub(crate) m_weights_: Option<Array1<Float>>,

    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl RobustDummyRegressor {
    /// Create a new robust dummy regressor
    pub fn new(strategy: RobustStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            robust_location_: None,
            robust_scale_: None,
            outlier_mask_: None,
            clean_data_: None,
            breakdown_point_: None,
            m_weights_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Get the breakdown point achieved by the fitted estimator
    pub fn breakdown_point(&self) -> Option<Float> {
        self.breakdown_point_
    }

    /// Get the outlier mask (available after fitting with outlier-resistant strategy)
    pub fn outlier_mask(&self) -> Option<&Array1<bool>> {
        self.outlier_mask_.as_ref()
    }

    /// Get the M-estimator weights (available after fitting with influence-resistant strategy)
    pub fn m_weights(&self) -> Option<&Array1<Float>> {
        self.m_weights_.as_ref()
    }
}

impl Default for RobustDummyRegressor {
    fn default() -> Self {
        Self::new(RobustStrategy::TrimmedMean {
            trim_proportion: 0.1,
        })
    }
}

impl Estimator for RobustDummyRegressor {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Float>> for RobustDummyRegressor {
    type Fitted = RobustDummyRegressor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        let mut fitted = RobustDummyRegressor {
            strategy: self.strategy.clone(),
            random_state: self.random_state,
            robust_location_: None,
            robust_scale_: None,
            outlier_mask_: None,
            clean_data_: None,
            breakdown_point_: None,
            m_weights_: None,
            _state: std::marker::PhantomData,
        };

        match &self.strategy {
            RobustStrategy::OutlierResistant {
                contamination,
                detection_method,
            } => {
                fitted.fit_outlier_resistant(y, *contamination, detection_method)?;
            }
            RobustStrategy::TrimmedMean { trim_proportion } => {
                fitted.fit_trimmed_mean(y, *trim_proportion)?;
            }
            RobustStrategy::RobustScale {
                scale_estimator,
                location_estimator,
            } => {
                fitted.fit_robust_scale(y, scale_estimator, location_estimator)?;
            }
            RobustStrategy::BreakdownPoint { breakdown_point } => {
                fitted.fit_breakdown_point(y, *breakdown_point)?;
            }
            RobustStrategy::InfluenceResistant {
                huber_delta,
                max_iter,
                tolerance,
            } => {
                fitted.fit_influence_resistant(y, *huber_delta, *max_iter, *tolerance)?;
            }
        }

        Ok(fitted)
    }
}

impl RobustDummyRegressor<sklears_core::traits::Trained> {
    /// Get the breakdown point achieved by the fitted estimator
    pub fn breakdown_point(&self) -> Option<Float> {
        self.breakdown_point_
    }

    /// Get the outlier mask (available after fitting with outlier-resistant strategy)
    pub fn outlier_mask(&self) -> Option<&Array1<bool>> {
        self.outlier_mask_.as_ref()
    }

    /// Get the M-estimator weights (available after fitting with influence-resistant strategy)
    pub fn m_weights(&self) -> Option<&Array1<Float>> {
        self.m_weights_.as_ref()
    }
    /// Fit outlier-resistant strategy
    fn fit_outlier_resistant(
        &mut self,
        y: &Array1<Float>,
        contamination: Float,
        detection_method: &OutlierDetectionMethod,
    ) -> Result<()> {
        let outlier_mask = self.detect_outliers(y, detection_method, contamination)?;
        let clean_data: Array1<Float> = y
            .iter()
            .zip(outlier_mask.iter())
            .filter_map(|(&value, &is_outlier)| if !is_outlier { Some(value) } else { None })
            .collect();

        if clean_data.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "All data points detected as outliers".to_string(),
            ));
        }

        let location = self.compute_median(&clean_data);
        let scale = self.compute_mad(&clean_data, location);

        // Calculate actual breakdown point achieved
        let n_outliers = outlier_mask.iter().filter(|&&x| x).count();
        let breakdown_point = n_outliers as Float / y.len() as Float;

        self.robust_location_ = Some(location);
        self.robust_scale_ = Some(scale);
        self.outlier_mask_ = Some(outlier_mask);
        self.clean_data_ = Some(clean_data);
        self.breakdown_point_ = Some(breakdown_point);

        Ok(())
    }

    /// Fit trimmed mean strategy
    fn fit_trimmed_mean(&mut self, y: &Array1<Float>, trim_proportion: Float) -> Result<()> {
        if !(0.0..0.5).contains(&trim_proportion) {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Trim proportion must be between 0 and 0.5".to_string(),
            ));
        }

        let mut sorted_y = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_y.len();
        let trim_count = (n as Float * trim_proportion).floor() as usize;

        if trim_count * 2 >= n {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Too much trimming for dataset size".to_string(),
            ));
        }

        let trimmed_data = &sorted_y[trim_count..(n - trim_count)];
        let location = trimmed_data.iter().sum::<Float>() / trimmed_data.len() as Float;

        // Robust scale using trimmed standard deviation
        let mean = location;
        let variance = trimmed_data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<Float>()
            / (trimmed_data.len() - 1) as Float;
        let scale = variance.sqrt();

        let breakdown_point = trim_proportion;

        self.robust_location_ = Some(location);
        self.robust_scale_ = Some(scale);
        self.breakdown_point_ = Some(breakdown_point);

        Ok(())
    }

    /// Fit robust scale strategy
    fn fit_robust_scale(
        &mut self,
        y: &Array1<Float>,
        scale_estimator: &ScaleEstimator,
        location_estimator: &LocationEstimator,
    ) -> Result<()> {
        let location = self.compute_robust_location(y, location_estimator)?;
        let scale = self.compute_robust_scale(y, scale_estimator, location)?;

        // Breakdown point depends on the estimators used
        let breakdown_point = match (location_estimator, scale_estimator) {
            (LocationEstimator::Median, ScaleEstimator::MAD) => 0.5,
            (LocationEstimator::Median, ScaleEstimator::Qn) => 0.5,
            (LocationEstimator::TrimmedMean { trim_proportion }, _) => *trim_proportion,
            _ => 0.25, // Conservative estimate
        };

        self.robust_location_ = Some(location);
        self.robust_scale_ = Some(scale);
        self.breakdown_point_ = Some(breakdown_point);

        Ok(())
    }

    /// Fit breakdown point strategy
    fn fit_breakdown_point(&mut self, y: &Array1<Float>, target_breakdown: Float) -> Result<()> {
        if target_breakdown <= 0.0 || target_breakdown >= 0.5 {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Breakdown point must be between 0 and 0.5".to_string(),
            ));
        }

        // Use trimmed mean with appropriate trimming to achieve target breakdown point
        let trim_proportion = target_breakdown;
        self.fit_trimmed_mean(y, trim_proportion)?;

        Ok(())
    }

    /// Fit influence-resistant strategy using M-estimators
    fn fit_influence_resistant(
        &mut self,
        y: &Array1<Float>,
        huber_delta: Float,
        max_iter: usize,
        tolerance: Float,
    ) -> Result<()> {
        // Initialize with median
        let mut location = self.compute_median(y);
        let initial_scale = self.compute_mad(y, location);

        let mut weights = Array1::ones(y.len());

        // Iteratively reweighted least squares (IRLS) for Huber M-estimator
        for _iter in 0..max_iter {
            let old_location = location;

            // Update weights based on Huber function
            for i in 0..y.len() {
                let residual = (y[i] - location).abs();
                let scaled_residual = residual / initial_scale;

                weights[i] = if scaled_residual <= huber_delta {
                    1.0
                } else {
                    huber_delta / scaled_residual
                };
            }

            // Update location estimate
            let weighted_sum: Float = y.iter().zip(weights.iter()).map(|(&yi, &wi)| wi * yi).sum();
            let weight_sum: Float = weights.sum();

            if weight_sum > 0.0 {
                location = weighted_sum / weight_sum;
            }

            // Check convergence
            if (location - old_location).abs() < tolerance {
                break;
            }
        }

        // Compute robust scale with final weights
        let weighted_variance: Float = y
            .iter()
            .zip(weights.iter())
            .map(|(&yi, &wi)| wi * (yi - location).powi(2))
            .sum();
        let effective_sample_size: Float = weights.sum();

        let scale = if effective_sample_size > 1.0 {
            (weighted_variance / (effective_sample_size - 1.0)).sqrt()
        } else {
            initial_scale
        };

        // Breakdown point for Huber M-estimator is approximately 1/(2*delta + 1)
        let breakdown_point = 1.0 / (2.0 * huber_delta + 1.0);

        self.robust_location_ = Some(location);
        self.robust_scale_ = Some(scale);
        self.m_weights_ = Some(weights);
        self.breakdown_point_ = Some(breakdown_point);

        Ok(())
    }

    /// Detect outliers using various methods
    fn detect_outliers(
        &self,
        y: &Array1<Float>,
        method: &OutlierDetectionMethod,
        _contamination: Float,
    ) -> Result<Array1<bool>> {
        let mut outlier_mask = Array1::from_elem(y.len(), false);

        match method {
            OutlierDetectionMethod::IQR { multiplier } => {
                let mut sorted_y = y.to_vec();
                sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let n = sorted_y.len();
                let q1_idx = n / 4;
                let q3_idx = 3 * n / 4;

                let q1 = sorted_y[q1_idx];
                let q3 = sorted_y[q3_idx];
                let iqr = q3 - q1;

                let lower_bound = q1 - multiplier * iqr;
                let upper_bound = q3 + multiplier * iqr;

                for (i, &value) in y.iter().enumerate() {
                    if value < lower_bound || value > upper_bound {
                        outlier_mask[i] = true;
                    }
                }
            }
            OutlierDetectionMethod::ModifiedZScore { threshold } => {
                let median = self.compute_median(y);
                let mad = self.compute_mad(y, median);

                if mad > 0.0 {
                    for (i, &value) in y.iter().enumerate() {
                        let modified_z_score = 0.6745 * (value - median).abs() / mad;
                        if modified_z_score > *threshold {
                            outlier_mask[i] = true;
                        }
                    }
                }
            }
            OutlierDetectionMethod::MedianDistance { threshold } => {
                let median = self.compute_median(y);
                let distances: Array1<Float> =
                    y.iter().map(|&value| (value - median).abs()).collect();
                let distance_threshold = self.compute_median(&distances) * threshold;

                for (i, &distance) in distances.iter().enumerate() {
                    if distance > distance_threshold {
                        outlier_mask[i] = true;
                    }
                }
            }
        }

        Ok(outlier_mask)
    }

    /// Compute median
    fn compute_median(&self, data: &Array1<Float>) -> Float {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len();
        if n % 2 == 0 {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
        } else {
            sorted_data[n / 2]
        }
    }

    /// Compute Median Absolute Deviation (MAD)
    fn compute_mad(&self, data: &Array1<Float>, median: Float) -> Float {
        let deviations: Array1<Float> = data.iter().map(|&x| (x - median).abs()).collect();
        self.compute_median(&deviations) * 1.4826 // Consistency factor for normal distribution
    }

    /// Compute robust location estimate
    fn compute_robust_location(
        &self,
        y: &Array1<Float>,
        estimator: &LocationEstimator,
    ) -> Result<Float> {
        match estimator {
            LocationEstimator::Median => Ok(self.compute_median(y)),
            LocationEstimator::TrimmedMean { trim_proportion } => {
                let mut sorted_y = y.to_vec();
                sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let n = sorted_y.len();
                let trim_count = (n as Float * trim_proportion).floor() as usize;

                if trim_count * 2 >= n {
                    return Err(sklears_core::error::SklearsError::InvalidInput(
                        "Too much trimming for dataset size".to_string(),
                    ));
                }

                let trimmed_data = &sorted_y[trim_count..(n - trim_count)];
                Ok(trimmed_data.iter().sum::<Float>() / trimmed_data.len() as Float)
            }
            LocationEstimator::Huber { delta } => {
                // Simplified Huber M-estimator
                let mut location = self.compute_median(y);
                let scale = self.compute_mad(y, location);

                for _iter in 0..10 {
                    let old_location = location;
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for &yi in y.iter() {
                        let residual = (yi - location).abs();
                        let weight = if residual <= delta * scale {
                            1.0
                        } else {
                            delta * scale / residual
                        };

                        weighted_sum += weight * yi;
                        weight_sum += weight;
                    }

                    if weight_sum > 0.0 {
                        location = weighted_sum / weight_sum;
                    }

                    if (location - old_location).abs() < 1e-6 {
                        break;
                    }
                }

                Ok(location)
            }
            LocationEstimator::Biweight => {
                // Biweight midvariance (simplified implementation)
                let median = self.compute_median(y);
                let mad = self.compute_mad(y, median);

                if mad == 0.0 {
                    return Ok(median);
                }

                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for &yi in y.iter() {
                    let u = (yi - median) / (9.0 * mad);
                    if u.abs() < 1.0 {
                        let weight = (1.0 - u * u).powi(2);
                        weighted_sum += weight * yi;
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    Ok(weighted_sum / weight_sum)
                } else {
                    Ok(median)
                }
            }
        }
    }

    /// Compute robust scale estimate
    fn compute_robust_scale(
        &self,
        y: &Array1<Float>,
        estimator: &ScaleEstimator,
        location: Float,
    ) -> Result<Float> {
        match estimator {
            ScaleEstimator::MAD => Ok(self.compute_mad(y, location)),
            ScaleEstimator::IQR => {
                let mut sorted_y = y.to_vec();
                sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let n = sorted_y.len();
                let q1_idx = n / 4;
                let q3_idx = 3 * n / 4;

                let q1 = sorted_y[q1_idx];
                let q3 = sorted_y[q3_idx];
                Ok((q3 - q1) / 1.349) // Consistency factor for normal distribution
            }
            ScaleEstimator::Qn => {
                // Simplified Qn estimator (first quartile of pairwise distances)
                let mut pairwise_distances = Vec::new();
                for i in 0..y.len() {
                    for j in (i + 1)..y.len() {
                        pairwise_distances.push((y[i] - y[j]).abs());
                    }
                }

                if pairwise_distances.is_empty() {
                    return Ok(0.0);
                }

                pairwise_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q1_idx = pairwise_distances.len() / 4;
                Ok(pairwise_distances[q1_idx] * 2.2219) // Consistency factor
            }
            ScaleEstimator::Sn => {
                // Simplified Sn estimator (median of medians of distances)
                let mut medians = Vec::new();

                for i in 0..y.len() {
                    let mut distances: Vec<Float> = y
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &yj)| (y[i] - yj).abs())
                        .collect();

                    if !distances.is_empty() {
                        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median_dist = if distances.len() % 2 == 0 {
                            (distances[distances.len() / 2 - 1] + distances[distances.len() / 2])
                                / 2.0
                        } else {
                            distances[distances.len() / 2]
                        };
                        medians.push(median_dist);
                    }
                }

                if medians.is_empty() {
                    return Ok(0.0);
                }

                medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let result = if medians.len() % 2 == 0 {
                    (medians[medians.len() / 2 - 1] + medians[medians.len() / 2]) / 2.0
                } else {
                    medians[medians.len() / 2]
                };

                Ok(result * 1.1926) // Consistency factor
            }
        }
    }
}

impl Predict<Features, Array1<Float>> for RobustDummyRegressor<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Float>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);
        let location = self.robust_location_.unwrap_or(0.0);
        let scale = self.robust_scale_.unwrap_or(1.0);

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        match &self.strategy {
            RobustStrategy::OutlierResistant { .. }
            | RobustStrategy::TrimmedMean { .. }
            | RobustStrategy::RobustScale { .. }
            | RobustStrategy::BreakdownPoint { .. } => {
                // For most strategies, predict the robust location estimate
                predictions.fill(location);
            }
            RobustStrategy::InfluenceResistant { .. } => {
                // For influence-resistant methods, we can add some controlled noise
                // based on the robust scale estimate
                if scale > 0.0 {
                    let normal = Normal::new(location, scale * 0.1).unwrap();
                    for i in 0..n_samples {
                        predictions[i] = normal.sample(&mut rng);
                    }
                } else {
                    predictions.fill(location);
                }
            }
        }

        Ok(predictions)
    }
}

/// Robust dummy classifier
#[derive(Debug, Clone)]
pub struct RobustDummyClassifier<State = sklears_core::traits::Untrained> {
    /// Strategy for robust predictions
    pub strategy: RobustStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,

    // Fitted parameters
    pub(crate) robust_class_probs_: Option<Array1<Float>>,
    pub(crate) classes_: Option<Array1<i32>>,
    pub(crate) outlier_mask_: Option<Array1<bool>>,

    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl RobustDummyClassifier {
    /// Create a new robust dummy classifier
    pub fn new(strategy: RobustStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            robust_class_probs_: None,
            classes_: None,
            outlier_mask_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Get the outlier mask (available after fitting)
    pub fn outlier_mask(&self) -> Option<&Array1<bool>> {
        self.outlier_mask_.as_ref()
    }
}

impl Default for RobustDummyClassifier {
    fn default() -> Self {
        Self::new(RobustStrategy::OutlierResistant {
            contamination: 0.1,
            detection_method: OutlierDetectionMethod::IQR { multiplier: 1.5 },
        })
    }
}

impl Estimator for RobustDummyClassifier {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<i32>> for RobustDummyClassifier {
    type Fitted = RobustDummyClassifier<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        // Get unique classes
        let mut unique_classes = y.iter().cloned().collect::<Vec<_>>();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);

        // For classification, we'll use a simplified approach:
        // Remove outliers and compute robust class probabilities
        let mut fitted = RobustDummyClassifier {
            strategy: self.strategy.clone(),
            random_state: self.random_state,
            robust_class_probs_: None,
            classes_: Some(classes.clone()),
            outlier_mask_: None,
            _state: std::marker::PhantomData,
        };

        // Simple outlier detection based on class frequency
        let mut class_counts = std::collections::HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        // Identify outlier classes (those with very low frequency)
        let total_samples = y.len() as f64;
        let min_frequency = 0.05; // Classes with < 5% frequency are considered outliers
        let mut outlier_mask = Array1::from_elem(y.len(), false);

        for (i, &class) in y.iter().enumerate() {
            let class_freq = *class_counts.get(&class).unwrap() as f64 / total_samples;
            if class_freq < min_frequency {
                outlier_mask[i] = true;
            }
        }

        // Compute robust class probabilities excluding outliers
        let mut robust_class_counts = std::collections::HashMap::new();
        let mut total_clean_samples = 0;

        for (i, &class) in y.iter().enumerate() {
            if !outlier_mask[i] {
                *robust_class_counts.entry(class).or_insert(0) += 1;
                total_clean_samples += 1;
            }
        }

        let mut class_probs = Array1::zeros(classes.len());
        for (i, &class) in classes.iter().enumerate() {
            let count = *robust_class_counts.get(&class).unwrap_or(&0);
            class_probs[i] = if total_clean_samples > 0 {
                count as Float / total_clean_samples as Float
            } else {
                1.0 / classes.len() as Float
            };
        }

        fitted.robust_class_probs_ = Some(class_probs);
        fitted.outlier_mask_ = Some(outlier_mask);

        Ok(fitted)
    }
}

impl RobustDummyClassifier<sklears_core::traits::Trained> {
    /// Get the outlier mask (available after fitting)
    pub fn outlier_mask(&self) -> Option<&Array1<bool>> {
        self.outlier_mask_.as_ref()
    }
}

impl Predict<Features, Array1<i32>> for RobustDummyClassifier<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<i32>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        let classes = self.classes_.as_ref().unwrap();
        let class_probs = self.robust_class_probs_.as_ref().unwrap();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        // Sample from robust class distribution
        for i in 0..n_samples {
            let rand_val: Float = rng.gen();
            let mut cumulative_prob = 0.0;
            let mut selected_class = classes[0];

            for (j, &class) in classes.iter().enumerate() {
                cumulative_prob += class_probs[j];
                if rand_val <= cumulative_prob {
                    selected_class = class;
                    break;
                }
            }
            predictions[i] = selected_class;
        }

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_outlier_resistant_regressor() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                100.0, 101.0, 102.0, 103.0, // Last two rows are outliers
            ],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0, 101.0]; // Last two are outliers

        let regressor = RobustDummyRegressor::new(RobustStrategy::OutlierResistant {
            contamination: 0.2,
            detection_method: OutlierDetectionMethod::IQR { multiplier: 1.5 },
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 10);

        // Check that some outliers were detected
        let outlier_mask = fitted.outlier_mask().unwrap();
        let n_outliers = outlier_mask.iter().filter(|&&x| x).count();
        assert!(n_outliers > 0);
    }

    #[test]
    fn test_trimmed_mean_regressor() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0, 101.0];

        let regressor = RobustDummyRegressor::new(RobustStrategy::TrimmedMean {
            trim_proportion: 0.2,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 10);

        // Trimmed mean should be more robust than regular mean
        let robust_mean = predictions[0];
        let regular_mean = y.mean().unwrap();
        assert!(robust_mean < regular_mean); // Should be less affected by outliers
    }

    #[test]
    fn test_robust_scale_regressor() {
        let x = Array2::from_shape_vec((8, 2), (0..16).map(|x| x as f64).collect()).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]; // Last value is outlier

        let regressor = RobustDummyRegressor::new(RobustStrategy::RobustScale {
            scale_estimator: ScaleEstimator::MAD,
            location_estimator: LocationEstimator::Median,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);

        // Should predict median value
        let mut sorted_y = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected_median = (sorted_y[3] + sorted_y[4]) / 2.0; // 4.5
        assert_abs_diff_eq!(predictions[0], expected_median, epsilon = 0.1);
    }

    #[test]
    fn test_breakdown_point_regressor() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let regressor = RobustDummyRegressor::new(RobustStrategy::BreakdownPoint {
            breakdown_point: 0.3,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 10);
        assert_eq!(fitted.breakdown_point().unwrap(), 0.3);
    }

    #[test]
    fn test_influence_resistant_regressor() {
        let x = Array2::from_shape_vec((8, 2), (0..16).map(|x| x as f64).collect()).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]; // Last value is outlier

        let regressor = RobustDummyRegressor::new(RobustStrategy::InfluenceResistant {
            huber_delta: 1.345,
            max_iter: 50,
            tolerance: 1e-6,
        });

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);

        // Should have M-estimator weights
        let weights = fitted.m_weights().unwrap();
        assert_eq!(weights.len(), 8);

        // Outlier should have lower weight
        assert!(weights[7] < weights[0]);
    }

    #[test]
    fn test_robust_classifier() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 3, 3]; // Class 3 is less frequent

        let classifier = RobustDummyClassifier::new(RobustStrategy::OutlierResistant {
            contamination: 0.1,
            detection_method: OutlierDetectionMethod::IQR { multiplier: 1.5 },
        })
        .with_random_state(42);

        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 10);

        // Check that predictions are valid classes
        let classes = fitted.classes_.as_ref().unwrap();
        for &pred in predictions.iter() {
            assert!(classes.iter().any(|&c| c == pred));
        }
    }
}
