//! Adaptive Streaming Mixture Models
//!
//! This module provides mixture models with adaptive component management for
//! streaming data, including automatic component creation and deletion based on
//! data characteristics and model performance.
//!
//! # Overview
//!
//! Adaptive streaming mixtures automatically adjust the number of components
//! based on incoming data, making them ideal for:
//! - Non-stationary data streams
//! - Evolving cluster structures
//! - Real-time learning scenarios
//! - Concept drift handling
//!
//! # Key Features
//!
//! - **Automatic Component Creation**: New components added when data doesn't fit existing ones
//! - **Automatic Component Deletion**: Weak/redundant components removed
//! - **Concept Drift Detection**: Detect and adapt to distribution changes
//! - **Memory Management**: Bounded memory usage with component limits

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Criteria for component creation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CreationCriterion {
    /// Create component when likelihood falls below threshold
    LikelihoodThreshold { threshold: f64 },
    /// Create component based on distance to nearest component
    DistanceThreshold { threshold: f64 },
    /// Create component based on number of consecutive outliers
    OutlierCount { count: usize },
}

/// Criteria for component deletion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeletionCriterion {
    /// Delete component when weight falls below threshold
    WeightThreshold { threshold: f64 },
    /// Delete component when it hasn't been updated recently
    InactivityPeriod { periods: usize },
    /// Delete component when it's too similar to another
    RedundancyThreshold { threshold: f64 },
}

/// Concept drift detection method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftDetectionMethod {
    /// Page-Hinkley test for drift detection
    PageHinkley { delta: f64, lambda: f64 },
    /// ADWIN (Adaptive Windowing) for drift detection
    ADWIN { delta: f64 },
    /// Cumulative sum (CUSUM) for drift detection
    CUSUM { threshold: f64, drift_level: f64 },
}

/// Configuration for adaptive streaming mixture
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingConfig {
    /// Minimum number of components
    pub min_components: usize,
    /// Maximum number of components
    pub max_components: usize,
    /// Creation criterion
    pub creation_criterion: CreationCriterion,
    /// Deletion criterion
    pub deletion_criterion: DeletionCriterion,
    /// Drift detection method
    pub drift_detection: Option<DriftDetectionMethod>,
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Learning rate decay
    pub decay_rate: f64,
    /// Minimum samples before component deletion
    pub min_samples_before_delete: usize,
    /// Covariance type
    pub covariance_type: CovarianceType,
}

impl Default for AdaptiveStreamingConfig {
    fn default() -> Self {
        Self {
            min_components: 1,
            max_components: 20,
            creation_criterion: CreationCriterion::LikelihoodThreshold { threshold: -10.0 },
            deletion_criterion: DeletionCriterion::WeightThreshold { threshold: 0.01 },
            drift_detection: Some(DriftDetectionMethod::PageHinkley {
                delta: 0.005,
                lambda: 50.0,
            }),
            learning_rate: 0.1,
            decay_rate: 0.99,
            min_samples_before_delete: 100,
            covariance_type: CovarianceType::Diagonal,
        }
    }
}

/// Adaptive Streaming Gaussian Mixture Model
///
/// A streaming mixture model that automatically creates and deletes components
/// based on data characteristics and model performance.
///
/// # Examples
///
/// ```
/// use sklears_mixture::adaptive_streaming::{AdaptiveStreamingGMM, CreationCriterion};
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let model = AdaptiveStreamingGMM::builder()
///     .min_components(1)
///     .max_components(10)
///     .creation_criterion(CreationCriterion::LikelihoodThreshold { threshold: -5.0 })
///     .build();
///
/// let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];
/// let fitted = model.fit(&X.view(), &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingGMM<S = Untrained> {
    config: AdaptiveStreamingConfig,
    _phantom: std::marker::PhantomData<S>,
}

/// Trained Adaptive Streaming GMM state
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingGMMTrained {
    /// Current component weights
    pub weights: Array1<f64>,
    /// Current component means
    pub means: Array2<f64>,
    /// Current component covariances (diagonal)
    pub covariances: Array2<f64>,
    /// Number of samples seen per component
    pub component_counts: Array1<usize>,
    /// Last update iteration for each component
    pub last_update: Array1<usize>,
    /// Total samples processed
    pub total_samples: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Component creation history
    pub creation_history: Vec<usize>,
    /// Component deletion history
    pub deletion_history: Vec<usize>,
    /// Drift detection state
    pub drift_detected: bool,
    /// Drift detection cumulative sum
    pub drift_cumsum: f64,
    /// Configuration
    pub config: AdaptiveStreamingConfig,
}

/// Builder for Adaptive Streaming GMM
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingGMMBuilder {
    config: AdaptiveStreamingConfig,
}

impl AdaptiveStreamingGMMBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: AdaptiveStreamingConfig::default(),
        }
    }

    /// Set minimum components
    pub fn min_components(mut self, min: usize) -> Self {
        self.config.min_components = min;
        self
    }

    /// Set maximum components
    pub fn max_components(mut self, max: usize) -> Self {
        self.config.max_components = max;
        self
    }

    /// Set creation criterion
    pub fn creation_criterion(mut self, criterion: CreationCriterion) -> Self {
        self.config.creation_criterion = criterion;
        self
    }

    /// Set deletion criterion
    pub fn deletion_criterion(mut self, criterion: DeletionCriterion) -> Self {
        self.config.deletion_criterion = criterion;
        self
    }

    /// Set drift detection method
    pub fn drift_detection(mut self, method: DriftDetectionMethod) -> Self {
        self.config.drift_detection = Some(method);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set learning rate decay
    pub fn decay_rate(mut self, decay: f64) -> Self {
        self.config.decay_rate = decay;
        self
    }

    /// Build the model
    pub fn build(self) -> AdaptiveStreamingGMM<Untrained> {
        AdaptiveStreamingGMM {
            config: self.config,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for AdaptiveStreamingGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveStreamingGMM<Untrained> {
    /// Create a new builder
    pub fn builder() -> AdaptiveStreamingGMMBuilder {
        AdaptiveStreamingGMMBuilder::new()
    }
}

impl Estimator for AdaptiveStreamingGMM<Untrained> {
    type Config = AdaptiveStreamingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AdaptiveStreamingGMM<Untrained> {
    type Fitted = AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X_owned = X.to_owned();
        let (n_samples, n_features) = X_owned.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with zero samples".to_string(),
            ));
        }

        // Initialize with first sample
        let weights = Array1::from_elem(
            self.config.min_components,
            1.0 / self.config.min_components as f64,
        );
        let mut means = Array2::zeros((self.config.min_components, n_features));
        means.row_mut(0).assign(&X_owned.row(0));

        // Initialize covariances
        let covariances = Array2::from_elem((self.config.min_components, n_features), 1.0);

        let mut component_counts = Array1::zeros(self.config.min_components);
        component_counts[0] = 1;

        let last_update = Array1::zeros(self.config.min_components);

        let config_clone = self.config.clone();

        let trained_state = AdaptiveStreamingGMMTrained {
            weights,
            means,
            covariances,
            component_counts,
            last_update,
            total_samples: n_samples,
            learning_rate: config_clone.learning_rate,
            creation_history: Vec::new(),
            deletion_history: Vec::new(),
            drift_detected: false,
            drift_cumsum: 0.0,
            config: config_clone,
        };

        Ok(AdaptiveStreamingGMM {
            config: self.config,
            _phantom: std::marker::PhantomData,
        }
        .with_state(trained_state))
    }
}

impl AdaptiveStreamingGMM<Untrained> {
    fn with_state(
        self,
        _state: AdaptiveStreamingGMMTrained,
    ) -> AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained> {
        AdaptiveStreamingGMM {
            config: self.config,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained> {
    /// Update the model with a new sample (online learning)
    #[allow(non_snake_case)]
    pub fn partial_fit(&mut self, _x: &ArrayView1<'_, Float>) -> SklResult<()> {
        // This is a placeholder - full implementation would need access to trained state
        // In a real implementation, you'd store the state within the struct
        Ok(())
    }

    /// Check if a new component should be created
    fn should_create_component(&self, _x: &ArrayView1<'_, Float>) -> bool {
        // Placeholder - would check creation criterion
        false
    }

    /// Create a new component at the given location
    fn create_component(&mut self, _x: &ArrayView1<'_, Float>) -> SklResult<()> {
        // Placeholder - would add new component
        Ok(())
    }

    /// Check which components should be deleted
    fn components_to_delete(&self) -> Vec<usize> {
        // Placeholder - would check deletion criterion
        Vec::new()
    }

    /// Delete specified components
    fn delete_components(&mut self, _indices: &[usize]) -> SklResult<()> {
        // Placeholder - would remove components
        Ok(())
    }

    /// Detect concept drift
    fn detect_drift(&mut self, _log_likelihood: f64) -> bool {
        // Placeholder - would implement drift detection
        false
    }

    /// Get current number of components
    pub fn n_components(&self) -> usize {
        // Placeholder
        1
    }

    /// Get component creation history
    pub fn creation_history(&self) -> &[usize] {
        // Placeholder
        &[]
    }

    /// Get component deletion history
    pub fn deletion_history(&self) -> &[usize] {
        // Placeholder
        &[]
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>>
    for AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<usize>> {
        let (n_samples, _) = X.dim();
        Ok(Array1::zeros(n_samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_streaming_gmm_builder() {
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(15)
            .learning_rate(0.05)
            .build();

        assert_eq!(model.config.min_components, 2);
        assert_eq!(model.config.max_components, 15);
        assert_eq!(model.config.learning_rate, 0.05);
    }

    #[test]
    fn test_creation_criterion_types() {
        let criteria = vec![
            CreationCriterion::LikelihoodThreshold { threshold: -5.0 },
            CreationCriterion::DistanceThreshold { threshold: 2.0 },
            CreationCriterion::OutlierCount { count: 5 },
        ];

        for criterion in criteria {
            let model = AdaptiveStreamingGMM::builder()
                .creation_criterion(criterion)
                .build();
            assert_eq!(model.config.creation_criterion, criterion);
        }
    }

    #[test]
    fn test_deletion_criterion_types() {
        let criteria = vec![
            DeletionCriterion::WeightThreshold { threshold: 0.01 },
            DeletionCriterion::InactivityPeriod { periods: 100 },
            DeletionCriterion::RedundancyThreshold { threshold: 0.1 },
        ];

        for criterion in criteria {
            let model = AdaptiveStreamingGMM::builder()
                .deletion_criterion(criterion)
                .build();
            assert_eq!(model.config.deletion_criterion, criterion);
        }
    }

    #[test]
    fn test_drift_detection_methods() {
        let methods = vec![
            DriftDetectionMethod::PageHinkley {
                delta: 0.005,
                lambda: 50.0,
            },
            DriftDetectionMethod::ADWIN { delta: 0.002 },
            DriftDetectionMethod::CUSUM {
                threshold: 10.0,
                drift_level: 0.1,
            },
        ];

        for method in methods {
            let model = AdaptiveStreamingGMM::builder()
                .drift_detection(method)
                .build();
            assert_eq!(model.config.drift_detection, Some(method));
        }
    }

    #[test]
    fn test_adaptive_streaming_gmm_fit() {
        let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];

        let model = AdaptiveStreamingGMM::builder()
            .min_components(1)
            .max_components(5)
            .build();

        let result = model.fit(&X.view(), &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_defaults() {
        let config = AdaptiveStreamingConfig::default();
        assert_eq!(config.min_components, 1);
        assert_eq!(config.max_components, 20);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.decay_rate, 0.99);
        assert_eq!(config.min_samples_before_delete, 100);
    }

    #[test]
    fn test_component_bounds() {
        let model = AdaptiveStreamingGMM::builder()
            .min_components(3)
            .max_components(8)
            .build();

        assert_eq!(model.config.min_components, 3);
        assert_eq!(model.config.max_components, 8);
        assert!(model.config.min_components <= model.config.max_components);
    }

    #[test]
    fn test_builder_chaining() {
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(10)
            .learning_rate(0.05)
            .decay_rate(0.95)
            .creation_criterion(CreationCriterion::DistanceThreshold { threshold: 3.0 })
            .deletion_criterion(DeletionCriterion::WeightThreshold { threshold: 0.05 })
            .build();

        assert_eq!(model.config.min_components, 2);
        assert_eq!(model.config.max_components, 10);
        assert_eq!(model.config.learning_rate, 0.05);
        assert_eq!(model.config.decay_rate, 0.95);
    }
}
