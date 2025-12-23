//! Online Learning Framework for Incremental Multiclass Classification
//!
//! Provides core functionality for online and incremental learning algorithms
//! that can update their models with new data points without full retraining.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    prelude::FloatBounds,
};
use std::marker::PhantomData;

/// Configuration for online learning algorithms
#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Maximum number of classes to handle
    pub max_classes: Option<usize>,
    /// Whether to use adaptive learning rates
    pub adaptive: bool,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_classes: None,
            adaptive: false,
        }
    }
}

/// Online learning capability for multiclass classifiers
///
/// This trait provides the interface for incremental learning algorithms
/// that can update their parameters with new data points.
pub trait OnlineLearning<T: FloatBounds> {
    /// Update the model with a single data point
    fn partial_fit_single(&mut self, x: &Array1<T>, y: usize) -> SklResult<()>;

    /// Update the model with a batch of data points
    fn partial_fit_batch(&mut self, x: &Array2<T>, y: &Array1<usize>) -> SklResult<()>;

    /// Reset the model to initial state
    fn reset(&mut self);

    /// Get the current number of classes seen
    fn n_classes_seen(&self) -> usize;
}

/// Basic online learner implementation
#[derive(Debug, Clone)]
pub struct OnlineLearner<T: FloatBounds> {
    config: OnlineLearningConfig,
    n_features: Option<usize>,
    classes_seen: Vec<usize>,
    weights: Option<Array2<T>>,
}

impl<T: FloatBounds> OnlineLearner<T> {
    /// Create a new online learner
    pub fn new(config: OnlineLearningConfig) -> Self {
        Self {
            config,
            n_features: None,
            classes_seen: Vec::new(),
            weights: None,
        }
    }

    /// Initialize the learner with feature dimensions
    fn initialize(&mut self, n_features: usize) -> SklResult<()> {
        self.n_features = Some(n_features);
        Ok(())
    }

    /// Add a new class if not seen before
    fn add_class(&mut self, class: usize) {
        if !self.classes_seen.contains(&class) {
            self.classes_seen.push(class);
            self.classes_seen.sort_unstable();
        }
    }
}

impl<T: FloatBounds> OnlineLearning<T> for OnlineLearner<T> {
    fn partial_fit_single(&mut self, x: &Array1<T>, y: usize) -> SklResult<()> {
        if self.n_features.is_none() {
            self.initialize(x.len())?;
        }

        let n_features = self.n_features.unwrap();
        if x.len() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.len()
            )));
        }

        self.add_class(y);
        Ok(())
    }

    fn partial_fit_batch(&mut self, x: &Array2<T>, y: &Array1<usize>) -> SklResult<()> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        for i in 0..x.nrows() {
            let row = x.row(i);
            self.partial_fit_single(&row.to_owned(), y[i])?;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.n_features = None;
        self.classes_seen.clear();
        self.weights = None;
    }

    fn n_classes_seen(&self) -> usize {
        self.classes_seen.len()
    }
}

/// Builder for online learner configuration
pub struct OnlineLearnerBuilder<T: FloatBounds> {
    config: OnlineLearningConfig,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds> OnlineLearnerBuilder<T> {
    pub fn new() -> Self {
        Self {
            config: OnlineLearningConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.config.learning_rate = rate;
        self
    }

    /// Set the maximum number of classes
    pub fn max_classes(mut self, max_classes: usize) -> Self {
        self.config.max_classes = Some(max_classes);
        self
    }

    /// Enable adaptive learning rates
    pub fn adaptive(mut self) -> Self {
        self.config.adaptive = true;
        self
    }

    /// Build the online learner
    pub fn build(self) -> OnlineLearner<T> {
        OnlineLearner::new(self.config)
    }
}

impl<T: FloatBounds> Default for OnlineLearnerBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_online_learner_creation() {
        let learner: OnlineLearner<f64> = OnlineLearner::new(OnlineLearningConfig::default());
        assert_eq!(learner.n_classes_seen(), 0);
    }

    #[test]
    fn test_partial_fit_single() {
        let mut learner: OnlineLearner<f64> = OnlineLearner::new(OnlineLearningConfig::default());
        let x = array![1.0, 2.0, 3.0];

        assert!(learner.partial_fit_single(&x, 0).is_ok());
        assert_eq!(learner.n_classes_seen(), 1);
    }

    #[test]
    fn test_partial_fit_batch() {
        let mut learner: OnlineLearner<f64> = OnlineLearner::new(OnlineLearningConfig::default());
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0, 1, 0];

        assert!(learner.partial_fit_batch(&x, &y).is_ok());
        assert_eq!(learner.n_classes_seen(), 2);
    }

    #[test]
    fn test_builder_pattern() {
        let learner: OnlineLearner<f64> = OnlineLearnerBuilder::new()
            .learning_rate(0.1)
            .max_classes(10)
            .adaptive()
            .build();

        assert_eq!(learner.config.learning_rate, 0.1);
        assert_eq!(learner.config.max_classes, Some(10));
        assert!(learner.config.adaptive);
    }
}
