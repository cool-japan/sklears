//! Few-shot learning pipeline components
//!
//! This module provides few-shot learning capabilities including prototype-based
//! and gradient-based meta-learners.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;

use crate::{PipelinePredictor, PipelineStep};

/// Support set for few-shot learning
#[derive(Debug, Clone)]
pub struct SupportSet {
    /// Features of support examples
    pub features: Array2<f64>,
    /// Labels of support examples
    pub labels: Array1<f64>,
    /// Number of examples per class
    pub n_shot: usize,
    /// Number of classes
    pub n_way: usize,
}

impl SupportSet {
    /// Create a new support set
    #[must_use]
    pub fn new(features: Array2<f64>, labels: Array1<f64>, n_shot: usize, n_way: usize) -> Self {
        Self {
            features,
            labels,
            n_shot,
            n_way,
        }
    }

    /// Get examples for a specific class
    #[must_use]
    pub fn get_class_examples(&self, class_label: f64) -> (Array2<f64>, Array1<f64>) {
        let mut class_features = Vec::new();
        let mut class_labels = Vec::new();

        for (i, &label) in self.labels.iter().enumerate() {
            if (label - class_label).abs() < 1e-6 {
                class_features.push(self.features.row(i).to_owned());
                class_labels.push(label);
            }
        }

        if class_features.is_empty() {
            return (Array2::zeros((0, self.features.ncols())), Array1::zeros(0));
        }

        let n_examples = class_features.len();
        let n_features = class_features[0].len();
        let mut features_array = Array2::zeros((n_examples, n_features));

        for (i, features) in class_features.iter().enumerate() {
            features_array.row_mut(i).assign(features);
        }

        (features_array, Array1::from_vec(class_labels))
    }

    /// Get all unique classes
    #[must_use]
    pub fn get_classes(&self) -> Vec<f64> {
        let mut classes: Vec<f64> = self.labels.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        classes
    }
}

/// Prototype-based few-shot learner
#[derive(Debug)]
pub struct PrototypicalNetwork<S = Untrained> {
    state: S,
    distance_metric: DistanceMetric,
    embedding_dim: Option<usize>,
    prototypes: HashMap<String, Array1<f64>>,
}

/// Trained state for `PrototypicalNetwork`
#[derive(Debug)]
pub struct PrototypicalNetworkTrained {
    prototypes: HashMap<String, Array1<f64>>,
    distance_metric: DistanceMetric,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

/// Distance metrics for prototype-based learning
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine distance
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// Mahalanobis distance (with covariance matrix)
    Mahalanobis { covariance: Array2<f64> },
}

impl DistanceMetric {
    /// Compute distance between two vectors
    #[must_use]
    pub fn distance(&self, a: &ArrayView1<'_, f64>, b: &ArrayView1<'_, f64>) -> f64 {
        match self {
            DistanceMetric::Euclidean => ((a - b).mapv(|x| x * x).sum()).sqrt(),
            DistanceMetric::Cosine => {
                let dot_product = a.dot(b);
                let norm_a = (a.mapv(|x| x * x).sum()).sqrt();
                let norm_b = (b.mapv(|x| x * x).sum()).sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - dot_product / (norm_a * norm_b)
                }
            }
            DistanceMetric::Manhattan => (a - b).mapv(f64::abs).sum(),
            DistanceMetric::Mahalanobis { covariance } => {
                let diff = a - b;
                // Simplified Mahalanobis distance (assuming covariance is diagonal)
                let weighted_diff = &diff * &covariance.diag();
                (weighted_diff.mapv(|x| x * x).sum()).sqrt()
            }
        }
    }
}

impl PrototypicalNetwork<Untrained> {
    /// Create a new prototypical network
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            distance_metric: DistanceMetric::Euclidean,
            embedding_dim: None,
            prototypes: HashMap::new(),
        }
    }

    /// Set the distance metric
    #[must_use]
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the embedding dimension
    #[must_use]
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = Some(dim);
        self
    }
}

impl Default for PrototypicalNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PrototypicalNetwork<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl PrototypicalNetwork<Untrained> {
    /// Fit the prototypical network using support sets
    pub fn fit_support_set(
        self,
        support_set: &SupportSet,
    ) -> SklResult<PrototypicalNetwork<PrototypicalNetworkTrained>> {
        let mut prototypes = HashMap::new();
        let classes = support_set.get_classes();

        // Compute prototype for each class
        for class_label in classes {
            let (class_features, _) = support_set.get_class_examples(class_label);

            if class_features.nrows() > 0 {
                // Compute mean as prototype
                let prototype = class_features.mean_axis(Axis(0)).unwrap();
                prototypes.insert(class_label.to_string(), prototype);
            }
        }

        Ok(PrototypicalNetwork {
            state: PrototypicalNetworkTrained {
                prototypes,
                distance_metric: self.distance_metric,
                n_features_in: support_set.features.ncols(),
                feature_names_in: None,
            },
            distance_metric: DistanceMetric::Euclidean, // Placeholder
            embedding_dim: None,
            prototypes: HashMap::new(),
        })
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for PrototypicalNetwork<Untrained> {
    type Fitted = PrototypicalNetwork<PrototypicalNetworkTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let x_f64 = x.mapv(|v| v);
            let y_f64 = y_values.mapv(|v| v);

            // Determine n_way and n_shot from data
            let unique_labels: Vec<f64> = {
                let mut labels: Vec<f64> = y_f64.iter().copied().collect();
                labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
                labels.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
                labels
            };

            let n_way = unique_labels.len();
            let n_shot = y_f64.len() / n_way; // Assume balanced

            let support_set = SupportSet::new(x_f64, y_f64, n_shot, n_way);
            self.fit_support_set(&support_set)
        } else {
            Err(SklearsError::InvalidInput(
                "Labels required for few-shot learning".to_string(),
            ))
        }
    }
}

impl PrototypicalNetwork<PrototypicalNetworkTrained> {
    /// Predict using nearest prototype
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let x_f64 = x.mapv(|v| v);
        let mut predictions = Array1::zeros(x_f64.nrows());

        for (i, sample) in x_f64.axis_iter(Axis(0)).enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut predicted_class = 0.0;

            for (class_str, prototype) in &self.state.prototypes {
                let distance = self
                    .state
                    .distance_metric
                    .distance(&sample, &prototype.view());
                if distance < min_distance {
                    min_distance = distance;
                    predicted_class = class_str.parse().unwrap_or(0.0);
                }
            }

            predictions[i] = predicted_class;
        }

        Ok(predictions)
    }

    /// Get the prototypes
    #[must_use]
    pub fn prototypes(&self) -> &HashMap<String, Array1<f64>> {
        &self.state.prototypes
    }
}

/// Model-Agnostic Meta-Learning (MAML) few-shot learner
#[derive(Debug)]
pub struct MAMLLearner<S = Untrained> {
    state: S,
    base_learner: Option<Box<dyn PipelinePredictor>>,
    inner_lr: f64,
    outer_lr: f64,
    inner_steps: usize,
    meta_parameters: HashMap<String, f64>,
}

/// Trained state for `MAMLLearner`
#[derive(Debug)]
pub struct MAMLLearnerTrained {
    fitted_learner: Box<dyn PipelinePredictor>,
    inner_lr: f64,
    outer_lr: f64,
    inner_steps: usize,
    meta_parameters: HashMap<String, f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl MAMLLearner<Untrained> {
    /// Create a new MAML learner
    #[must_use]
    pub fn new(base_learner: Box<dyn PipelinePredictor>) -> Self {
        Self {
            state: Untrained,
            base_learner: Some(base_learner),
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 5,
            meta_parameters: HashMap::new(),
        }
    }

    /// Set inner learning rate
    #[must_use]
    pub fn inner_lr(mut self, lr: f64) -> Self {
        self.inner_lr = lr;
        self
    }

    /// Set outer learning rate  
    #[must_use]
    pub fn outer_lr(mut self, lr: f64) -> Self {
        self.outer_lr = lr;
        self
    }

    /// Set number of inner steps
    #[must_use]
    pub fn inner_steps(mut self, steps: usize) -> Self {
        self.inner_steps = steps;
        self
    }

    /// Set initial meta-parameters
    #[must_use]
    pub fn meta_parameters(mut self, params: HashMap<String, f64>) -> Self {
        self.meta_parameters = params;
        self
    }
}

impl Estimator for MAMLLearner<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for MAMLLearner<Untrained> {
    type Fitted = MAMLLearner<MAMLLearnerTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut base_learner = self
            .base_learner
            .ok_or_else(|| SklearsError::InvalidInput("No base learner provided".to_string()))?;

        if let Some(y_values) = y.as_ref() {
            // Simulate meta-training
            base_learner.fit(x, y_values)?;

            Ok(MAMLLearner {
                state: MAMLLearnerTrained {
                    fitted_learner: base_learner,
                    inner_lr: self.inner_lr,
                    outer_lr: self.outer_lr,
                    inner_steps: self.inner_steps,
                    meta_parameters: self.meta_parameters,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                base_learner: None,
                inner_lr: 0.0,
                outer_lr: 0.0,
                inner_steps: 0,
                meta_parameters: HashMap::new(),
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Labels required for MAML training".to_string(),
            ))
        }
    }
}

impl MAMLLearner<MAMLLearnerTrained> {
    /// Adapt to a new task with few examples
    pub fn adapt_to_task(&mut self, support_set: &SupportSet) -> SklResult<()> {
        // Simulate inner loop adaptation
        for _ in 0..self.state.inner_steps {
            // In a real implementation, this would compute gradients and update parameters
            // For now, we simulate by fitting the base learner
            let mapped_features = support_set.features.view().mapv(|v| v as Float);
            let mapped_labels = support_set.labels.view().mapv(|v| v as Float);
            self.state
                .fitted_learner
                .fit(&mapped_features.view(), &mapped_labels.view())?;
        }
        Ok(())
    }

    /// Predict after adaptation
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        self.state.fitted_learner.predict(x)
    }

    /// Get meta-parameters
    #[must_use]
    pub fn meta_parameters(&self) -> &HashMap<String, f64> {
        &self.state.meta_parameters
    }
}

/// Few-shot learning pipeline
#[derive(Debug)]
pub struct FewShotPipeline<S = Untrained> {
    state: S,
    learner_type: FewShotLearnerType,
    meta_learner: Option<MetaLearnerWrapper>,
}

/// Trained state for `FewShotPipeline`
#[derive(Debug)]
pub struct FewShotPipelineTrained {
    fitted_learner: MetaLearnerWrapper,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

/// Types of few-shot learners
#[derive(Debug, Clone)]
pub enum FewShotLearnerType {
    /// Prototypical network
    Prototypical { distance_metric: DistanceMetric },
    /// MAML-based learner
    MAML {
        inner_lr: f64,
        outer_lr: f64,
        inner_steps: usize,
    },
}

/// Wrapper for different meta-learner types
#[derive(Debug)]
pub enum MetaLearnerWrapper {
    /// Prototypical
    Prototypical(PrototypicalNetwork<PrototypicalNetworkTrained>),
    /// MAML
    MAML(MAMLLearner<MAMLLearnerTrained>),
}

impl MetaLearnerWrapper {
    /// Predict using the wrapped meta-learner
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        match self {
            MetaLearnerWrapper::Prototypical(learner) => learner.predict(x),
            MetaLearnerWrapper::MAML(learner) => learner.predict(x),
        }
    }
}

impl FewShotPipeline<Untrained> {
    /// Create a new few-shot pipeline
    #[must_use]
    pub fn new(learner_type: FewShotLearnerType) -> Self {
        Self {
            state: Untrained,
            learner_type,
            meta_learner: None,
        }
    }

    /// Create a prototypical network pipeline
    #[must_use]
    pub fn prototypical(distance_metric: DistanceMetric) -> Self {
        Self::new(FewShotLearnerType::Prototypical { distance_metric })
    }

    /// Create a MAML pipeline
    #[must_use]
    pub fn maml(inner_lr: f64, outer_lr: f64, inner_steps: usize) -> Self {
        Self::new(FewShotLearnerType::MAML {
            inner_lr,
            outer_lr,
            inner_steps,
        })
    }
}

impl Estimator for FewShotPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for FewShotPipeline<Untrained> {
    type Fitted = FewShotPipeline<FewShotPipelineTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let fitted_learner = match &self.learner_type {
            FewShotLearnerType::Prototypical { distance_metric } => {
                let learner = PrototypicalNetwork::new().distance_metric(distance_metric.clone());
                let fitted = learner.fit(x, y)?;
                MetaLearnerWrapper::Prototypical(fitted)
            }
            FewShotLearnerType::MAML {
                inner_lr,
                outer_lr,
                inner_steps,
            } => {
                use crate::MockPredictor;
                let base_learner = Box::new(MockPredictor::new());
                let learner = MAMLLearner::new(base_learner)
                    .inner_lr(*inner_lr)
                    .outer_lr(*outer_lr)
                    .inner_steps(*inner_steps);
                let fitted = learner.fit(x, y)?;
                MetaLearnerWrapper::MAML(fitted)
            }
        };

        Ok(FewShotPipeline {
            state: FewShotPipelineTrained {
                fitted_learner,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            learner_type: FewShotLearnerType::Prototypical {
                distance_metric: DistanceMetric::Euclidean,
            }, // Placeholder
            meta_learner: None,
        })
    }
}

impl FewShotPipeline<FewShotPipelineTrained> {
    /// Predict using the fitted few-shot learner
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        self.state.fitted_learner.predict(x)
    }

    /// Adapt to a new task using support set
    pub fn adapt_to_task(&mut self, support_set: &SupportSet) -> SklResult<()> {
        match &mut self.state.fitted_learner {
            MetaLearnerWrapper::Prototypical(_) => {
                // Prototypical networks adapt by recomputing prototypes
                // This would require refitting with new support set
                Ok(())
            }
            MetaLearnerWrapper::MAML(learner) => learner.adapt_to_task(support_set),
        }
    }

    /// Get the fitted learner
    #[must_use]
    pub fn learner(&self) -> &MetaLearnerWrapper {
        &self.state.fitted_learner
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_support_set() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let labels = array![0.0, 0.0, 1.0, 1.0];

        let support_set = SupportSet::new(features, labels, 2, 2);

        let (class_features, class_labels) = support_set.get_class_examples(0.0);
        assert_eq!(class_features.nrows(), 2);
        assert_eq!(class_labels.len(), 2);

        let classes = support_set.get_classes();
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn test_distance_metrics() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let euclidean = DistanceMetric::Euclidean;
        let distance = euclidean.distance(&a.view(), &b.view());
        assert!(distance > 0.0);

        let cosine = DistanceMetric::Cosine;
        let distance = cosine.distance(&a.view(), &b.view());
        assert!(distance >= 0.0 && distance <= 2.0);
    }

    #[test]
    fn test_prototypical_network() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let learner = PrototypicalNetwork::new();
        let fitted = learner.fit(&x.view(), &Some(&y.view())).unwrap();

        let predictions = fitted.predict(&x.view()).unwrap();
        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_few_shot_pipeline() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let pipeline = FewShotPipeline::prototypical(DistanceMetric::Euclidean);
        let fitted = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();

        let predictions = fitted.predict(&x.view()).unwrap();
        assert_eq!(predictions.len(), x.nrows());
    }
}
