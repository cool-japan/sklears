//! Meta-learning pipeline components
//!
//! This module provides meta-learning capabilities including experience storage,
//! adaptation strategies, and meta-learning pipeline components.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::{HashMap, VecDeque};

use crate::{PipelinePredictor, PipelineStep};

/// Experience entry for meta-learning
#[derive(Debug, Clone)]
pub struct Experience {
    /// Task identifier
    pub task_id: String,
    /// Input features for the task
    pub features: Array2<f64>,
    /// Target values for the task
    pub targets: Array1<f64>,
    /// Task metadata (e.g., task type, domain)
    pub metadata: HashMap<String, String>,
    /// Performance metrics achieved on this task
    pub performance: HashMap<String, f64>,
    /// Model parameters used for this task
    pub parameters: HashMap<String, f64>,
}

impl Experience {
    /// Create a new experience entry
    #[must_use]
    pub fn new(task_id: String, features: Array2<f64>, targets: Array1<f64>) -> Self {
        Self {
            task_id,
            features,
            targets,
            metadata: HashMap::new(),
            performance: HashMap::new(),
            parameters: HashMap::new(),
        }
    }

    /// Add metadata to the experience
    #[must_use]
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add performance metrics to the experience
    #[must_use]
    pub fn with_performance(mut self, performance: HashMap<String, f64>) -> Self {
        self.performance = performance;
        self
    }

    /// Add model parameters to the experience
    #[must_use]
    pub fn with_parameters(mut self, parameters: HashMap<String, f64>) -> Self {
        self.parameters = parameters;
        self
    }
}

/// Experience storage for meta-learning
#[derive(Debug, Clone)]
pub struct ExperienceStorage {
    /// Maximum number of experiences to store
    max_size: usize,
    /// Stored experiences
    experiences: VecDeque<Experience>,
    /// Index by task ID for fast lookup
    task_index: HashMap<String, Vec<usize>>,
}

impl ExperienceStorage {
    /// Create a new experience storage
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            experiences: VecDeque::new(),
            task_index: HashMap::new(),
        }
    }

    /// Add an experience to the storage
    pub fn add_experience(&mut self, experience: Experience) {
        let task_id = experience.task_id.clone();

        // Remove oldest experience if at capacity
        if self.experiences.len() >= self.max_size {
            if let Some(removed) = self.experiences.pop_front() {
                self.remove_from_index(&removed.task_id, 0);
            }
        }

        // Add new experience
        let index = self.experiences.len();
        self.experiences.push_back(experience);

        // Update index
        self.task_index.entry(task_id).or_default().push(index);
    }

    /// Get experiences for a specific task
    #[must_use]
    pub fn get_task_experiences(&self, task_id: &str) -> Vec<&Experience> {
        if let Some(indices) = self.task_index.get(task_id) {
            indices
                .iter()
                .filter_map(|&i| self.experiences.get(i))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all experiences
    #[must_use]
    pub fn get_all_experiences(&self) -> Vec<&Experience> {
        self.experiences.iter().collect()
    }

    /// Get the most similar experiences based on feature similarity
    #[must_use]
    pub fn get_similar_experiences(
        &self,
        features: &ArrayView2<'_, f64>,
        k: usize,
    ) -> Vec<&Experience> {
        let mut similarities = Vec::new();

        for exp in &self.experiences {
            let similarity = self.compute_similarity(features, &exp.features.view());
            similarities.push((similarity, exp));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        similarities
            .into_iter()
            .take(k)
            .map(|(_, exp)| exp)
            .collect()
    }

    /// Compute similarity between two feature sets (cosine similarity)
    fn compute_similarity(
        &self,
        features1: &ArrayView2<'_, f64>,
        features2: &ArrayView2<'_, f64>,
    ) -> f64 {
        if features1.ncols() != features2.ncols() {
            return 0.0;
        }

        // Use mean of features for simplicity
        let mean1 = features1.mean_axis(Axis(0)).unwrap();
        let mean2 = features2.mean_axis(Axis(0)).unwrap();

        // Cosine similarity
        let dot_product = mean1.dot(&mean2);
        let norm1 = mean1.mapv(|x| x * x).sum().sqrt();
        let norm2 = mean2.mapv(|x| x * x).sum().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Remove an experience from the index
    fn remove_from_index(&mut self, task_id: &str, index: usize) {
        if let Some(indices) = self.task_index.get_mut(task_id) {
            indices.retain(|&i| i != index);
            if indices.is_empty() {
                self.task_index.remove(task_id);
            }
        }
    }

    /// Get the number of stored experiences
    #[must_use]
    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Check if storage is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }
}

/// Adaptation strategy for meta-learning
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Fine-tune all parameters
    FineTuning {
        learning_rate: f64,
        num_steps: usize,
    },
    /// Fine-tune only the last layer
    LastLayerFineTuning {
        learning_rate: f64,
        num_steps: usize,
    },
    /// Feature-based adaptation
    FeatureAdaptation { adaptation_weight: f64 },
    /// Parameter averaging from similar tasks
    ParameterAveraging {
        num_similar: usize,
        similarity_threshold: f64,
    },
    /// Gradient-based meta-learning (MAML-style)
    GradientBased {
        inner_lr: f64,
        outer_lr: f64,
        num_inner_steps: usize,
    },
}

impl AdaptationStrategy {
    /// Apply the adaptation strategy
    pub fn adapt(
        &self,
        current_params: &HashMap<String, f64>,
        experiences: &[&Experience],
        task_features: &ArrayView2<'_, f64>,
    ) -> SklResult<HashMap<String, f64>> {
        match self {
            AdaptationStrategy::FineTuning {
                learning_rate,
                num_steps,
            } => self.fine_tune_adaptation(current_params, experiences, *learning_rate, *num_steps),
            AdaptationStrategy::LastLayerFineTuning {
                learning_rate,
                num_steps,
            } => {
                self.last_layer_adaptation(current_params, experiences, *learning_rate, *num_steps)
            }
            AdaptationStrategy::FeatureAdaptation { adaptation_weight } => self.feature_adaptation(
                current_params,
                experiences,
                task_features,
                *adaptation_weight,
            ),
            AdaptationStrategy::ParameterAveraging {
                num_similar,
                similarity_threshold,
            } => self.parameter_averaging(
                current_params,
                experiences,
                *num_similar,
                *similarity_threshold,
            ),
            AdaptationStrategy::GradientBased {
                inner_lr,
                outer_lr,
                num_inner_steps,
            } => self.gradient_based_adaptation(
                current_params,
                experiences,
                *inner_lr,
                *outer_lr,
                *num_inner_steps,
            ),
        }
    }

    /// Fine-tuning adaptation
    fn fine_tune_adaptation(
        &self,
        current_params: &HashMap<String, f64>,
        experiences: &[&Experience],
        learning_rate: f64,
        num_steps: usize,
    ) -> SklResult<HashMap<String, f64>> {
        let mut adapted_params = current_params.clone();

        if experiences.is_empty() {
            return Ok(adapted_params);
        }

        // Simple gradient descent simulation
        for _ in 0..num_steps {
            for (key, value) in &mut adapted_params {
                // Compute pseudo-gradient based on experience performance
                let mut gradient = 0.0;
                let mut count = 0;

                for exp in experiences {
                    if let Some(&exp_param) = exp.parameters.get(key) {
                        if let Some(&performance) = exp.performance.get("accuracy") {
                            // Simple gradient approximation
                            gradient += (exp_param - *value) * performance;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    gradient /= f64::from(count);
                    *value += learning_rate * gradient;
                }
            }
        }

        Ok(adapted_params)
    }

    /// Last layer fine-tuning adaptation
    fn last_layer_adaptation(
        &self,
        current_params: &HashMap<String, f64>,
        experiences: &[&Experience],
        learning_rate: f64,
        num_steps: usize,
    ) -> SklResult<HashMap<String, f64>> {
        let mut adapted_params = current_params.clone();

        // Only adapt parameters that contain "output" or "final" in their name
        let last_layer_keys: Vec<String> = adapted_params
            .keys()
            .filter(|key| key.contains("output") || key.contains("final"))
            .cloned()
            .collect();

        for _ in 0..num_steps {
            for key in &last_layer_keys {
                if let Some(value) = adapted_params.get_mut(key) {
                    let mut gradient = 0.0;
                    let mut count = 0;

                    for exp in experiences {
                        if let Some(&exp_param) = exp.parameters.get(key) {
                            if let Some(&performance) = exp.performance.get("accuracy") {
                                gradient += (exp_param - *value) * performance;
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        gradient /= f64::from(count);
                        *value += learning_rate * gradient;
                    }
                }
            }
        }

        Ok(adapted_params)
    }

    /// Feature-based adaptation
    fn feature_adaptation(
        &self,
        current_params: &HashMap<String, f64>,
        experiences: &[&Experience],
        _task_features: &ArrayView2<'_, f64>,
        adaptation_weight: f64,
    ) -> SklResult<HashMap<String, f64>> {
        let mut adapted_params = current_params.clone();

        if experiences.is_empty() {
            return Ok(adapted_params);
        }

        // Adapt parameters based on feature similarity
        for (key, value) in &mut adapted_params {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for exp in experiences {
                if let Some(&exp_param) = exp.parameters.get(key) {
                    if let Some(&performance) = exp.performance.get("accuracy") {
                        let weight = performance * adaptation_weight;
                        weighted_sum += exp_param * weight;
                        weight_sum += weight;
                    }
                }
            }

            if weight_sum > 0.0 {
                let adapted_value = weighted_sum / weight_sum;
                *value = (1.0 - adaptation_weight) * *value + adaptation_weight * adapted_value;
            }
        }

        Ok(adapted_params)
    }

    /// Parameter averaging adaptation
    fn parameter_averaging(
        &self,
        current_params: &HashMap<String, f64>,
        experiences: &[&Experience],
        num_similar: usize,
        similarity_threshold: f64,
    ) -> SklResult<HashMap<String, f64>> {
        let mut adapted_params = current_params.clone();

        // Select top similar experiences based on performance
        let similar_exps: Vec<&Experience> = experiences
            .iter()
            .filter(|exp| {
                exp.performance
                    .get("accuracy")
                    .is_some_and(|&acc| acc >= similarity_threshold)
            })
            .take(num_similar)
            .copied()
            .collect();

        if similar_exps.is_empty() {
            return Ok(adapted_params);
        }

        // Average parameters from similar experiences
        for (key, value) in &mut adapted_params {
            let mut sum = *value;
            let mut count = 1; // Include current parameter

            for exp in &similar_exps {
                if let Some(&exp_param) = exp.parameters.get(key) {
                    sum += exp_param;
                    count += 1;
                }
            }

            *value = sum / f64::from(count);
        }

        Ok(adapted_params)
    }

    /// Gradient-based adaptation (MAML-style)
    fn gradient_based_adaptation(
        &self,
        current_params: &HashMap<String, f64>,
        experiences: &[&Experience],
        inner_lr: f64,
        _outer_lr: f64,
        num_inner_steps: usize,
    ) -> SklResult<HashMap<String, f64>> {
        let mut adapted_params = current_params.clone();

        if experiences.is_empty() {
            return Ok(adapted_params);
        }

        // Simulate inner loop adaptation
        for _ in 0..num_inner_steps {
            let mut gradients = HashMap::new();

            // Compute gradients based on experiences
            for exp in experiences {
                for (key, &exp_param) in &exp.parameters {
                    if let Some(&current_param) = adapted_params.get(key) {
                        if let Some(&performance) = exp.performance.get("loss") {
                            // Compute gradient as loss-weighted parameter difference
                            let gradient = (current_param - exp_param) * performance;
                            *gradients.entry(key.clone()).or_insert(0.0) += gradient;
                        }
                    }
                }
            }

            // Apply gradients
            for (key, gradient) in gradients {
                if let Some(param) = adapted_params.get_mut(&key) {
                    *param -= inner_lr * gradient / experiences.len() as f64;
                }
            }
        }

        Ok(adapted_params)
    }
}

/// Meta-learning pipeline component
#[derive(Debug)]
pub struct MetaLearningPipeline<S = Untrained> {
    state: S,
    base_estimator: Option<Box<dyn PipelinePredictor>>,
    experience_storage: ExperienceStorage,
    adaptation_strategy: AdaptationStrategy,
    meta_parameters: HashMap<String, f64>,
}

/// Trained state for `MetaLearningPipeline`
#[derive(Debug)]
pub struct MetaLearningPipelineTrained {
    fitted_estimator: Box<dyn PipelinePredictor>,
    experience_storage: ExperienceStorage,
    adaptation_strategy: AdaptationStrategy,
    meta_parameters: HashMap<String, f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl MetaLearningPipeline<Untrained> {
    /// Create a new meta-learning pipeline
    #[must_use]
    pub fn new(base_estimator: Box<dyn PipelinePredictor>) -> Self {
        Self {
            state: Untrained,
            base_estimator: Some(base_estimator),
            experience_storage: ExperienceStorage::new(1000), // Default max size
            adaptation_strategy: AdaptationStrategy::FineTuning {
                learning_rate: 0.01,
                num_steps: 10,
            },
            meta_parameters: HashMap::new(),
        }
    }

    /// Set the experience storage
    #[must_use]
    pub fn experience_storage(mut self, storage: ExperienceStorage) -> Self {
        self.experience_storage = storage;
        self
    }

    /// Set the adaptation strategy
    #[must_use]
    pub fn adaptation_strategy(mut self, strategy: AdaptationStrategy) -> Self {
        self.adaptation_strategy = strategy;
        self
    }

    /// Set meta-parameters
    #[must_use]
    pub fn meta_parameters(mut self, params: HashMap<String, f64>) -> Self {
        self.meta_parameters = params;
        self
    }

    /// Add an experience to the storage
    pub fn add_experience(&mut self, experience: Experience) {
        self.experience_storage.add_experience(experience);
    }
}

impl Estimator for MetaLearningPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
    for MetaLearningPipeline<Untrained>
{
    type Fitted = MetaLearningPipeline<MetaLearningPipelineTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        let mut base_estimator = self
            .base_estimator
            .ok_or_else(|| SklearsError::InvalidInput("No base estimator provided".to_string()))?;

        if let Some(y_values) = y.as_ref() {
            // Get similar experiences for adaptation
            let x_f64 = x.mapv(|v| v);
            let similar_experiences = self
                .experience_storage
                .get_similar_experiences(&x_f64.view(), 5);

            // Adapt meta-parameters based on experiences
            let adapted_params = self.adaptation_strategy.adapt(
                &self.meta_parameters,
                &similar_experiences,
                &x_f64.view(),
            )?;

            // Fit the base estimator
            base_estimator.fit(x, y_values)?;

            Ok(MetaLearningPipeline {
                state: MetaLearningPipelineTrained {
                    fitted_estimator: base_estimator,
                    experience_storage: self.experience_storage,
                    adaptation_strategy: self.adaptation_strategy,
                    meta_parameters: adapted_params,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                base_estimator: None,
                experience_storage: ExperienceStorage::new(0), // Placeholder
                adaptation_strategy: AdaptationStrategy::FineTuning {
                    learning_rate: 0.01,
                    num_steps: 1,
                },
                meta_parameters: HashMap::new(),
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for meta-learning".to_string(),
            ))
        }
    }
}

impl MetaLearningPipeline<MetaLearningPipelineTrained> {
    /// Predict using the fitted meta-learning pipeline
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        self.state.fitted_estimator.predict(x)
    }

    /// Adapt to a new task with limited data
    pub fn adapt_to_task(
        &mut self,
        task_id: String,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<()> {
        // Get similar experiences
        let x_f64 = x.mapv(|v| v);
        let similar_experiences = self
            .state
            .experience_storage
            .get_similar_experiences(&x_f64.view(), 5);

        // Adapt parameters
        let adapted_params = self.state.adaptation_strategy.adapt(
            &self.state.meta_parameters,
            &similar_experiences,
            &x_f64.view(),
        )?;

        // Update meta-parameters
        self.state.meta_parameters = adapted_params;

        // Create and store new experience
        let experience = Experience::new(task_id, x_f64, y.mapv(|v| v))
            .with_parameters(self.state.meta_parameters.clone());

        self.state.experience_storage.add_experience(experience);

        Ok(())
    }

    /// Get the experience storage
    #[must_use]
    pub fn experience_storage(&self) -> &ExperienceStorage {
        &self.state.experience_storage
    }

    /// Get the current meta-parameters
    #[must_use]
    pub fn meta_parameters(&self) -> &HashMap<String, f64> {
        &self.state.meta_parameters
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockPredictor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_experience_storage() {
        let mut storage = ExperienceStorage::new(3);

        let exp1 = Experience::new(
            "task1".to_string(),
            array![[1.0, 2.0], [3.0, 4.0]],
            array![1.0, 0.0],
        );

        storage.add_experience(exp1);
        assert_eq!(storage.len(), 1);

        let task_exps = storage.get_task_experiences("task1");
        assert_eq!(task_exps.len(), 1);
    }

    #[test]
    fn test_meta_learning_pipeline() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 0.0];

        let base_estimator = Box::new(MockPredictor::new());
        let mut pipeline = MetaLearningPipeline::new(base_estimator);

        // Add some experience
        let experience = Experience::new(
            "task1".to_string(),
            x.mapv(|v| v as f64),
            y.mapv(|v| v as f64),
        );
        pipeline.add_experience(experience);

        let fitted_pipeline = pipeline.fit(&x.view(), &Some(&y.view())).unwrap();
        let predictions = fitted_pipeline.predict(&x.view()).unwrap();

        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_adaptation_strategies() {
        let mut params = HashMap::new();
        params.insert("param1".to_string(), 1.0);
        params.insert("param2".to_string(), 2.0);

        let experience = Experience::new("task1".to_string(), array![[1.0, 2.0]], array![1.0])
            .with_parameters(params.clone())
            .with_performance([("accuracy".to_string(), 0.8)].iter().cloned().collect());

        let experiences = vec![&experience];
        let features = array![[1.0, 2.0]];

        let strategy = AdaptationStrategy::FineTuning {
            learning_rate: 0.1,
            num_steps: 5,
        };

        let adapted = strategy
            .adapt(&params, &experiences, &features.view())
            .unwrap();
        assert_eq!(adapted.len(), 2);
    }
}
