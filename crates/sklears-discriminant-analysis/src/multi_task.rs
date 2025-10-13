//! # Multi-Task Discriminant Learning
//!
//! Multi-task discriminant learning extends traditional discriminant analysis to handle
//! multiple related classification tasks simultaneously. This approach leverages shared
//! information across tasks to improve performance, especially when individual tasks
//! have limited training data.
//!
//! ## Key Features
//! - Shared discriminant subspace across multiple tasks
//! - Task-specific discriminant components  
//! - Regularization to control sharing vs. task specialization
//! - Support for both LDA and QDA base classifiers
//! - Flexible task weighting strategies
//! - Transfer learning capabilities for new tasks
//!
//! ## Applications
//! - Multi-domain classification (e.g., sentiment analysis across different domains)
//! - Multi-label classification with structured label dependencies
//! - Few-shot learning with related tasks
//! - Domain adaptation scenarios

use crate::lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
use crate::qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig};
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained},
    types::Float,
};

/// Configuration for multi-task discriminant learning
#[derive(Debug, Clone)]
pub struct MultiTaskDiscriminantLearningConfig {
    /// Number of shared discriminant components
    pub n_shared_components: Option<usize>,
    /// Number of task-specific components per task
    pub n_task_components: Option<usize>,
    /// Regularization parameter for shared components (higher = more sharing)
    pub sharing_penalty: Float,
    /// Regularization parameter for task-specific components
    pub task_penalty: Float,
    /// Base discriminant type ("lda" or "qda")
    pub base_discriminant: String,
    /// Task weighting strategy ("uniform", "proportional", "inverse")
    pub task_weighting: String,
    /// Whether to normalize task weights
    pub normalize_weights: bool,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to use warm start for new tasks
    pub warm_start: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// LDA configuration for base classifiers
    pub lda_config: LinearDiscriminantAnalysisConfig,
    /// QDA configuration for base classifiers
    pub qda_config: QuadraticDiscriminantAnalysisConfig,
}

impl Default for MultiTaskDiscriminantLearningConfig {
    fn default() -> Self {
        Self {
            n_shared_components: None,
            n_task_components: None,
            sharing_penalty: 1.0,
            task_penalty: 1.0,
            base_discriminant: "lda".to_string(),
            task_weighting: "uniform".to_string(),
            normalize_weights: true,
            max_iter: 100,
            tol: 1e-6,
            warm_start: false,
            random_state: None,
            lda_config: LinearDiscriminantAnalysisConfig::default(),
            qda_config: QuadraticDiscriminantAnalysisConfig::default(),
        }
    }
}

/// Represents a single task in multi-task learning
#[derive(Debug, Clone)]
pub struct Task {
    /// Task identifier
    pub task_id: usize,
    /// Training data for this task
    pub x: Array2<Float>,
    /// Training labels for this task
    pub y: Array1<i32>,
    /// Task weight (importance)
    pub weight: Float,
    /// Task-specific class mapping
    pub classes: Vec<i32>,
}

impl Task {
    /// Create a new task
    pub fn new(task_id: usize, x: Array2<Float>, y: Array1<i32>) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        let classes: Vec<i32> = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        Ok(Self {
            task_id,
            x,
            y,
            weight: 1.0,
            classes,
        })
    }

    /// Set task weight
    pub fn with_weight(mut self, weight: Float) -> Self {
        self.weight = weight;
        self
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    /// Get number of classes
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

/// Multi-task discriminant learning estimator
#[derive(Debug, Clone)]
pub struct MultiTaskDiscriminantLearning {
    config: MultiTaskDiscriminantLearningConfig,
}

impl MultiTaskDiscriminantLearning {
    /// Create a new multi-task discriminant learning estimator
    pub fn new() -> Self {
        Self {
            config: MultiTaskDiscriminantLearningConfig::default(),
        }
    }

    /// Set number of shared discriminant components
    pub fn n_shared_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_shared_components = n_components;
        self
    }

    /// Set number of task-specific components
    pub fn n_task_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_task_components = n_components;
        self
    }

    /// Set sharing penalty (regularization for shared components)
    pub fn sharing_penalty(mut self, penalty: Float) -> Self {
        self.config.sharing_penalty = penalty;
        self
    }

    /// Set task penalty (regularization for task-specific components)
    pub fn task_penalty(mut self, penalty: Float) -> Self {
        self.config.task_penalty = penalty;
        self
    }

    /// Set base discriminant type
    pub fn base_discriminant(mut self, discriminant_type: &str) -> Self {
        self.config.base_discriminant = discriminant_type.to_string();
        self
    }

    /// Set task weighting strategy
    pub fn task_weighting(mut self, weighting: &str) -> Self {
        self.config.task_weighting = weighting.to_string();
        self
    }

    /// Set whether to normalize task weights
    pub fn normalize_weights(mut self, normalize: bool) -> Self {
        self.config.normalize_weights = normalize;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set warm start option
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Compute task weights based on the weighting strategy
    fn compute_task_weights(&self, tasks: &[Task]) -> Vec<Float> {
        let mut weights = match self.config.task_weighting.as_str() {
            "uniform" => vec![1.0; tasks.len()],
            "proportional" => tasks.iter().map(|t| t.n_samples() as Float).collect(),
            "inverse" => tasks
                .iter()
                .map(|t| 1.0 / (t.n_samples() as Float))
                .collect(),
            _ => vec![1.0; tasks.len()],
        };

        // Apply manual task weights
        for (i, task) in tasks.iter().enumerate() {
            weights[i] *= task.weight;
        }

        // Normalize if requested
        if self.config.normalize_weights {
            let sum: Float = weights.iter().sum();
            if sum > 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
        }

        weights
    }

    /// Compute shared discriminant subspace across all tasks
    fn compute_shared_subspace(&self, tasks: &[Task]) -> Result<Array2<Float>> {
        let n_features = tasks[0].n_features();
        let n_shared = self
            .config
            .n_shared_components
            .unwrap_or((n_features / 2).max(1));

        // Stack all data from all tasks
        let mut all_x = Vec::new();
        let mut all_y = Vec::new();
        let mut task_indices = Vec::new();

        for (task_idx, task) in tasks.iter().enumerate() {
            for (i, row) in task.x.axis_iter(Axis(0)).enumerate() {
                all_x.push(row.to_owned());
                all_y.push(task.y[i]);
                task_indices.push(task_idx);
            }
        }

        if all_x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No training data provided".to_string(),
            ));
        }

        // Convert to arrays
        let combined_x = Array2::from_shape_vec(
            (all_x.len(), n_features),
            all_x.into_iter().flatten().collect(),
        )
        .map_err(|_| SklearsError::InvalidInput("Failed to stack task data".to_string()))?;

        let combined_y = Array1::from_vec(all_y);

        // Fit a global discriminant analysis
        let shared_components = match self.config.base_discriminant.as_str() {
            "lda" => {
                let lda = LinearDiscriminantAnalysis::new().n_components(Some(n_shared));
                let fitted_lda = lda.fit(&combined_x, &combined_y)?;
                fitted_lda.components().clone()
            }
            "qda" => {
                // For QDA, we use the pooled covariance approach similar to LDA
                let lda = LinearDiscriminantAnalysis::new().n_components(Some(n_shared));
                let fitted_lda = lda.fit(&combined_x, &combined_y)?;
                fitted_lda.components().clone()
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "base_discriminant".to_string(),
                    reason: format!(
                        "Unknown base discriminant: {}",
                        self.config.base_discriminant
                    ),
                })
            }
        };

        Ok(shared_components)
    }

    /// Compute task-specific discriminant components
    fn compute_task_specific_components(
        &self,
        task: &Task,
        shared_components: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let n_features = task.n_features();
        let n_task = self
            .config
            .n_task_components
            .unwrap_or((n_features / 4).max(1));

        // Project data to the orthogonal space of shared components
        let task_x = &task.x;

        // Create orthogonal projection matrix (I - P_shared)
        let shared_proj = shared_components.t().dot(shared_components);
        let mut ortho_proj = Array2::eye(n_features);
        ortho_proj = ortho_proj - shared_proj;

        // Project task data to orthogonal space
        let projected_x = task_x.dot(&ortho_proj);

        // Fit discriminant analysis on projected data
        let task_components = match self.config.base_discriminant.as_str() {
            "lda" => {
                let mut lda_config = self.config.lda_config.clone();
                lda_config.n_components = Some(n_task);
                let lda = LinearDiscriminantAnalysis::new();
                let fitted_lda = lda.fit(&projected_x, &task.y)?;
                fitted_lda.components().clone()
            }
            "qda" => {
                // For QDA, use LDA approach for component extraction
                let mut lda_config = self.config.lda_config.clone();
                lda_config.n_components = Some(n_task);
                let lda = LinearDiscriminantAnalysis::new();
                let fitted_lda = lda.fit(&projected_x, &task.y)?;
                fitted_lda.components().clone()
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "base_discriminant".to_string(),
                    reason: format!(
                        "Unknown base discriminant: {}",
                        self.config.base_discriminant
                    ),
                })
            }
        };

        // Transform components back to original space
        let final_components = task_components.dot(&ortho_proj);

        Ok(final_components)
    }
}

impl Default for MultiTaskDiscriminantLearning {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTaskDiscriminantLearning {
    type Config = MultiTaskDiscriminantLearningConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Trained multi-task discriminant learning model
#[derive(Debug)]
pub struct TrainedMultiTaskDiscriminantLearning {
    /// Shared discriminant components
    shared_components: Array2<Float>,
    /// Task-specific components for each task
    task_components: Vec<Array2<Float>>,
    /// Task-specific classifiers
    task_classifiers: Vec<TaskClassifier>,
    /// Task information
    tasks: Vec<Task>,
    /// Task weights
    task_weights: Vec<Float>,
    /// Global classes (union of all task classes)
    global_classes: Vec<i32>,
    /// Configuration
    config: MultiTaskDiscriminantLearningConfig,
}

/// Task-specific classifier
#[derive(Debug)]
pub enum TaskClassifier {
    /// LDA classifier for this task
    LDA(LinearDiscriminantAnalysis<Trained>),
    /// QDA classifier for this task  
    QDA(QuadraticDiscriminantAnalysis<Trained>),
}

impl TaskClassifier {
    /// Predict using the task classifier
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        match self {
            TaskClassifier::LDA(lda) => lda.predict(x),
            TaskClassifier::QDA(qda) => qda.predict(x),
        }
    }

    /// Predict probabilities using the task classifier
    pub fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        match self {
            TaskClassifier::LDA(lda) => lda.predict_proba(x),
            TaskClassifier::QDA(qda) => qda.predict_proba(x),
        }
    }

    /// Get classes
    pub fn classes(&self) -> &[i32] {
        match self {
            TaskClassifier::LDA(lda) => lda.classes().as_slice().unwrap(),
            TaskClassifier::QDA(qda) => qda.classes().as_slice().unwrap(),
        }
    }
}

impl TrainedMultiTaskDiscriminantLearning {
    /// Get the shared components
    pub fn shared_components(&self) -> &Array2<Float> {
        &self.shared_components
    }

    /// Get task-specific components for a task
    pub fn task_components(&self, task_id: usize) -> Option<&Array2<Float>> {
        self.task_components.get(task_id)
    }

    /// Get global classes
    pub fn global_classes(&self) -> &[i32] {
        &self.global_classes
    }

    /// Get task information
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// Get task weights
    pub fn task_weights(&self) -> &[Float] {
        &self.task_weights
    }

    /// Predict for a specific task
    pub fn predict_task(&self, x: &Array2<Float>, task_id: usize) -> Result<Array1<i32>> {
        if task_id >= self.task_classifiers.len() {
            return Err(SklearsError::InvalidParameter {
                name: "task_id".to_string(),
                reason: format!("Task {} not found", task_id),
            });
        }

        // Transform data using both shared and task-specific components
        let transformed_x = self.transform_task(x, task_id)?;

        // Predict using task classifier
        self.task_classifiers[task_id].predict(&transformed_x)
    }

    /// Predict probabilities for a specific task
    pub fn predict_proba_task(&self, x: &Array2<Float>, task_id: usize) -> Result<Array2<Float>> {
        if task_id >= self.task_classifiers.len() {
            return Err(SklearsError::InvalidParameter {
                name: "task_id".to_string(),
                reason: format!("Task {} not found", task_id),
            });
        }

        // Transform data using both shared and task-specific components
        let transformed_x = self.transform_task(x, task_id)?;

        // Predict probabilities using task classifier
        self.task_classifiers[task_id].predict_proba(&transformed_x)
    }

    /// Transform data for a specific task using shared and task-specific components
    pub fn transform_task(&self, x: &Array2<Float>, task_id: usize) -> Result<Array2<Float>> {
        if task_id >= self.task_components.len() {
            return Err(SklearsError::InvalidParameter {
                name: "task_id".to_string(),
                reason: format!("Task {} not found", task_id),
            });
        }

        // Project to shared subspace
        let shared_proj = x.dot(&self.shared_components.t());

        // Project to task-specific subspace
        let task_proj = x.dot(&self.task_components[task_id].t());

        // Concatenate projections
        let mut combined = Array2::zeros((x.nrows(), shared_proj.ncols() + task_proj.ncols()));
        combined
            .slice_mut(s![.., ..shared_proj.ncols()])
            .assign(&shared_proj);
        combined
            .slice_mut(s![.., shared_proj.ncols()..])
            .assign(&task_proj);

        Ok(combined)
    }

    /// Transform data using only shared components
    pub fn transform_shared(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.dot(&self.shared_components.t()))
    }

    /// Add a new task (transfer learning)
    pub fn add_task(&mut self, task: Task) -> Result<usize> {
        let task_id = self.tasks.len();

        // Compute task-specific components for new task
        let task_components = self.compute_task_components(&task)?;

        // Train task classifier
        let task_classifier = self.train_task_classifier(&task, &task_components)?;

        // Add to model
        self.tasks.push(task);
        self.task_components.push(task_components);
        self.task_classifiers.push(task_classifier);
        self.task_weights.push(1.0);

        // Update global classes
        self.update_global_classes();

        Ok(task_id)
    }

    fn compute_task_components(&self, task: &Task) -> Result<Array2<Float>> {
        let n_features = task.n_features();
        let n_task = self
            .config
            .n_task_components
            .unwrap_or((n_features / 4).max(1));

        // Create orthogonal projection matrix (I - P_shared)
        let shared_proj = self.shared_components.t().dot(&self.shared_components);
        let mut ortho_proj = Array2::eye(n_features);
        ortho_proj = ortho_proj - shared_proj;

        // Project task data to orthogonal space
        let projected_x = task.x.dot(&ortho_proj);

        // Fit discriminant analysis on projected data
        let task_components = match self.config.base_discriminant.as_str() {
            "lda" => {
                let mut lda_config = self.config.lda_config.clone();
                lda_config.n_components = Some(n_task);
                let lda = LinearDiscriminantAnalysis::new();
                let fitted_lda = lda.fit(&projected_x, &task.y)?;
                fitted_lda.components().clone()
            }
            "qda" => {
                let mut lda_config = self.config.lda_config.clone();
                lda_config.n_components = Some(n_task);
                let lda = LinearDiscriminantAnalysis::new();
                let fitted_lda = lda.fit(&projected_x, &task.y)?;
                fitted_lda.components().clone()
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "base_discriminant".to_string(),
                    reason: format!(
                        "Unknown base discriminant: {}",
                        self.config.base_discriminant
                    ),
                })
            }
        };

        // Transform components back to original space
        Ok(task_components.dot(&ortho_proj))
    }

    fn train_task_classifier(
        &self,
        task: &Task,
        task_components: &Array2<Float>,
    ) -> Result<TaskClassifier> {
        // Transform task data
        let shared_proj = task.x.dot(&self.shared_components.t());
        let task_proj = task.x.dot(&task_components.t());

        let mut combined = Array2::zeros((task.x.nrows(), shared_proj.ncols() + task_proj.ncols()));
        combined
            .slice_mut(s![.., ..shared_proj.ncols()])
            .assign(&shared_proj);
        combined
            .slice_mut(s![.., shared_proj.ncols()..])
            .assign(&task_proj);

        match self.config.base_discriminant.as_str() {
            "lda" => {
                let lda = LinearDiscriminantAnalysis::new();
                let fitted_lda = lda.fit(&combined, &task.y)?;
                Ok(TaskClassifier::LDA(fitted_lda))
            }
            "qda" => {
                let qda = QuadraticDiscriminantAnalysis::new();
                let fitted_qda = qda.fit(&combined, &task.y)?;
                Ok(TaskClassifier::QDA(fitted_qda))
            }
            _ => Err(SklearsError::InvalidParameter {
                name: "base_discriminant".to_string(),
                reason: format!(
                    "Unknown base discriminant: {}",
                    self.config.base_discriminant
                ),
            }),
        }
    }

    fn update_global_classes(&mut self) {
        let mut all_classes = Vec::new();
        for task in &self.tasks {
            all_classes.extend(&task.classes);
        }
        all_classes.sort_unstable();
        all_classes.dedup();
        self.global_classes = all_classes;
    }
}

impl Fit<Vec<Task>, ()> for MultiTaskDiscriminantLearning {
    type Fitted = TrainedMultiTaskDiscriminantLearning;

    fn fit(self, tasks: &Vec<Task>, _y: &()) -> Result<Self::Fitted> {
        if tasks.is_empty() {
            return Err(SklearsError::InvalidInput("No tasks provided".to_string()));
        }

        // Validate that all tasks have the same number of features
        let n_features = tasks[0].n_features();
        for task in tasks {
            if task.n_features() != n_features {
                return Err(SklearsError::InvalidInput(
                    "All tasks must have the same number of features".to_string(),
                ));
            }
        }

        // Compute task weights
        let task_weights = self.compute_task_weights(tasks);

        // Compute shared discriminant subspace
        let shared_components = self.compute_shared_subspace(tasks)?;

        // Compute task-specific components and train classifiers
        let mut task_components = Vec::new();
        let mut task_classifiers = Vec::new();

        for task in tasks {
            let task_comp = self.compute_task_specific_components(task, &shared_components)?;

            // Transform task data using both shared and task-specific components
            let shared_proj = task.x.dot(&shared_components.t());
            let task_proj = task.x.dot(&task_comp.t());

            let mut combined =
                Array2::zeros((task.x.nrows(), shared_proj.ncols() + task_proj.ncols()));
            combined
                .slice_mut(s![.., ..shared_proj.ncols()])
                .assign(&shared_proj);
            combined
                .slice_mut(s![.., shared_proj.ncols()..])
                .assign(&task_proj);

            // Train task classifier
            let classifier = match self.config.base_discriminant.as_str() {
                "lda" => {
                    let lda = LinearDiscriminantAnalysis::new();
                    let fitted_lda = lda.fit(&combined, &task.y)?;
                    TaskClassifier::LDA(fitted_lda)
                }
                "qda" => {
                    let qda = QuadraticDiscriminantAnalysis::new();
                    let fitted_qda = qda.fit(&combined, &task.y)?;
                    TaskClassifier::QDA(fitted_qda)
                }
                _ => {
                    return Err(SklearsError::InvalidParameter {
                        name: "base_discriminant".to_string(),
                        reason: format!(
                            "Unknown base discriminant: {}",
                            self.config.base_discriminant
                        ),
                    })
                }
            };

            task_components.push(task_comp);
            task_classifiers.push(classifier);
        }

        // Compute global classes
        let mut global_classes = Vec::new();
        for task in tasks {
            global_classes.extend(&task.classes);
        }
        global_classes.sort_unstable();
        global_classes.dedup();

        Ok(TrainedMultiTaskDiscriminantLearning {
            shared_components,
            task_components,
            task_classifiers,
            tasks: tasks.clone(),
            task_weights,
            global_classes,
            config: self.config.clone(),
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_task_creation() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let task = Task::new(0, x.clone(), y.clone()).unwrap();

        assert_eq!(task.task_id, 0);
        assert_eq!(task.n_samples(), 4);
        assert_eq!(task.n_features(), 2);
        assert_eq!(task.n_classes(), 2);
        assert_eq!(task.classes, vec![0, 1]);
    }

    #[test]
    fn test_multi_task_discriminant_learning_basic() {
        // Create two related tasks
        let task1_x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2] // Class 1
        ];
        let task1_y = array![0, 0, 0, 1, 1, 1];

        let task2_x = array![
            [1.5, 2.5],
            [1.6, 2.6],
            [1.7, 2.7], // Class 0
            [3.5, 4.5],
            [3.6, 4.6],
            [3.7, 4.7] // Class 1
        ];
        let task2_y = array![0, 0, 0, 1, 1, 1];

        let task1 = Task::new(0, task1_x.clone(), task1_y.clone()).unwrap();
        let task2 = Task::new(1, task2_x.clone(), task2_y.clone()).unwrap();
        let tasks = vec![task1, task2];

        let mtdl = MultiTaskDiscriminantLearning::new();
        let fitted = mtdl.fit(&tasks, &()).unwrap();

        // Test prediction for task 1
        let predictions = fitted.predict_task(&task1_x, 0).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test prediction for task 2
        let predictions = fitted.predict_task(&task2_x, 1).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_multi_task_predict_proba() {
        let task1_x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let task1_y = array![0, 0, 1, 1];

        let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let task2_y = array![0, 0, 1, 1];

        let task1 = Task::new(0, task1_x.clone(), task1_y.clone()).unwrap();
        let task2 = Task::new(1, task2_x.clone(), task2_y.clone()).unwrap();
        let tasks = vec![task1, task2];

        let mtdl = MultiTaskDiscriminantLearning::new();
        let fitted = mtdl.fit(&tasks, &()).unwrap();

        let probas = fitted.predict_proba_task(&task1_x, 0).unwrap();
        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_multi_task_transform() {
        let task1_x = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.5],
            [3.0, 4.0, 2.0],
            [4.0, 3.0, 3.5],
            [5.0, 2.0, 4.0],
            [6.0, 1.0, 4.5]
        ];
        let task1_y = array![0, 0, 1, 1, 2, 2];

        let task2_x = array![
            [1.5, 2.5, 0.8],
            [2.5, 1.5, 1.8],
            [3.5, 4.5, 2.5],
            [4.5, 3.5, 3.8],
            [5.5, 2.5, 4.3],
            [6.5, 1.5, 4.8]
        ];
        let task2_y = array![0, 0, 1, 1, 2, 2];

        let task1 = Task::new(0, task1_x.clone(), task1_y.clone()).unwrap();
        let task2 = Task::new(1, task2_x.clone(), task2_y.clone()).unwrap();
        let tasks = vec![task1, task2];

        let mtdl = MultiTaskDiscriminantLearning::new()
            .n_shared_components(Some(2))
            .n_task_components(Some(1));
        let fitted = mtdl.fit(&tasks, &()).unwrap();

        // Test shared transformation
        let shared_transformed = fitted.transform_shared(&task1_x).unwrap();
        assert!(shared_transformed.ncols() >= 1); // Ensure we get some components

        // Test task-specific transformation
        let task_transformed = fitted.transform_task(&task1_x, 0).unwrap();
        assert!(task_transformed.ncols() >= shared_transformed.ncols()); // Should include shared + task components
    }

    #[test]
    fn test_multi_task_with_qda() {
        let task1_x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let task1_y = array![0, 0, 1, 1];

        let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let task2_y = array![0, 0, 1, 1];

        let task1 = Task::new(0, task1_x.clone(), task1_y.clone()).unwrap();
        let task2 = Task::new(1, task2_x.clone(), task2_y.clone()).unwrap();
        let tasks = vec![task1, task2];

        let mtdl = MultiTaskDiscriminantLearning::new().base_discriminant("qda");
        let fitted = mtdl.fit(&tasks, &()).unwrap();

        let predictions = fitted.predict_task(&task1_x, 0).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_task_weighting_strategies() {
        let task1_x = array![[1.0, 2.0], [2.0, 3.0]];
        let task1_y = array![0, 1];

        let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let task2_y = array![0, 0, 1, 1];

        let task1 = Task::new(0, task1_x.clone(), task1_y.clone()).unwrap();
        let task2 = Task::new(1, task2_x.clone(), task2_y.clone()).unwrap();
        let tasks = vec![task1, task2];

        let strategies = ["uniform", "proportional", "inverse"];
        for strategy in &strategies {
            let mtdl = MultiTaskDiscriminantLearning::new().task_weighting(strategy);
            let fitted = mtdl.fit(&tasks, &()).unwrap();

            assert_eq!(fitted.task_weights().len(), 2);
            assert!(fitted.task_weights().iter().all(|&w| w > 0.0));
        }
    }

    #[test]
    fn test_add_new_task() {
        let task1_x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let task1_y = array![0, 0, 1, 1];

        let task2_x = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let task2_y = array![0, 0, 1, 1];

        let task1 = Task::new(0, task1_x.clone(), task1_y.clone()).unwrap();
        let task2 = Task::new(1, task2_x.clone(), task2_y.clone()).unwrap();
        let tasks = vec![task1, task2];

        let mtdl = MultiTaskDiscriminantLearning::new();
        let mut fitted = mtdl.fit(&tasks, &()).unwrap();

        // Add a new task
        let task3_x = array![[2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let task3_y = array![0, 0, 1, 1];
        let task3 = Task::new(2, task3_x.clone(), task3_y.clone()).unwrap();

        let new_task_id = fitted.add_task(task3).unwrap();
        assert_eq!(new_task_id, 2);

        // Test prediction for new task
        let predictions = fitted.predict_task(&task3_x, new_task_id).unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
