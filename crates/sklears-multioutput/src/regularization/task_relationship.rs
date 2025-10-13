//! Task Relationship Learning for Multi-Task Learning
//!
//! This method learns explicit relationships between tasks and uses this information
//! to regularize the learning process. Tasks that are determined to be related
//! are encouraged to have similar parameters.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::RandNormal;
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Methods for computing task similarity
#[derive(Debug, Clone, PartialEq)]
pub enum TaskSimilarityMethod {
    /// Correlation-based similarity
    Correlation,
    /// Cosine similarity of task parameters
    Cosine,
    /// Euclidean distance-based similarity
    Euclidean,
    /// Mutual information-based similarity
    MutualInformation,
}

/// Task Relationship Learning for Multi-Task Learning
///
/// This method learns explicit relationships between tasks and uses this information
/// to regularize the learning process. Tasks that are determined to be related
/// are encouraged to have similar parameters.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::regularization::TaskRelationshipLearning;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let mut y_tasks = HashMap::new();
/// y_tasks.insert("task1".to_string(), array![[1.0], [2.0], [1.5], [2.5]]);
/// y_tasks.insert("task2".to_string(), array![[0.5], [1.0], [0.8], [1.2]]);
///
/// let task_relationship = TaskRelationshipLearning::new()
///     .relationship_strength(0.1)
///     .similarity_threshold(0.5)
///     .max_iter(1000);
/// ```
#[derive(Debug, Clone)]
pub struct TaskRelationshipLearning<S = Untrained> {
    pub(crate) state: S,
    /// Strength of relationship regularization
    pub(crate) relationship_strength: Float,
    /// Threshold for task similarity to be considered related
    pub(crate) similarity_threshold: Float,
    /// Base regularization strength
    pub(crate) base_alpha: Float,
    /// Maximum iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance
    pub(crate) tolerance: Float,
    /// Learning rate
    pub(crate) learning_rate: Float,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Include intercept term
    pub(crate) fit_intercept: bool,
    /// Method for computing task similarity
    pub(crate) similarity_method: TaskSimilarityMethod,
}

/// Trained state for TaskRelationshipLearning
#[derive(Debug, Clone)]
pub struct TaskRelationshipLearningTrained {
    /// Coefficients for each task
    pub(crate) coefficients: HashMap<String, Array2<Float>>,
    /// Intercepts for each task
    pub(crate) intercepts: HashMap<String, Array1<Float>>,
    /// Task relationship matrix (similarity scores)
    pub(crate) relationship_matrix: Array2<Float>,
    /// Task names in order
    pub(crate) task_names: Vec<String>,
    /// Number of input features
    pub(crate) n_features: usize,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Training parameters
    pub(crate) relationship_strength: Float,
    pub(crate) similarity_threshold: Float,
    pub(crate) similarity_method: TaskSimilarityMethod,
    /// Training iterations performed
    pub(crate) n_iter: usize,
}

impl TaskRelationshipLearning<Untrained> {
    /// Create a new TaskRelationshipLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            relationship_strength: 1.0,
            similarity_threshold: 0.5,
            base_alpha: 1.0,
            max_iter: 1000,
            tolerance: 1e-4,
            learning_rate: 0.01,
            task_outputs: HashMap::new(),
            fit_intercept: true,
            similarity_method: TaskSimilarityMethod::Correlation,
        }
    }

    /// Set relationship regularization strength
    pub fn relationship_strength(mut self, strength: Float) -> Self {
        self.relationship_strength = strength;
        self
    }

    /// Set similarity threshold for relationships
    pub fn similarity_threshold(mut self, threshold: Float) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set base regularization strength
    pub fn base_alpha(mut self, alpha: Float) -> Self {
        self.base_alpha = alpha;
        self
    }

    /// Set task similarity method
    pub fn similarity_method(mut self, method: TaskSimilarityMethod) -> Self {
        self.similarity_method = method;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set task outputs
    pub fn task_outputs(mut self, outputs: &[(&str, usize)]) -> Self {
        self.task_outputs = outputs
            .iter()
            .map(|(name, size)| (name.to_string(), *size))
            .collect();
        self
    }
}

impl Default for TaskRelationshipLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TaskRelationshipLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for TaskRelationshipLearning<Untrained>
{
    type Fitted = TaskRelationshipLearning<TaskRelationshipLearningTrained>;

    fn fit(
        self,
        X: &ArrayView2<'_, Float>,
        y: &HashMap<String, Array2<Float>>,
    ) -> SklResult<Self::Fitted> {
        let x = X.to_owned();
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let task_names: Vec<String> = y.keys().cloned().collect();
        let n_tasks = task_names.len();

        // Initialize task coefficients
        let mut task_coefficients: HashMap<String, Array2<Float>> = HashMap::new();
        let mut task_intercepts: HashMap<String, Array1<Float>> = HashMap::new();

        let mut rng_gen = thread_rng();

        for (task_name, y_task) in y {
            let n_outputs = y_task.ncols();
            let mut coef = Array2::<Float>::zeros((n_features, n_outputs));
            let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
            for i in 0..n_features {
                for j in 0..n_outputs {
                    coef[[i, j]] = rng_gen.sample(normal_dist);
                }
            }
            let intercept = Array1::<Float>::zeros(n_outputs);
            task_coefficients.insert(task_name.clone(), coef);
            task_intercepts.insert(task_name.clone(), intercept);
        }

        // Compute task similarity matrix
        let mut relationship_matrix = Array2::<Float>::zeros((n_tasks, n_tasks));

        for (i, task_i) in task_names.iter().enumerate() {
            for (j, task_j) in task_names.iter().enumerate() {
                if i != j {
                    let similarity = self.compute_task_similarity(
                        &y[task_i],
                        &y[task_j],
                        &self.similarity_method,
                    );
                    relationship_matrix[[i, j]] = similarity;
                } else {
                    relationship_matrix[[i, j]] = 1.0;
                }
            }
        }

        // Training loop
        let mut prev_loss = Float::INFINITY;
        let mut n_iter = 0;

        for iteration in 0..self.max_iter {
            let mut total_loss = 0.0;

            // Update coefficients for each task
            for (task_name, y_task) in y {
                let current_coef = &task_coefficients[task_name];
                let current_intercept = &task_intercepts[task_name];

                // Compute predictions
                let predictions = x.dot(current_coef);
                let predictions_with_intercept = &predictions + current_intercept;

                // Compute residuals
                let residuals = &predictions_with_intercept - y_task;

                // Compute gradients
                let grad_coef = x.t().dot(&residuals) / (n_samples as Float);
                let grad_intercept = residuals.sum_axis(Axis(0)) / (n_samples as Float);

                // Add relationship regularization
                let mut reg_grad_coef = grad_coef.clone();

                // Find current task index
                let task_idx = task_names.iter().position(|t| t == task_name).unwrap();

                // Add relationship penalties
                for (other_idx, other_task) in task_names.iter().enumerate() {
                    if other_task != task_name {
                        let similarity = relationship_matrix[[task_idx, other_idx]];
                        if similarity > self.similarity_threshold {
                            let relationship_penalty = &(current_coef
                                - &task_coefficients[other_task])
                                * self.relationship_strength
                                * similarity;
                            reg_grad_coef = reg_grad_coef + relationship_penalty;
                        }
                    }
                }

                // Update parameters
                let new_coef = current_coef - &(&reg_grad_coef * self.learning_rate);
                let new_intercept = current_intercept - &(&grad_intercept * self.learning_rate);

                task_coefficients.insert(task_name.clone(), new_coef);
                task_intercepts.insert(task_name.clone(), new_intercept);

                // Add to loss
                total_loss += residuals.mapv(|x| x * x).sum();
            }

            // Check convergence
            if (prev_loss - total_loss).abs() < self.tolerance {
                n_iter = iteration + 1;
                break;
            }
            prev_loss = total_loss;
            n_iter = iteration + 1;
        }

        Ok(TaskRelationshipLearning {
            state: TaskRelationshipLearningTrained {
                coefficients: task_coefficients,
                intercepts: task_intercepts,
                relationship_matrix,
                task_names,
                n_features,
                task_outputs: self.task_outputs.clone(),
                relationship_strength: self.relationship_strength,
                similarity_threshold: self.similarity_threshold,
                similarity_method: self.similarity_method.clone(),
                n_iter,
            },
            relationship_strength: self.relationship_strength,
            similarity_threshold: self.similarity_threshold,
            base_alpha: self.base_alpha,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            task_outputs: self.task_outputs,
            fit_intercept: self.fit_intercept,
            similarity_method: self.similarity_method,
        })
    }
}

impl TaskRelationshipLearning<Untrained> {
    fn compute_task_similarity(
        &self,
        y1: &Array2<Float>,
        y2: &Array2<Float>,
        method: &TaskSimilarityMethod,
    ) -> Float {
        match method {
            TaskSimilarityMethod::Correlation => {
                // Compute correlation between task outputs
                let y1_flat: Vec<Float> = y1.iter().copied().collect();
                let y2_flat: Vec<Float> = y2.iter().copied().collect();

                if y1_flat.len() != y2_flat.len() {
                    return 0.0;
                }

                let mean1: Float = y1_flat.iter().sum::<Float>() / y1_flat.len() as Float;
                let mean2: Float = y2_flat.iter().sum::<Float>() / y2_flat.len() as Float;

                let mut num = 0.0;
                let mut den1 = 0.0;
                let mut den2 = 0.0;

                for (v1, v2) in y1_flat.iter().zip(y2_flat.iter()) {
                    let d1 = v1 - mean1;
                    let d2 = v2 - mean2;
                    num += d1 * d2;
                    den1 += d1 * d1;
                    den2 += d2 * d2;
                }

                if den1 > 0.0 && den2 > 0.0 {
                    (num / (den1.sqrt() * den2.sqrt())).abs()
                } else {
                    0.0
                }
            }
            TaskSimilarityMethod::Cosine => {
                // Compute cosine similarity
                let y1_flat: Vec<Float> = y1.iter().copied().collect();
                let y2_flat: Vec<Float> = y2.iter().copied().collect();

                let dot_product: Float =
                    y1_flat.iter().zip(y2_flat.iter()).map(|(a, b)| a * b).sum();
                let norm1: Float = y1_flat.iter().map(|x| x * x).sum::<Float>().sqrt();
                let norm2: Float = y2_flat.iter().map(|x| x * x).sum::<Float>().sqrt();

                if norm1 > 0.0 && norm2 > 0.0 {
                    (dot_product / (norm1 * norm2)).abs()
                } else {
                    0.0
                }
            }
            TaskSimilarityMethod::Euclidean => {
                // Compute inverse euclidean distance as similarity
                let y1_flat: Vec<Float> = y1.iter().copied().collect();
                let y2_flat: Vec<Float> = y2.iter().copied().collect();

                let distance: Float = y1_flat
                    .iter()
                    .zip(y2_flat.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<Float>()
                    .sqrt();

                1.0 / (1.0 + distance)
            }
            TaskSimilarityMethod::MutualInformation => {
                // Simple approximation of mutual information using correlation
                self.compute_task_similarity(y1, y2, &TaskSimilarityMethod::Correlation)
            }
        }
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for TaskRelationshipLearning<TaskRelationshipLearningTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<HashMap<String, Array2<Float>>> {
        let x = X.to_owned();
        let (n_samples, n_features) = x.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let mut predictions = HashMap::new();

        for (task_name, coef) in &self.state.coefficients {
            let task_predictions = x.dot(coef);
            let intercept = &self.state.intercepts[task_name];
            let final_predictions = &task_predictions + intercept;
            predictions.insert(task_name.clone(), final_predictions);
        }

        Ok(predictions)
    }
}

impl TaskRelationshipLearningTrained {
    /// Get coefficients for a specific task
    pub fn task_coefficients(&self, task_name: &str) -> Option<&Array2<Float>> {
        self.coefficients.get(task_name)
    }

    /// Get intercepts for a specific task
    pub fn task_intercepts(&self, task_name: &str) -> Option<&Array1<Float>> {
        self.intercepts.get(task_name)
    }

    /// Get the relationship matrix (task similarity scores)
    pub fn relationship_matrix(&self) -> &Array2<Float> {
        &self.relationship_matrix
    }

    /// Get task names in order
    pub fn task_names(&self) -> &Vec<String> {
        &self.task_names
    }

    /// Get similarity score between two tasks
    pub fn task_similarity(&self, task1: &str, task2: &str) -> Option<Float> {
        let idx1 = self.task_names.iter().position(|t| t == task1)?;
        let idx2 = self.task_names.iter().position(|t| t == task2)?;
        Some(self.relationship_matrix[[idx1, idx2]])
    }

    /// Get related tasks for a given task (similarity above threshold)
    pub fn related_tasks(&self, task_name: &str) -> Vec<(&String, Float)> {
        if let Some(task_idx) = self.task_names.iter().position(|t| t == task_name) {
            self.task_names
                .iter()
                .enumerate()
                .filter_map(|(other_idx, other_task)| {
                    if other_idx != task_idx {
                        let similarity = self.relationship_matrix[[task_idx, other_idx]];
                        if similarity > self.similarity_threshold {
                            Some((other_task, similarity))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}
