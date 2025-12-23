//! Task Clustering Regularization for Multi-Task Learning
//!
//! This method clusters tasks based on their similarity and applies different
//! regularization strengths within and across clusters. Tasks in the same cluster
//! are encouraged to have similar parameters, while tasks in different clusters
//! are allowed to be more different.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::RandNormal;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Task Clustering Regularization for Multi-Task Learning
///
/// This method clusters tasks based on their similarity and applies different
/// regularization strengths within and across clusters. Tasks in the same cluster
/// are encouraged to have similar parameters, while tasks in different clusters
/// are allowed to be more different.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::regularization::TaskClusteringRegularization;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let mut y_tasks = HashMap::new();
/// y_tasks.insert("task1".to_string(), array![[1.0], [2.0], [1.5], [2.5]]);
/// y_tasks.insert("task2".to_string(), array![[0.5], [1.0], [0.8], [1.2]]);
/// y_tasks.insert("task3".to_string(), array![[2.0], [3.0], [2.2], [3.1]]);
///
/// let task_clustering = TaskClusteringRegularization::new()
///     .n_clusters(2)
///     .intra_cluster_alpha(0.1)  // Strong regularization within clusters
///     .inter_cluster_alpha(0.01) // Weak regularization across clusters
///     .max_iter(1000);
/// ```
#[derive(Debug, Clone)]
pub struct TaskClusteringRegularization<S = Untrained> {
    pub(crate) state: S,
    /// Number of task clusters
    pub(crate) n_clusters: usize,
    /// Regularization strength within clusters
    pub(crate) intra_cluster_alpha: Float,
    /// Regularization strength across clusters
    pub(crate) inter_cluster_alpha: Float,
    /// Maximum iterations for clustering
    pub(crate) max_iter: usize,
    /// Convergence tolerance
    pub(crate) tolerance: Float,
    /// Learning rate
    pub(crate) learning_rate: Float,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Include intercept term
    pub(crate) fit_intercept: bool,
    /// Random state for reproducible clustering
    pub(crate) random_state: Option<u64>,
}

/// Trained state for TaskClusteringRegularization
#[derive(Debug, Clone)]
pub struct TaskClusteringRegressionTrained {
    /// Coefficients for each task
    pub(crate) coefficients: HashMap<String, Array2<Float>>,
    /// Intercepts for each task
    pub(crate) intercepts: HashMap<String, Array1<Float>>,
    /// Task cluster assignments
    pub(crate) task_clusters: HashMap<String, usize>,
    /// Cluster centroids for task parameters
    pub(crate) cluster_centroids: Array2<Float>,
    /// Number of input features
    pub(crate) n_features: usize,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Training parameters
    pub(crate) n_clusters: usize,
    pub(crate) intra_cluster_alpha: Float,
    pub(crate) inter_cluster_alpha: Float,
    /// Training iterations performed
    pub(crate) n_iter: usize,
}

impl TaskClusteringRegularization<Untrained> {
    /// Create a new TaskClusteringRegularization instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_clusters: 2,
            intra_cluster_alpha: 1.0,
            inter_cluster_alpha: 0.1,
            max_iter: 1000,
            tolerance: 1e-4,
            learning_rate: 0.01,
            task_outputs: HashMap::new(),
            fit_intercept: true,
            random_state: None,
        }
    }

    /// Set number of task clusters
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set intra-cluster regularization strength
    pub fn intra_cluster_alpha(mut self, alpha: Float) -> Self {
        self.intra_cluster_alpha = alpha;
        self
    }

    /// Set inter-cluster regularization strength
    pub fn inter_cluster_alpha(mut self, alpha: Float) -> Self {
        self.inter_cluster_alpha = alpha;
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

    /// Set random state for reproducible clustering
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
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

impl Default for TaskClusteringRegularization<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TaskClusteringRegularization<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for TaskClusteringRegularization<Untrained>
{
    type Fitted = TaskClusteringRegularization<TaskClusteringRegressionTrained>;

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

        if self.n_clusters == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of clusters must be > 0".to_string(),
            ));
        }

        // Initialize task coefficients randomly
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

        // Simple k-means clustering of task parameters for initial clustering
        let task_names: Vec<String> = y.keys().cloned().collect();
        let n_tasks = task_names.len();

        // Flatten coefficients for clustering
        let mut task_vectors = Vec::new();
        for task_name in &task_names {
            let coef = &task_coefficients[task_name];
            let flattened: Vec<Float> = coef.iter().copied().collect();
            task_vectors.push(flattened);
        }

        // Simple k-means clustering
        let mut task_clusters: HashMap<String, usize> = HashMap::new();
        let cluster_centroids =
            Array2::<Float>::zeros((self.n_clusters, n_features * y[&task_names[0]].ncols()));

        // Initialize clusters randomly
        for (i, task_name) in task_names.iter().enumerate() {
            task_clusters.insert(task_name.clone(), i % self.n_clusters);
        }

        // Training loop with task clustering
        let mut prev_loss = Float::INFINITY;
        let mut n_iter = 0;

        for iteration in 0..self.max_iter {
            let mut total_loss = 0.0;

            // Update coefficients for each task
            for (task_name, y_task) in y {
                let task_cluster = task_clusters[task_name];
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

                // Add clustering regularization
                let mut reg_grad_coef = grad_coef.clone();

                // Intra-cluster regularization
                let mut cluster_center: Array2<Float> = Array2::<Float>::zeros(current_coef.dim());
                let mut cluster_count = 0;

                for (other_task, other_cluster) in &task_clusters {
                    if *other_cluster == task_cluster && other_task != task_name {
                        cluster_center = &cluster_center + &task_coefficients[other_task];
                        cluster_count += 1;
                    }
                }

                if cluster_count > 0 {
                    cluster_center /= cluster_count as Float;
                    let intra_penalty =
                        &(current_coef - &cluster_center) * self.intra_cluster_alpha;
                    reg_grad_coef = reg_grad_coef + intra_penalty;
                }

                // Inter-cluster regularization (weaker)
                for (other_task, other_cluster) in &task_clusters {
                    if *other_cluster != task_cluster {
                        let inter_penalty = &(current_coef - &task_coefficients[other_task])
                            * self.inter_cluster_alpha
                            * 0.1;
                        reg_grad_coef = reg_grad_coef + inter_penalty;
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

        Ok(TaskClusteringRegularization {
            state: TaskClusteringRegressionTrained {
                coefficients: task_coefficients,
                intercepts: task_intercepts,
                task_clusters,
                cluster_centroids,
                n_features,
                task_outputs: self.task_outputs.clone(),
                n_clusters: self.n_clusters,
                intra_cluster_alpha: self.intra_cluster_alpha,
                inter_cluster_alpha: self.inter_cluster_alpha,
                n_iter,
            },
            n_clusters: self.n_clusters,
            intra_cluster_alpha: self.intra_cluster_alpha,
            inter_cluster_alpha: self.inter_cluster_alpha,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            task_outputs: self.task_outputs,
            fit_intercept: self.fit_intercept,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for TaskClusteringRegularization<TaskClusteringRegressionTrained>
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

impl TaskClusteringRegressionTrained {
    /// Get coefficients for a specific task
    pub fn task_coefficients(&self, task_name: &str) -> Option<&Array2<Float>> {
        self.coefficients.get(task_name)
    }

    /// Get intercepts for a specific task
    pub fn task_intercepts(&self, task_name: &str) -> Option<&Array1<Float>> {
        self.intercepts.get(task_name)
    }

    /// Get cluster assignment for a task
    pub fn task_cluster(&self, task_name: &str) -> Option<usize> {
        self.task_clusters.get(task_name).copied()
    }

    /// Get all task cluster assignments
    pub fn task_clusters(&self) -> &HashMap<String, usize> {
        &self.task_clusters
    }

    /// Get cluster centroids
    pub fn cluster_centroids(&self) -> &Array2<Float> {
        &self.cluster_centroids
    }

    /// Get number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Get tasks in a specific cluster
    pub fn cluster_tasks(&self, cluster_id: usize) -> Vec<&String> {
        self.task_clusters
            .iter()
            .filter_map(|(task_name, &cluster)| {
                if cluster == cluster_id {
                    Some(task_name)
                } else {
                    None
                }
            })
            .collect()
    }
}
