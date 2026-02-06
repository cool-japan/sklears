//! Shared representation learning for multi-task models
//!
//! This module implements shared representation learning where multiple tasks
//! learn common representations while maintaining task-specific components.
//! This is particularly useful when tasks are related and can benefit from
//! shared knowledge.

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Distribution;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

// Helper functions for safe operations
#[inline]
fn create_normal_distribution() -> Result<Normal<Float>> {
    Normal::new(0.0, 1.0).map_err(|e| {
        SklearsError::NumericalError(format!("Failed to create normal distribution: {}", e))
    })
}

#[inline]
fn compare_floats(a: &Float, b: &Float) -> Result<std::cmp::Ordering> {
    a.partial_cmp(b).ok_or_else(|| {
        SklearsError::InvalidInput("Cannot compare values: NaN encountered".to_string())
    })
}

/// Strategy for shared representation learning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SharedRepresentationStrategy {
    /// Linear shared representation: X -> Shared Features -> Task-specific layers
    Linear,
    /// Factorized representation: W = U * V^T where U is shared, V is task-specific
    Factorized,
    /// Hierarchical representation: multiple levels of shared features
    Hierarchical { n_levels: usize },
    /// Attention-based sharing: tasks attend to different parts of shared representation
    Attention,
}

/// Configuration for shared representation learning
#[derive(Debug, Clone)]
pub struct SharedRepresentationConfig {
    /// Strategy for shared representation
    pub strategy: SharedRepresentationStrategy,
    /// Dimension of shared representation
    pub shared_dim: usize,
    /// Dimension of task-specific representation
    pub task_specific_dim: usize,
    /// L2 regularization for shared parameters
    pub shared_l2_reg: Float,
    /// L2 regularization for task-specific parameters  
    pub task_l2_reg: Float,
    /// Weight for encouraging similarity between tasks
    pub task_similarity_weight: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate for optimization
    pub learning_rate: Float,
    /// Whether to use orthogonality constraints on shared features
    pub orthogonal_shared: bool,
    /// Whether to normalize shared representations
    pub normalize_shared: bool,
}

impl Default for SharedRepresentationConfig {
    fn default() -> Self {
        Self {
            strategy: SharedRepresentationStrategy::Linear,
            shared_dim: 10,
            task_specific_dim: 5,
            shared_l2_reg: 0.01,
            task_l2_reg: 0.01,
            task_similarity_weight: 0.1,
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            orthogonal_shared: false,
            normalize_shared: true,
        }
    }
}

/// Multi-task model with shared representation learning
#[derive(Debug, Clone)]
pub struct MultiTaskSharedRepresentation<State = Untrained> {
    config: SharedRepresentationConfig,
    state: PhantomData<State>,
    // Model parameters
    shared_weights_: Option<Array2<Float>>, // X -> shared representation
    task_weights_: Option<Array3<Float>>,   // shared + task-specific -> output
    task_input_weights_: Option<Array3<Float>>, // raw input -> task-specific features
    shared_bias_: Option<Array1<Float>>,    // bias for shared representation
    task_bias_: Option<Array2<Float>>,      // bias for each task
    attention_weights_: Option<Array2<Float>>, // attention weights for shared features
    // Model metadata
    n_features_: Option<usize>,
    n_tasks_: Option<usize>,
    n_iter_: Option<usize>,
}

impl MultiTaskSharedRepresentation<Untrained> {
    /// Create a new shared representation model
    pub fn new() -> Self {
        Self {
            config: SharedRepresentationConfig::default(),
            state: PhantomData,
            shared_weights_: None,
            task_weights_: None,
            task_input_weights_: None,
            shared_bias_: None,
            task_bias_: None,
            attention_weights_: None,
            n_features_: None,
            n_tasks_: None,
            n_iter_: None,
        }
    }

    /// Set shared representation dimension
    pub fn shared_dim(mut self, dim: usize) -> Self {
        self.config.shared_dim = dim;
        self
    }

    /// Set task-specific dimension
    pub fn task_specific_dim(mut self, dim: usize) -> Self {
        self.config.task_specific_dim = dim;
        self
    }

    /// Set regularization parameters
    pub fn regularization(mut self, shared_l2: Float, task_l2: Float) -> Self {
        self.config.shared_l2_reg = shared_l2;
        self.config.task_l2_reg = task_l2;
        self
    }

    /// Set representation strategy
    pub fn strategy(mut self, strategy: SharedRepresentationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Builder pattern
    pub fn builder() -> SharedRepresentationBuilder {
        SharedRepresentationBuilder::default()
    }
}

/// Builder for SharedRepresentation
#[derive(Debug, Default)]
pub struct SharedRepresentationBuilder {
    config: SharedRepresentationConfig,
}

impl SharedRepresentationBuilder {
    pub fn strategy(mut self, strategy: SharedRepresentationStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn shared_dim(mut self, dim: usize) -> Self {
        self.config.shared_dim = dim;
        self
    }

    pub fn task_specific_dim(mut self, dim: usize) -> Self {
        self.config.task_specific_dim = dim;
        self
    }

    pub fn regularization(mut self, shared_l2: Float, task_l2: Float) -> Self {
        self.config.shared_l2_reg = shared_l2;
        self.config.task_l2_reg = task_l2;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn build(self) -> MultiTaskSharedRepresentation<Untrained> {
        MultiTaskSharedRepresentation {
            config: self.config,
            state: PhantomData,
            shared_weights_: None,
            task_weights_: None,
            task_input_weights_: None,
            shared_bias_: None,
            task_bias_: None,
            attention_weights_: None,
            n_features_: None,
            n_tasks_: None,
            n_iter_: None,
        }
    }
}

impl Default for MultiTaskSharedRepresentation<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> Estimator<State> for MultiTaskSharedRepresentation<State> {
    type Float = Float;
    type Config = SharedRepresentationConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<State> MultiTaskSharedRepresentation<State> {
    /// Helper function to compute shared features
    fn compute_shared_features(
        &self,
        x: &Array2<Float>,
        shared_weights: &Array2<Float>,
        shared_bias: &Array1<Float>,
    ) -> Array2<Float> {
        let features = x.dot(shared_weights);
        let bias_broadcasted = shared_bias.clone().insert_axis(Axis(0));
        let result = features + &bias_broadcasted;

        if self.config.normalize_shared {
            // Apply normalization
            let mut normalized = result.clone();
            for mut row in normalized.axis_iter_mut(Axis(0)) {
                let norm = row.mapv(|x| x * x).sum().sqrt();
                if norm > 0.0 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
            normalized
        } else {
            result
        }
    }

    /// Helper function to compute task-specific features
    fn compute_task_features(
        &self,
        x: &Array2<Float>,
        task_weights: &Array3<Float>,
        task_id: usize,
    ) -> Array2<Float> {
        let task_w = task_weights.slice(s![task_id, .., ..]);
        x.dot(&task_w.slice(s![.., ..self.config.task_specific_dim]))
    }

    /// Helper function to combine shared and task-specific features
    fn combine_features(
        &self,
        shared_features: &Array2<Float>,
        task_features: &Array2<Float>,
    ) -> Array2<Float> {
        // Concatenate along feature dimension
        let mut combined = Array2::zeros((
            shared_features.nrows(),
            shared_features.ncols() + task_features.ncols(),
        ));
        combined
            .slice_mut(s![.., ..shared_features.ncols()])
            .assign(shared_features);
        combined
            .slice_mut(s![.., shared_features.ncols()..])
            .assign(task_features);
        combined
    }
}

impl Fit<Array2<Float>, Array2<Float>> for MultiTaskSharedRepresentation<Untrained> {
    type Fitted = MultiTaskSharedRepresentation<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_tasks = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == Y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, Y.shape[0]={}", n_samples, y.nrows()),
            });
        }

        if n_samples == 0 || n_features == 0 || n_tasks == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        let fitted_model = match self.config.strategy {
            SharedRepresentationStrategy::Linear => {
                self.fit_linear_shared(x, y, n_features, n_tasks)?
            }
            SharedRepresentationStrategy::Factorized => {
                self.fit_factorized(x, y, n_features, n_tasks)?
            }
            SharedRepresentationStrategy::Hierarchical { n_levels } => {
                self.fit_hierarchical(x, y, n_features, n_tasks, n_levels)?
            }
            SharedRepresentationStrategy::Attention => {
                self.fit_attention_based(x, y, n_features, n_tasks)?
            }
        };

        Ok(fitted_model)
    }
}

impl MultiTaskSharedRepresentation<Untrained> {
    /// Fit linear shared representation model
    fn fit_linear_shared(
        self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        n_features: usize,
        n_tasks: usize,
    ) -> Result<MultiTaskSharedRepresentation<Trained>> {
        let shared_dim = self.config.shared_dim;
        let task_dim = self.config.task_specific_dim;

        // Initialize parameters
        let mut rng = thread_rng();
        let normal = create_normal_distribution()?;
        let mut shared_weights = Array2::zeros((n_features, shared_dim));
        for elem in shared_weights.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut task_weights = Array3::zeros((n_tasks, shared_dim + task_dim, 1));
        for elem in task_weights.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut shared_bias = Array1::zeros(shared_dim);
        let mut task_bias = Array2::zeros((n_tasks, 1));

        // Task-specific input projections for additional features
        let mut task_input_weights = Array3::zeros((n_tasks, n_features, task_dim));
        for elem in task_input_weights.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }

        let mut n_iter = 0;

        // Gradient descent optimization
        for iter in 0..self.config.max_iter {
            let mut total_loss = 0.0;

            // Forward pass for all tasks
            let shared_features = self.compute_shared_features(x, &shared_weights, &shared_bias);

            let mut shared_weights_grad: Array2<Float> = Array2::zeros(shared_weights.raw_dim());
            let mut shared_bias_grad: Array1<Float> = Array1::zeros(shared_bias.raw_dim());
            let mut task_weights_grad: Array3<Float> = Array3::zeros(task_weights.raw_dim());
            let mut task_bias_grad: Array2<Float> = Array2::zeros(task_bias.raw_dim());
            let mut task_input_grad = Array3::zeros(task_input_weights.raw_dim());

            for task_id in 0..n_tasks {
                // Compute task-specific features
                let task_input_proj = self.compute_task_features(x, &task_input_weights, task_id);

                // Combine shared and task-specific features
                let combined_features = self.combine_features(&shared_features, &task_input_proj);

                // Predict for current task
                let task_w = task_weights.slice(s![task_id, .., ..]);
                let task_b = task_bias.slice(s![task_id, ..]);
                let predictions: Array1<Float> =
                    combined_features.dot(&task_w.slice(s![.., 0])) + task_b[0];

                // Compute loss and gradients
                let y_task = y.column(task_id);
                let residual = predictions.to_owned() - y_task.to_owned();
                let task_loss = residual.mapv(|x| x * x).sum() / (2.0 * x.nrows() as Float);
                total_loss += task_loss;

                // Backward pass
                let n_samples_f = x.nrows() as Float;

                // Task-specific weight gradients
                for i in 0..combined_features.ncols() {
                    task_weights_grad[[task_id, i, 0]] =
                        combined_features.column(i).dot(&residual) / n_samples_f;
                }
                task_bias_grad[[task_id, 0]] = residual.sum() / n_samples_f;

                // Shared feature gradients (accumulated across tasks)
                let task_w_shared = task_w.slice(s![..shared_dim, 0]);
                let shared_grad_contrib = Array2::from_shape_vec(
                    (x.nrows(), shared_dim),
                    (0..x.nrows())
                        .flat_map(|i| {
                            (0..shared_dim)
                                .map(|j| residual[i] * task_w_shared[j] / n_samples_f)
                                .collect::<Vec<_>>()
                        })
                        .collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Accumulate gradients for shared weights
                for i in 0..n_features {
                    for j in 0..shared_dim {
                        shared_weights_grad[[i, j]] +=
                            x.column(i).dot(&shared_grad_contrib.column(j));
                    }
                }

                for j in 0..shared_dim {
                    shared_bias_grad[j] += shared_grad_contrib.column(j).sum();
                }

                // Task-specific input weight gradients
                let task_w_specific = task_w.slice(s![shared_dim.., 0]);
                for i in 0..n_features {
                    for j in 0..task_dim {
                        task_input_grad[[task_id, i, j]] =
                            x.column(i).dot(&residual) * task_w_specific[j] / n_samples_f;
                    }
                }
            }

            // Add regularization
            total_loss += self.config.shared_l2_reg * shared_weights.mapv(|x| x * x).sum();
            total_loss += self.config.task_l2_reg * task_weights.mapv(|x| x * x).sum();

            // Add regularization to gradients
            shared_weights_grad =
                shared_weights_grad + &shared_weights * (2.0 * self.config.shared_l2_reg);
            task_weights_grad = task_weights_grad + &task_weights * (2.0 * self.config.task_l2_reg);

            // Update parameters
            shared_weights = shared_weights - &shared_weights_grad * self.config.learning_rate;
            shared_bias = shared_bias - &shared_bias_grad * self.config.learning_rate;
            task_weights = task_weights - &task_weights_grad * self.config.learning_rate;
            task_bias = task_bias - &task_bias_grad * self.config.learning_rate;
            task_input_weights = task_input_weights - &task_input_grad * self.config.learning_rate;

            // Apply orthogonality constraint if requested
            if self.config.orthogonal_shared {
                shared_weights = self.apply_orthogonality_constraint(shared_weights)?;
            }

            // Check convergence
            if iter > 0 && total_loss.abs() < self.config.tol {
                n_iter += 1;
                break;
            }
            n_iter += 1;
        }

        // Combine task-specific weights with shared weights for final model
        let mut final_task_weights = Array3::zeros((n_tasks, shared_dim + task_dim, 1));
        for task_id in 0..n_tasks {
            final_task_weights
                .slice_mut(s![task_id, .., ..])
                .assign(&task_weights.slice(s![task_id, .., ..]));
        }

        Ok(MultiTaskSharedRepresentation {
            config: self.config,
            state: PhantomData,
            shared_weights_: Some(shared_weights),
            task_weights_: Some(final_task_weights),
            task_input_weights_: Some(task_input_weights),
            shared_bias_: Some(shared_bias),
            task_bias_: Some(task_bias),
            attention_weights_: None,
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
            n_iter_: Some(n_iter),
        })
    }

    /// Fit factorized representation model
    fn fit_factorized(
        self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        n_features: usize,
        n_tasks: usize,
    ) -> Result<MultiTaskSharedRepresentation<Trained>> {
        let rank = self.config.shared_dim;

        // Factorize W = U * V^T where U is shared (n_features x rank), V is task-specific (n_tasks x rank)
        let mut rng = thread_rng();
        let normal = create_normal_distribution()?;
        let mut u_shared = Array2::zeros((n_features, rank));
        for elem in u_shared.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut v_tasks = Array2::zeros((n_tasks, rank));
        for elem in v_tasks.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut bias = Array1::zeros(n_tasks);

        let mut n_iter = 0;

        // Alternating least squares optimization
        for _iter in 0..self.config.max_iter {
            let old_u = u_shared.clone();
            let old_v = v_tasks.clone();

            // Update U (shared component) for fixed V
            for i in 0..n_features {
                let x_i = x.column(i);
                let mut numerator = Array1::zeros(rank);
                let mut denominator = Array2::zeros((rank, rank));

                for task_id in 0..n_tasks {
                    let y_task = y.column(task_id);
                    let v_task = v_tasks.row(task_id);

                    // Compute residual excluding current feature
                    let mut residual = y_task.to_owned();
                    residual.mapv_inplace(|v| v - bias[task_id]);
                    for j in 0..n_features {
                        if j != i {
                            let u_j = u_shared.row(j);
                            let scalar = u_j.dot(&v_task);
                            residual -= &(x.column(j).to_owned() * scalar);
                        }
                    }

                    // Accumulate for least squares solution
                    numerator += &(v_task.to_owned() * x_i.dot(&residual));
                    for r1 in 0..rank {
                        for r2 in 0..rank {
                            denominator[[r1, r2]] += v_task[r1] * v_task[r2] * x_i.dot(&x_i);
                        }
                    }
                }

                // Add regularization
                for r in 0..rank {
                    denominator[[r, r]] += self.config.shared_l2_reg;
                }

                // Solve for u_i
                if let Ok(u_new) = self.solve_linear_system(&denominator, &numerator) {
                    u_shared.row_mut(i).assign(&u_new);
                }
            }

            // Update V (task-specific components) for fixed U
            for task_id in 0..n_tasks {
                let y_task = y.column(task_id);
                let mut numerator = Array1::zeros(rank);
                let mut denominator = Array2::zeros((rank, rank));

                // Compute X * U for current task
                let xu = x.dot(&u_shared);

                // Accumulate for least squares solution
                for s in 0..x.nrows() {
                    let xu_s = xu.row(s);
                    numerator += &(xu_s.to_owned() * (y_task[s] - bias[task_id]));
                    for r1 in 0..rank {
                        for r2 in 0..rank {
                            denominator[[r1, r2]] += xu_s[r1] * xu_s[r2];
                        }
                    }
                }

                // Add regularization
                for r in 0..rank {
                    denominator[[r, r]] += self.config.task_l2_reg;
                }

                // Solve for v_task
                if let Ok(v_new) = self.solve_linear_system(&denominator, &numerator) {
                    v_tasks.row_mut(task_id).assign(&v_new);
                }
            }

            // Update bias
            for task_id in 0..n_tasks {
                let y_task = y.column(task_id);
                let xu = x.dot(&u_shared);
                let v_task = v_tasks.row(task_id);

                let predictions: Array1<Float> = xu.dot(&v_task);
                bias[task_id] = (y_task.to_owned() - predictions).mean().unwrap_or(0.0);
            }

            // Check convergence
            let u_change = (&u_shared - &old_u).mapv(Float::abs).sum();
            let v_change = (&v_tasks - &old_v).mapv(Float::abs).sum();

            if u_change + v_change < self.config.tol {
                n_iter += 1;
                break;
            }
            n_iter += 1;
        }

        // Reconstruct task weights from factorization
        let mut task_weights = Array3::zeros((n_tasks, n_features, 1));
        for task_id in 0..n_tasks {
            let v_task = v_tasks.row(task_id);
            for i in 0..n_features {
                let u_i = u_shared.row(i);
                task_weights[[task_id, i, 0]] = u_i.dot(&v_task);
            }
        }

        let task_bias = Array2::from_shape_vec((n_tasks, 1), bias.to_vec())
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        Ok(MultiTaskSharedRepresentation {
            config: self.config,
            state: PhantomData,
            shared_weights_: Some(u_shared),
            task_weights_: Some(task_weights),
            task_input_weights_: None,
            shared_bias_: Some(Array1::zeros(rank)),
            task_bias_: Some(task_bias),
            attention_weights_: None,
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
            n_iter_: Some(n_iter),
        })
    }

    /// Fit hierarchical representation model
    fn fit_hierarchical(
        self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        n_features: usize,
        n_tasks: usize,
        n_levels: usize,
    ) -> Result<MultiTaskSharedRepresentation<Trained>> {
        // For simplicity, implement as multiple linear layers
        // This could be extended to more sophisticated hierarchical structures
        let total_dim = self.config.shared_dim * n_levels;
        let config_modified = SharedRepresentationConfig {
            shared_dim: total_dim,
            ..self.config
        };

        let modified_self = MultiTaskSharedRepresentation {
            config: config_modified,
            ..self
        };

        modified_self.fit_linear_shared(x, y, n_features, n_tasks)
    }

    /// Fit attention-based shared representation model
    fn fit_attention_based(
        self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        n_features: usize,
        n_tasks: usize,
    ) -> Result<MultiTaskSharedRepresentation<Trained>> {
        let shared_dim = self.config.shared_dim;

        // Initialize shared representation and attention weights
        let mut rng = thread_rng();
        let normal = create_normal_distribution()?;
        let mut shared_weights = Array2::zeros((n_features, shared_dim));
        for elem in shared_weights.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut attention_weights = Array2::zeros((n_tasks, shared_dim));
        for elem in attention_weights.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut task_weights = Array3::zeros((n_tasks, shared_dim, 1));
        for elem in task_weights.iter_mut() {
            *elem = normal.sample(&mut rng) * 0.1;
        }
        let mut bias = Array1::zeros(n_tasks);

        let mut n_iter = 0;

        // Gradient descent with attention mechanism
        for iter in 0..self.config.max_iter {
            let mut total_loss = 0.0;

            // Compute shared features
            let shared_features = x.dot(&shared_weights);

            // Initialize gradients
            let mut shared_weights_grad: Array2<Float> = Array2::zeros(shared_weights.raw_dim());
            let mut attention_grad: Array2<Float> = Array2::zeros(attention_weights.raw_dim());
            let mut task_weights_grad: Array3<Float> = Array3::zeros(task_weights.raw_dim());
            let mut bias_grad: Array1<Float> = Array1::zeros(bias.raw_dim());

            for task_id in 0..n_tasks {
                // Apply attention to shared features
                let attention_task = attention_weights.row(task_id);
                let attended_features = &shared_features * &attention_task.insert_axis(Axis(0));

                // Predict for current task
                let task_w = task_weights.slice(s![task_id, .., 0]);
                let predictions: Array1<Float> = attended_features.dot(&task_w) + bias[task_id];

                // Compute loss
                let y_task = y.column(task_id);
                let residual = predictions.to_owned() - y_task.to_owned();
                let task_loss = residual.mapv(|x| x * x).sum() / (2.0 * x.nrows() as Float);
                total_loss += task_loss;

                // Compute gradients
                let n_samples_f = x.nrows() as Float;

                // Task weight gradients
                for j in 0..shared_dim {
                    task_weights_grad[[task_id, j, 0]] =
                        attended_features.column(j).dot(&residual) / n_samples_f;
                }

                // Bias gradients
                bias_grad[task_id] = residual.sum() / n_samples_f;

                // Attention gradients
                for j in 0..shared_dim {
                    attention_grad[[task_id, j]] =
                        shared_features.column(j).dot(&residual) * task_w[j] / n_samples_f;
                }

                // Shared weight gradients (with attention weighting)
                for i in 0..n_features {
                    for j in 0..shared_dim {
                        shared_weights_grad[[i, j]] +=
                            x.column(i).dot(&residual) * attention_task[j] * task_w[j]
                                / n_samples_f;
                    }
                }
            }

            // Add regularization
            total_loss += self.config.shared_l2_reg * shared_weights.mapv(|x| x * x).sum();
            total_loss += self.config.task_l2_reg * task_weights.mapv(|x| x * x).sum();

            shared_weights_grad =
                shared_weights_grad + &shared_weights * (2.0 * self.config.shared_l2_reg);
            task_weights_grad = task_weights_grad + &task_weights * (2.0 * self.config.task_l2_reg);

            // Update parameters
            shared_weights = shared_weights - &shared_weights_grad * self.config.learning_rate;
            attention_weights = attention_weights - &attention_grad * self.config.learning_rate;
            task_weights = task_weights - &task_weights_grad * self.config.learning_rate;
            bias = bias - &bias_grad * self.config.learning_rate;

            // Normalize attention weights
            for task_id in 0..n_tasks {
                let attention_norm = attention_weights
                    .row(task_id)
                    .mapv(|x: Float| x * x)
                    .sum()
                    .sqrt();
                if attention_norm > 0.0 {
                    attention_weights
                        .row_mut(task_id)
                        .mapv_inplace(|x| x / attention_norm);
                }
            }

            // Check convergence
            if iter > 0 && total_loss.abs() < self.config.tol {
                n_iter += 1;
                break;
            }
            n_iter += 1;
        }

        let task_bias = Array2::from_shape_vec((n_tasks, 1), bias.to_vec())
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

        Ok(MultiTaskSharedRepresentation {
            config: self.config,
            state: PhantomData,
            shared_weights_: Some(shared_weights),
            task_weights_: Some(task_weights),
            task_input_weights_: None,
            shared_bias_: Some(Array1::zeros(shared_dim)),
            task_bias_: Some(task_bias),
            attention_weights_: Some(attention_weights),
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
            n_iter_: Some(n_iter),
        })
    }

    /// Apply orthogonality constraint to shared weights
    fn apply_orthogonality_constraint(&self, mut weights: Array2<Float>) -> Result<Array2<Float>> {
        // Simple QR decomposition for orthogonalization
        // In practice, you'd use a proper linear algebra library

        // For now, just normalize columns
        for mut col in weights.axis_iter_mut(Axis(1)) {
            let norm = col.mapv(|x| x * x).sum().sqrt();
            if norm > 0.0 {
                col.mapv_inplace(|x| x / norm);
            }
        }

        Ok(weights)
    }

    /// Solve linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
        // Simple implementation - in practice use proper linear algebra
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "Square matrix and matching vector".to_string(),
                actual: format!("Matrix {}x{}, vector {}", a.nrows(), a.ncols(), b.len()),
            });
        }

        // Use pseudo-inverse for simplicity (not optimal)
        let mut result = Array1::zeros(n);
        for i in 0..n {
            if a[[i, i]].abs() > 1e-10 {
                result[i] = b[i] / a[[i, i]];
            }
        }

        Ok(result)
    }
}

impl Predict<Array2<Float>, Array2<Float>> for MultiTaskSharedRepresentation<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        let n_tasks = self.n_tasks_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let shared_weights =
            self.shared_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;
        let task_weights = self
            .task_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let shared_bias = self
            .shared_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let task_bias = self
            .task_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array2::zeros((x.nrows(), n_tasks));

        match self.config.strategy {
            SharedRepresentationStrategy::Attention => {
                // Use attention-based prediction
                let attention_weights =
                    self.attention_weights_
                        .as_ref()
                        .ok_or_else(|| SklearsError::NotFitted {
                            operation: "predict".to_string(),
                        })?;
                let shared_features = self.compute_shared_features(x, shared_weights, shared_bias);

                for task_id in 0..n_tasks {
                    let attention_task = attention_weights.row(task_id);
                    let attended_features = &shared_features * &attention_task.insert_axis(Axis(0));
                    let task_w = task_weights.slice(s![task_id, .., 0]);
                    let dot_result: Array1<Float> = attended_features.dot(&task_w);
                    let task_pred = dot_result + task_bias[[task_id, 0]];
                    predictions.column_mut(task_id).assign(&task_pred);
                }
            }
            SharedRepresentationStrategy::Factorized => {
                for task_id in 0..n_tasks {
                    let task_w = task_weights.slice(s![task_id, .., 0]);
                    let dot_result: Array1<Float> = x.dot(&task_w);
                    let task_pred = dot_result + task_bias[[task_id, 0]];
                    predictions.column_mut(task_id).assign(&task_pred);
                }
            }
            _ => {
                // Linear or hierarchical shared representation prediction
                let shared_features = self.compute_shared_features(x, shared_weights, shared_bias);
                let needs_task_specific = self.config.task_specific_dim > 0;
                let task_input_weights = if needs_task_specific {
                    Some(self.task_input_weights_.as_ref().ok_or_else(|| {
                        SklearsError::InvalidState(
                            "Task input weights missing for linear shared representation"
                                .to_string(),
                        )
                    })?)
                } else {
                    None
                };

                for task_id in 0..n_tasks {
                    let feature_matrix = if let Some(task_input_weights) = task_input_weights {
                        let task_features =
                            self.compute_task_features(x, task_input_weights, task_id);
                        self.combine_features(&shared_features, &task_features)
                    } else {
                        shared_features.clone()
                    };

                    let task_w = task_weights.slice(s![task_id, .., 0]);
                    let dot_result: Array1<Float> = feature_matrix.dot(&task_w);
                    let task_pred = dot_result + task_bias[[task_id, 0]];
                    predictions.column_mut(task_id).assign(&task_pred);
                }
            }
        }

        Ok(predictions)
    }
}

impl MultiTaskSharedRepresentation<Trained> {
    /// Get shared representation weights
    pub fn shared_weights(&self) -> Result<&Array2<Float>> {
        self.shared_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "shared_weights".to_string(),
            })
    }

    /// Get task-specific weights
    pub fn task_weights(&self) -> Result<&Array3<Float>> {
        self.task_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "task_weights".to_string(),
            })
    }

    /// Get attention weights (if using attention strategy)
    pub fn attention_weights(&self) -> Option<&Array2<Float>> {
        self.attention_weights_.as_ref()
    }

    /// Compute shared representation for input data
    pub fn transform_to_shared(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let shared_weights =
            self.shared_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform_to_shared".to_string(),
                })?;
        let shared_bias = self
            .shared_bias_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_to_shared".to_string(),
            })?;
        Ok(self.compute_shared_features(x, shared_weights, shared_bias))
    }

    /// Get number of iterations taken during training
    pub fn n_iter(&self) -> Result<usize> {
        self.n_iter_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_iter".to_string(),
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_shared_representation() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],];

        let y = array![
            [2.0, 3.0], // Two related tasks
            [1.0, 1.5],
            [3.0, 4.5],
            [5.0, 6.5],
        ];

        let model = MultiTaskSharedRepresentation::builder()
            .strategy(SharedRepresentationStrategy::Linear)
            .shared_dim(2)
            .task_specific_dim(1)
            .learning_rate(0.01)
            .max_iter(100)
            .build();

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 2]);

        // Check that predictions are reasonable
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                assert!(predictions[[i, j]].is_finite());
            }
        }
    }

    #[test]
    fn test_factorized_representation() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let y = array![[1.0, 2.0], [0.5, 1.0], [1.5, 3.0],];

        let model = MultiTaskSharedRepresentation::builder()
            .strategy(SharedRepresentationStrategy::Factorized)
            .shared_dim(2)
            .max_iter(50)
            .build();

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[3, 2]);
    }

    #[test]
    fn test_attention_based_representation() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let y = array![[1.0, 2.0], [0.5, 1.0], [1.5, 3.0],];

        let model = MultiTaskSharedRepresentation::builder()
            .strategy(SharedRepresentationStrategy::Attention)
            .shared_dim(3)
            .learning_rate(0.01)
            .max_iter(50)
            .build();

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[3, 2]);

        // Should have attention weights
        assert!(fitted.attention_weights().is_some());
    }

    #[test]
    fn test_shared_representation_transform() {
        let x = array![[1.0, 0.0], [0.0, 1.0],];

        let y = array![[1.0, 2.0], [0.5, 1.0],];

        let model = MultiTaskSharedRepresentation::builder()
            .strategy(SharedRepresentationStrategy::Linear)
            .shared_dim(2)
            .max_iter(10)
            .build();

        let fitted = model.fit(&x, &y).unwrap();
        let shared_features = fitted.transform_to_shared(&x).unwrap();

        assert_eq!(shared_features.shape(), &[2, 2]);
    }

    #[test]
    fn test_builder_pattern() {
        let model = MultiTaskSharedRepresentation::builder()
            .strategy(SharedRepresentationStrategy::Hierarchical { n_levels: 3 })
            .shared_dim(10)
            .task_specific_dim(5)
            .regularization(0.01, 0.05)
            .learning_rate(0.001)
            .max_iter(500)
            .build();

        assert_eq!(model.config.shared_dim, 10);
        assert_eq!(model.config.task_specific_dim, 5);
        assert_eq!(model.config.shared_l2_reg, 0.01);
        assert_eq!(model.config.task_l2_reg, 0.05);
        assert_eq!(model.config.learning_rate, 0.001);
        assert_eq!(model.config.max_iter, 500);
    }
}
