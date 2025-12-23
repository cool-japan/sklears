//! Meta-Learning for Multi-Task Learning
//!
//! This method learns meta-parameters that can quickly adapt to new tasks.
//! It uses a model-agnostic meta-learning (MAML) approach adapted for multi-task scenarios.

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

/// Meta-Learning for Multi-Task Learning
///
/// This method learns meta-parameters that can quickly adapt to new tasks.
/// It uses a model-agnostic meta-learning (MAML) approach adapted for multi-task scenarios.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::regularization::MetaLearningMultiTask;
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
/// let meta_learning = MetaLearningMultiTask::new()
///     .meta_learning_rate(0.01)
///     .inner_learning_rate(0.1)
///     .n_inner_steps(5)
///     .max_iter(1000);
/// ```
#[derive(Debug, Clone)]
pub struct MetaLearningMultiTask<S = Untrained> {
    pub(crate) state: S,
    /// Meta-learning rate for updating meta-parameters
    pub(crate) meta_learning_rate: Float,
    /// Inner learning rate for task-specific adaptation
    pub(crate) inner_learning_rate: Float,
    /// Number of inner gradient steps per task
    pub(crate) n_inner_steps: usize,
    /// Maximum meta-iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance
    pub(crate) tolerance: Float,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Include intercept term
    pub(crate) fit_intercept: bool,
    /// Random state for reproducible meta-learning
    pub(crate) random_state: Option<u64>,
}

/// Trained state for MetaLearningMultiTask
#[derive(Debug, Clone)]
pub struct MetaLearningMultiTaskTrained {
    /// Meta-parameters (initialization for new tasks)
    pub(crate) meta_parameters: Array2<Float>,
    /// Meta-intercepts
    pub(crate) meta_intercepts: Array1<Float>,
    /// Task-specific adapted parameters
    pub(crate) task_parameters: HashMap<String, Array2<Float>>,
    /// Task-specific adapted intercepts
    pub(crate) task_intercepts: HashMap<String, Array1<Float>>,
    /// Number of input features
    pub(crate) n_features: usize,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Training parameters
    pub(crate) meta_learning_rate: Float,
    pub(crate) inner_learning_rate: Float,
    pub(crate) n_inner_steps: usize,
    /// Training iterations performed
    pub(crate) n_iter: usize,
}

impl MetaLearningMultiTask<Untrained> {
    /// Create a new MetaLearningMultiTask instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            meta_learning_rate: 0.01,
            inner_learning_rate: 0.1,
            n_inner_steps: 5,
            max_iter: 1000,
            tolerance: 1e-4,
            task_outputs: HashMap::new(),
            fit_intercept: true,
            random_state: None,
        }
    }

    /// Set meta-learning rate
    pub fn meta_learning_rate(mut self, lr: Float) -> Self {
        self.meta_learning_rate = lr;
        self
    }

    /// Set inner learning rate
    pub fn inner_learning_rate(mut self, lr: Float) -> Self {
        self.inner_learning_rate = lr;
        self
    }

    /// Set number of inner gradient steps
    pub fn n_inner_steps(mut self, steps: usize) -> Self {
        self.n_inner_steps = steps;
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

    /// Set random state
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

impl Default for MetaLearningMultiTask<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MetaLearningMultiTask<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for MetaLearningMultiTask<Untrained>
{
    type Fitted = MetaLearningMultiTask<MetaLearningMultiTaskTrained>;

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

        // Initialize meta-parameters
        let mut rng_gen = thread_rng();

        // Use first task to determine output size for meta-parameters
        let first_task_outputs = y.values().next().unwrap().ncols();
        let mut meta_parameters = Array2::<Float>::zeros((n_features, first_task_outputs));
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..n_features {
            for j in 0..first_task_outputs {
                meta_parameters[[i, j]] = rng_gen.sample(normal_dist);
            }
        }
        let mut meta_intercepts = Array1::<Float>::zeros(first_task_outputs);

        let task_names: Vec<String> = y.keys().cloned().collect();
        let mut task_parameters: HashMap<String, Array2<Float>> = HashMap::new();
        let mut task_intercepts: HashMap<String, Array1<Float>> = HashMap::new();

        // Meta-learning loop
        let mut prev_loss = Float::INFINITY;
        let mut n_iter = 0;

        for iteration in 0..self.max_iter {
            let mut total_meta_loss = 0.0;
            let mut meta_grad_sum: Array2<Float> = Array2::<Float>::zeros(meta_parameters.dim());
            let mut meta_intercept_grad_sum: Array1<Float> =
                Array1::<Float>::zeros(meta_intercepts.len());

            // For each task, perform inner loop adaptation
            for (task_name, y_task) in y {
                // Initialize task parameters from meta-parameters
                let mut task_params = meta_parameters.clone();
                let mut task_intercept = meta_intercepts.clone();

                // Inner loop: adapt to specific task
                for _inner_step in 0..self.n_inner_steps {
                    // Compute predictions
                    let predictions = x.dot(&task_params);
                    let predictions_with_intercept = &predictions + &task_intercept;

                    // Compute residuals
                    let residuals = &predictions_with_intercept - y_task;

                    // Compute gradients
                    let grad_params = x.t().dot(&residuals) / (n_samples as Float);
                    let grad_intercept = residuals.sum_axis(Axis(0)) / (n_samples as Float);

                    // Update task-specific parameters
                    task_params -= &(&grad_params * self.inner_learning_rate);
                    task_intercept -= &(&grad_intercept * self.inner_learning_rate);
                }

                // Compute final loss for this task
                let final_predictions = x.dot(&task_params);
                let final_predictions_with_intercept = &final_predictions + &task_intercept;
                let final_residuals = &final_predictions_with_intercept - y_task;
                let task_loss = final_residuals.mapv(|x| x * x).sum();
                total_meta_loss += task_loss;

                // Compute meta-gradients (how changes in meta-parameters affect final loss)
                let meta_grad_params = x.t().dot(&final_residuals) / (n_samples as Float);
                let meta_grad_intercept = final_residuals.sum_axis(Axis(0)) / (n_samples as Float);

                meta_grad_sum = meta_grad_sum + meta_grad_params;
                meta_intercept_grad_sum = meta_intercept_grad_sum + meta_grad_intercept;

                // Store adapted parameters
                task_parameters.insert(task_name.clone(), task_params);
                task_intercepts.insert(task_name.clone(), task_intercept);
            }

            // Update meta-parameters
            let n_tasks = y.len() as Float;
            meta_parameters -= &(&(meta_grad_sum / n_tasks) * self.meta_learning_rate);
            meta_intercepts -= &(&(meta_intercept_grad_sum / n_tasks) * self.meta_learning_rate);

            // Check convergence
            if (prev_loss - total_meta_loss).abs() < self.tolerance {
                n_iter = iteration + 1;
                break;
            }
            prev_loss = total_meta_loss;
            n_iter = iteration + 1;
        }

        Ok(MetaLearningMultiTask {
            state: MetaLearningMultiTaskTrained {
                meta_parameters,
                meta_intercepts,
                task_parameters,
                task_intercepts,
                n_features,
                task_outputs: self.task_outputs.clone(),
                meta_learning_rate: self.meta_learning_rate,
                inner_learning_rate: self.inner_learning_rate,
                n_inner_steps: self.n_inner_steps,
                n_iter,
            },
            meta_learning_rate: self.meta_learning_rate,
            inner_learning_rate: self.inner_learning_rate,
            n_inner_steps: self.n_inner_steps,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            task_outputs: self.task_outputs,
            fit_intercept: self.fit_intercept,
            random_state: self.random_state,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for MetaLearningMultiTask<MetaLearningMultiTaskTrained>
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

        for (task_name, coef) in &self.state.task_parameters {
            let task_predictions = x.dot(coef);
            let intercept = &self.state.task_intercepts[task_name];
            let final_predictions = &task_predictions + intercept;
            predictions.insert(task_name.clone(), final_predictions);
        }

        Ok(predictions)
    }
}

impl MetaLearningMultiTask<MetaLearningMultiTaskTrained> {
    /// Adapt meta-parameters to a new task with few examples
    pub fn adapt_to_new_task(
        &self,
        X: &ArrayView2<Float>,
        y: &Array2<Float>,
        n_adaptation_steps: usize,
    ) -> SklResult<(Array2<Float>, Array1<Float>)> {
        let x = X.to_owned();
        let (n_samples, n_features) = x.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        // Start with meta-parameters
        let mut adapted_params = self.state.meta_parameters.clone();
        let mut adapted_intercept = self.state.meta_intercepts.clone();

        // Perform adaptation steps
        for _step in 0..n_adaptation_steps {
            // Compute predictions
            let predictions = x.dot(&adapted_params);
            let predictions_with_intercept = &predictions + &adapted_intercept;

            // Compute residuals
            let residuals = &predictions_with_intercept - y;

            // Compute gradients
            let grad_params = x.t().dot(&residuals) / (n_samples as Float);
            let grad_intercept = residuals.sum_axis(Axis(0)) / (n_samples as Float);

            // Update parameters
            adapted_params -= &(&grad_params * self.state.inner_learning_rate);
            adapted_intercept -= &(&grad_intercept * self.state.inner_learning_rate);
        }

        Ok((adapted_params, adapted_intercept))
    }

    /// Get meta-parameters for initialization of new tasks
    pub fn get_meta_parameters(&self) -> (&Array2<Float>, &Array1<Float>) {
        (&self.state.meta_parameters, &self.state.meta_intercepts)
    }
}

impl MetaLearningMultiTaskTrained {
    /// Get meta-parameters
    pub fn meta_parameters(&self) -> &Array2<Float> {
        &self.meta_parameters
    }

    /// Get meta-intercepts
    pub fn meta_intercepts(&self) -> &Array1<Float> {
        &self.meta_intercepts
    }

    /// Get task-specific parameters
    pub fn task_parameters(&self, task_name: &str) -> Option<&Array2<Float>> {
        self.task_parameters.get(task_name)
    }

    /// Get task-specific intercepts
    pub fn task_intercepts(&self, task_name: &str) -> Option<&Array1<Float>> {
        self.task_intercepts.get(task_name)
    }

    /// Get number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Get meta-learning parameters
    pub fn meta_learning_config(&self) -> (Float, Float, usize) {
        (
            self.meta_learning_rate,
            self.inner_learning_rate,
            self.n_inner_steps,
        )
    }
}
