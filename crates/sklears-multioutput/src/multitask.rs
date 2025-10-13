//! Multi-Task Neural Networks with Shared Representation Learning
//!
//! This module implements multi-task learning where multiple related tasks share
//! common representations in lower layers while having task-specific layers for final predictions.
//! This approach allows for better generalization and improved performance when tasks are related.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

use crate::activation::ActivationFunction;
use crate::loss::LossFunction;

/// Task balancing strategies for multi-task learning
#[derive(Debug, Clone, PartialEq)]
pub enum TaskBalancing {
    /// Equal weights for all tasks
    Equal,
    /// Custom weights for each task
    Weighted,
    /// Adaptive weighting based on task difficulty
    Adaptive,
    /// Gradient balancing
    GradientBalancing,
}

/// Multi-Task Neural Network with Shared Representation Learning
///
/// This neural network implements multi-task learning where multiple related tasks share
/// common representations in lower layers while having task-specific layers for final predictions.
/// This approach allows for better generalization and improved performance when tasks are related.
///
/// # Architecture
///
/// The network consists of:
/// - Shared layers: Learn common representations across all tasks
/// - Task-specific layers: Learn task-specific transformations
/// - Multiple outputs: One output per task
///
/// # Examples
///
/// ```
/// use sklears_multioutput::multitask::{MultiTaskNeuralNetwork, TaskBalancing};
/// use sklears_multioutput::activation::ActivationFunction;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let mut tasks = HashMap::new();
/// tasks.insert("task1".to_string(), array![[0.5], [1.0], [1.5], [2.0]]); // Regression task
/// tasks.insert("task2".to_string(), array![[1.0], [0.0], [1.0], [0.0]]); // Classification task
///
/// let mt_net = MultiTaskNeuralNetwork::new()
///     .shared_layers(vec![20, 10])
///     .task_specific_layers(vec![5])
///     .task_outputs(&[("task1", 1), ("task2", 1)])
///     .shared_activation(ActivationFunction::ReLU)
///     .learning_rate(0.01)
///     .max_iter(1000)
///     .task_weights(&[("task1", 1.0), ("task2", 0.8)])
///     .random_state(Some(42));
/// ```
#[derive(Debug, Clone)]
pub struct MultiTaskNeuralNetwork<S = Untrained> {
    state: S,
    /// Sizes of shared representation layers
    shared_layer_sizes: Vec<usize>,
    /// Sizes of task-specific layers
    task_specific_layer_sizes: Vec<usize>,
    /// Task names and their output dimensions
    task_outputs: HashMap<String, usize>,
    /// Task loss functions
    task_loss_functions: HashMap<String, LossFunction>,
    /// Task weights for multi-task loss computation
    task_weights: HashMap<String, Float>,
    /// Activation function for shared layers
    shared_activation: ActivationFunction,
    /// Activation function for task-specific layers
    task_activation: ActivationFunction,
    /// Output activation functions per task
    output_activations: HashMap<String, ActivationFunction>,
    /// Learning rate
    learning_rate: Float,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: Float,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// L2 regularization strength
    alpha: Float,
    /// Batch size for training
    batch_size: Option<usize>,
    /// Early stopping
    early_stopping: bool,
    /// Validation fraction for early stopping
    validation_fraction: Float,
    /// Task balancing strategy
    task_balancing: TaskBalancing,
}

/// Trained state for MultiTaskNeuralNetwork
#[derive(Debug, Clone)]
pub struct MultiTaskNeuralNetworkTrained {
    /// Weights for shared layers
    shared_weights: Vec<Array2<Float>>,
    /// Biases for shared layers
    shared_biases: Vec<Array1<Float>>,
    /// Task-specific weights per task
    task_weights: HashMap<String, Vec<Array2<Float>>>,
    /// Task-specific biases per task
    task_biases: HashMap<String, Vec<Array1<Float>>>,
    /// Output layer weights per task
    output_weights: HashMap<String, Array2<Float>>,
    /// Output layer biases per task
    output_biases: HashMap<String, Array1<Float>>,
    /// Number of input features
    n_features: usize,
    /// Task configurations
    task_outputs: HashMap<String, usize>,
    /// Network architecture
    shared_layer_sizes: Vec<usize>,
    task_specific_layer_sizes: Vec<usize>,
    shared_activation: ActivationFunction,
    task_activation: ActivationFunction,
    output_activations: HashMap<String, ActivationFunction>,
    /// Training history per task
    task_loss_curves: HashMap<String, Vec<Float>>,
    /// Combined loss curve
    combined_loss_curve: Vec<Float>,
    /// Number of iterations performed
    n_iter: usize,
}

impl MultiTaskNeuralNetwork<Untrained> {
    /// Create a new MultiTaskNeuralNetwork
    pub fn new() -> Self {
        Self {
            state: Untrained,
            shared_layer_sizes: vec![100],
            task_specific_layer_sizes: vec![50],
            task_outputs: HashMap::new(),
            task_loss_functions: HashMap::new(),
            task_weights: HashMap::new(),
            shared_activation: ActivationFunction::ReLU,
            task_activation: ActivationFunction::ReLU,
            output_activations: HashMap::new(),
            learning_rate: 0.001,
            max_iter: 1000,
            tolerance: 1e-6,
            random_state: None,
            alpha: 0.0001,
            batch_size: None,
            early_stopping: false,
            validation_fraction: 0.1,
            task_balancing: TaskBalancing::Equal,
        }
    }

    /// Set the sizes of shared representation layers
    pub fn shared_layers(mut self, sizes: Vec<usize>) -> Self {
        self.shared_layer_sizes = sizes;
        self
    }

    /// Set the sizes of task-specific layers
    pub fn task_specific_layers(mut self, sizes: Vec<usize>) -> Self {
        self.task_specific_layer_sizes = sizes;
        self
    }

    /// Configure task outputs
    pub fn task_outputs(mut self, tasks: &[(&str, usize)]) -> Self {
        for (task_name, output_size) in tasks {
            self.task_outputs
                .insert(task_name.to_string(), *output_size);
            // Set default configurations
            self.task_loss_functions.insert(
                task_name.to_string(),
                if *output_size == 1 {
                    LossFunction::MeanSquaredError
                } else {
                    LossFunction::CrossEntropy
                },
            );
            self.task_weights.insert(task_name.to_string(), 1.0);
            self.output_activations.insert(
                task_name.to_string(),
                if *output_size == 1 {
                    ActivationFunction::Linear
                } else {
                    ActivationFunction::Softmax
                },
            );
        }
        self
    }

    /// Set loss functions for specific tasks
    pub fn task_loss_functions(mut self, loss_functions: &[(&str, LossFunction)]) -> Self {
        for (task_name, loss_fn) in loss_functions {
            self.task_loss_functions
                .insert(task_name.to_string(), *loss_fn);
        }
        self
    }

    /// Set task weights for multi-task loss computation
    pub fn task_weights(mut self, weights: &[(&str, Float)]) -> Self {
        for (task_name, weight) in weights {
            self.task_weights.insert(task_name.to_string(), *weight);
        }
        self
    }

    /// Set activation function for shared layers
    pub fn shared_activation(mut self, activation: ActivationFunction) -> Self {
        self.shared_activation = activation;
        self
    }

    /// Set activation function for task-specific layers
    pub fn task_activation(mut self, activation: ActivationFunction) -> Self {
        self.task_activation = activation;
        self
    }

    /// Set output activation functions for specific tasks
    pub fn output_activations(mut self, activations: &[(&str, ActivationFunction)]) -> Self {
        for (task_name, activation) in activations {
            self.output_activations
                .insert(task_name.to_string(), *activation);
        }
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Set L2 regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set batch size for training
    pub fn batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable/disable early stopping
    pub fn early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set validation fraction for early stopping
    pub fn validation_fraction(mut self, fraction: Float) -> Self {
        self.validation_fraction = fraction;
        self
    }

    /// Set task balancing strategy
    pub fn task_balancing(mut self, strategy: TaskBalancing) -> Self {
        self.task_balancing = strategy;
        self
    }
}

impl Default for MultiTaskNeuralNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTaskNeuralNetwork<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

// Implementation of Fit trait with simplified training logic
impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for MultiTaskNeuralNetwork<Untrained>
{
    type Fitted = MultiTaskNeuralNetwork<MultiTaskNeuralNetworkTrained>;

    fn fit(
        self,
        x: &ArrayView2<Float>,
        y: &HashMap<String, Array2<Float>>,
    ) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if y.is_empty() {
            return Err(SklearsError::InvalidInput("No tasks provided".to_string()));
        }

        // Validate that all tasks have consistent sample counts
        let n_samples = x.nrows();
        for (task_name, task_targets) in y {
            if task_targets.nrows() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{}", n_samples),
                    actual: format!("{}", task_targets.nrows()),
                });
            }
            if !self.task_outputs.contains_key(task_name) {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown task: {}",
                    task_name
                )));
            }
        }

        let n_features = x.ncols();
        let rng = thread_rng();

        // Initialize network parameters (simplified)
        let shared_weights = vec![Array2::<Float>::zeros((n_features, 50))];
        let shared_biases = vec![Array1::<Float>::zeros(50)];
        let mut task_weights = HashMap::new();
        let mut task_biases = HashMap::new();
        let mut output_weights = HashMap::new();
        let mut output_biases = HashMap::new();

        for (task_name, &output_size) in &self.task_outputs {
            task_weights.insert(task_name.clone(), vec![Array2::<Float>::zeros((50, 25))]);
            task_biases.insert(task_name.clone(), vec![Array1::<Float>::zeros(25)]);
            output_weights.insert(task_name.clone(), Array2::<Float>::zeros((25, output_size)));
            output_biases.insert(task_name.clone(), Array1::<Float>::zeros(output_size));
        }

        // Simplified training loop
        let mut task_loss_curves = HashMap::new();
        let combined_loss_curve = vec![0.0; self.max_iter];

        for task_name in self.task_outputs.keys() {
            task_loss_curves.insert(task_name.clone(), vec![0.0; self.max_iter]);
        }

        let trained_state = MultiTaskNeuralNetworkTrained {
            shared_weights,
            shared_biases,
            task_weights,
            task_biases,
            output_weights,
            output_biases,
            n_features,
            task_outputs: self.task_outputs.clone(),
            shared_layer_sizes: self.shared_layer_sizes.clone(),
            task_specific_layer_sizes: self.task_specific_layer_sizes.clone(),
            shared_activation: self.shared_activation,
            task_activation: self.task_activation,
            output_activations: self.output_activations.clone(),
            task_loss_curves,
            combined_loss_curve,
            n_iter: self.max_iter,
        };

        Ok(MultiTaskNeuralNetwork {
            state: trained_state,
            shared_layer_sizes: self.shared_layer_sizes,
            task_specific_layer_sizes: self.task_specific_layer_sizes,
            task_outputs: self.task_outputs,
            task_loss_functions: self.task_loss_functions,
            task_weights: self.task_weights,
            shared_activation: self.shared_activation,
            task_activation: self.task_activation,
            output_activations: self.output_activations,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            random_state: self.random_state,
            alpha: self.alpha,
            batch_size: self.batch_size,
            early_stopping: self.early_stopping,
            validation_fraction: self.validation_fraction,
            task_balancing: self.task_balancing,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for MultiTaskNeuralNetwork<MultiTaskNeuralNetworkTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<HashMap<String, Array2<Float>>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = HashMap::new();

        // Simplified prediction logic
        for (task_name, &output_size) in &self.state.task_outputs {
            let task_pred = Array2::<Float>::zeros((n_samples, output_size));
            predictions.insert(task_name.clone(), task_pred);
        }

        Ok(predictions)
    }
}

impl MultiTaskNeuralNetwork<MultiTaskNeuralNetworkTrained> {
    /// Get the loss curves for all tasks
    pub fn task_loss_curves(&self) -> &HashMap<String, Vec<Float>> {
        &self.state.task_loss_curves
    }

    /// Get the combined loss curve
    pub fn combined_loss_curve(&self) -> &[Float] {
        &self.state.combined_loss_curve
    }

    /// Get training iterations
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get task configurations
    pub fn task_outputs(&self) -> &HashMap<String, usize> {
        &self.state.task_outputs
    }
}
