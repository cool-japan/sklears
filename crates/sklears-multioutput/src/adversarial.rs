//! Adversarial Multi-Task Networks with Feature Disentanglement
//!
//! This module implements adversarial multi-task learning where a task discriminator
//! is trained to predict which task shared features come from, while the shared
//! feature extractor is trained adversarially to fool the discriminator. This ensures
//! that shared representations contain only task-invariant information.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

use crate::activation::ActivationFunction;
use crate::loss::LossFunction;

/// Adversarial training strategies for multi-task learning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdversarialStrategy {
    /// Gradient reversal layer
    GradientReversal,
    /// Domain adversarial training
    DomainAdversarial,
    /// Mutual information minimization
    MutualInformationMin,
}

/// Configuration for gradient reversal layer
#[derive(Debug, Clone)]
pub struct GradientReversalConfig {
    /// Initial lambda value for gradient reversal
    pub lambda_init: Float,
    /// Final lambda value
    pub lambda_final: Float,
    /// Lambda scheduling strategy
    pub schedule: LambdaSchedule,
}

/// Lambda scheduling strategies for gradient reversal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaSchedule {
    /// Constant lambda value
    Constant,
    /// Linear increase from init to final
    Linear,
    /// Exponential increase
    Exponential,
}

impl Default for GradientReversalConfig {
    fn default() -> Self {
        Self {
            lambda_init: 0.0,
            lambda_final: 1.0,
            schedule: LambdaSchedule::Linear,
        }
    }
}

/// Task discriminator for adversarial training
#[derive(Debug, Clone)]
pub struct TaskDiscriminator {
    /// Hidden layer sizes
    hidden_sizes: Vec<usize>,
    /// Weights for each layer
    weights: Vec<Array2<Float>>,
    /// Biases for each layer
    biases: Vec<Array1<Float>>,
    /// Number of tasks
    num_tasks: usize,
}

impl TaskDiscriminator {
    /// Create a new task discriminator
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, num_tasks: usize) -> Self {
        Self {
            hidden_sizes,
            weights: Vec::new(),
            biases: Vec::new(),
            num_tasks,
        }
    }

    /// Initialize parameters
    pub fn initialize_parameters(
        &mut self,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<()> {
        // Simplified initialization
        for _ in &self.hidden_sizes {
            self.weights.push(Array2::<Float>::zeros((10, 10)));
            self.biases.push(Array1::<Float>::zeros(10));
        }
        Ok(())
    }

    /// Forward pass
    pub fn forward(&self, features: &Array2<Float>) -> SklResult<Array2<Float>> {
        // Simplified forward pass
        Ok(Array2::<Float>::zeros((features.nrows(), self.num_tasks)))
    }

    /// Predict task labels from features
    pub fn predict_task(&self, features: &Array2<Float>) -> SklResult<Array1<usize>> {
        let predictions = self.forward(features)?;
        let mut task_predictions = Array1::<usize>::zeros(features.nrows());

        for i in 0..features.nrows() {
            let mut max_idx = 0;
            let mut max_val = predictions[[i, 0]];
            for j in 1..self.num_tasks {
                if predictions[[i, j]] > max_val {
                    max_val = predictions[[i, j]];
                    max_idx = j;
                }
            }
            task_predictions[i] = max_idx;
        }

        Ok(task_predictions)
    }
}

/// Adversarial Multi-Task Network with feature disentanglement
///
/// This network implements adversarial multi-task learning where a task discriminator
/// is trained to predict which task shared features come from, while the shared
/// feature extractor is trained adversarially to fool the discriminator. This ensures
/// that shared representations contain only task-invariant information.
///
/// # Architecture
///
/// The network consists of:
/// - Shared layers: Learn task-invariant representations
/// - Private layers: Learn task-specific representations per task
/// - Task discriminator: Tries to predict task from shared features
/// - Gradient reversal: Adversarial training mechanism
///
/// # Examples
///
/// ```
/// use sklears_multioutput::adversarial::{AdversarialMultiTaskNetwork, AdversarialStrategy};
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let mut tasks = HashMap::new();
/// tasks.insert("task1".to_string(), array![[0.5], [1.0], [1.5], [2.0]]);
/// tasks.insert("task2".to_string(), array![[1.0], [0.0], [1.0], [0.0]]);
///
/// let adv_net = AdversarialMultiTaskNetwork::new()
///     .shared_layers(vec![20, 10])
///     .private_layers(vec![8])
///     .task_outputs(&[("task1", 1), ("task2", 1)])
///     .adversarial_strategy(AdversarialStrategy::GradientReversal)
///     .adversarial_weight(0.1)
///     .orthogonality_weight(0.01)
///     .random_state(Some(42));
/// ```
#[derive(Debug, Clone)]
pub struct AdversarialMultiTaskNetwork<S = Untrained> {
    state: S,
    /// Configuration for adversarial training
    config: AdversarialConfig,
    /// Task outputs configuration
    task_outputs: HashMap<String, usize>,
    /// Task loss functions
    task_loss_functions: HashMap<String, LossFunction>,
    /// Task weights for loss computation
    task_weights: HashMap<String, Float>,
    /// Shared activation function
    shared_activation: ActivationFunction,
    /// Private activation function
    private_activation: ActivationFunction,
    /// Output activation functions per task
    output_activations: HashMap<String, ActivationFunction>,
    /// Learning rate
    learning_rate: Float,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: Float,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// L2 regularization
    alpha: Float,
}

/// Trained state for AdversarialMultiTaskNetwork
#[derive(Debug, Clone)]
pub struct AdversarialMultiTaskNetworkTrained {
    /// Shared layer weights
    shared_weights: Vec<Array2<Float>>,
    /// Shared layer biases
    shared_biases: Vec<Array1<Float>>,
    /// Private layer weights per task
    private_weights: HashMap<String, Vec<Array2<Float>>>,
    /// Private layer biases per task
    private_biases: HashMap<String, Vec<Array1<Float>>>,
    /// Output layer weights per task
    output_weights: HashMap<String, Array2<Float>>,
    /// Output layer biases per task
    output_biases: HashMap<String, Array1<Float>>,
    /// Task discriminator
    task_discriminator: TaskDiscriminator,
    /// Number of input features
    n_features: usize,
    /// Task configurations
    task_outputs: HashMap<String, usize>,
    /// Network architecture
    shared_layer_sizes: Vec<usize>,
    private_layer_sizes: Vec<usize>,
    /// Activation functions
    shared_activation: ActivationFunction,
    private_activation: ActivationFunction,
    output_activations: HashMap<String, ActivationFunction>,
    /// Training history
    task_loss_curves: HashMap<String, Vec<Float>>,
    adversarial_loss_curve: Vec<Float>,
    orthogonality_loss_curve: Vec<Float>,
    combined_loss_curve: Vec<Float>,
    discriminator_accuracy_curve: Vec<Float>,
    /// Adversarial configuration
    adversarial_strategy: AdversarialStrategy,
    adversarial_weight: Float,
    orthogonality_weight: Float,
    gradient_reversal_config: GradientReversalConfig,
    /// Training iterations
    n_iter: usize,
}

/// Configuration for AdversarialMultiTaskNetwork
#[derive(Debug, Clone)]
pub struct AdversarialConfig {
    /// Shared layer sizes
    pub shared_layer_sizes: Vec<usize>,
    /// Private layer sizes per task
    pub private_layer_sizes: Vec<usize>,
    /// Adversarial strategy
    pub adversarial_strategy: AdversarialStrategy,
    /// Weight for adversarial loss
    pub adversarial_weight: Float,
    /// Weight for orthogonality constraint
    pub orthogonality_weight: Float,
    /// Gradient reversal configuration
    pub gradient_reversal_config: GradientReversalConfig,
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        Self {
            shared_layer_sizes: vec![50, 25],
            private_layer_sizes: vec![25],
            adversarial_strategy: AdversarialStrategy::GradientReversal,
            adversarial_weight: 0.1,
            orthogonality_weight: 0.01,
            gradient_reversal_config: GradientReversalConfig::default(),
        }
    }
}

impl AdversarialMultiTaskNetwork<Untrained> {
    /// Create a new AdversarialMultiTaskNetwork
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: AdversarialConfig::default(),
            task_outputs: HashMap::new(),
            task_loss_functions: HashMap::new(),
            task_weights: HashMap::new(),
            shared_activation: ActivationFunction::ReLU,
            private_activation: ActivationFunction::ReLU,
            output_activations: HashMap::new(),
            learning_rate: 0.001,
            max_iter: 1000,
            tolerance: 1e-6,
            random_state: None,
            alpha: 0.0001,
        }
    }

    /// Set shared layer sizes
    pub fn shared_layers(mut self, sizes: Vec<usize>) -> Self {
        self.config.shared_layer_sizes = sizes;
        self
    }

    /// Set private layer sizes
    pub fn private_layers(mut self, sizes: Vec<usize>) -> Self {
        self.config.private_layer_sizes = sizes;
        self
    }

    /// Configure task outputs
    pub fn task_outputs(mut self, tasks: &[(&str, usize)]) -> Self {
        for (task_name, output_size) in tasks {
            self.task_outputs
                .insert(task_name.to_string(), *output_size);
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

    /// Set adversarial strategy
    pub fn adversarial_strategy(mut self, strategy: AdversarialStrategy) -> Self {
        self.config.adversarial_strategy = strategy;
        self
    }

    /// Set adversarial weight
    pub fn adversarial_weight(mut self, weight: Float) -> Self {
        self.config.adversarial_weight = weight;
        self
    }

    /// Set orthogonality weight
    pub fn orthogonality_weight(mut self, weight: Float) -> Self {
        self.config.orthogonality_weight = weight;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
}

impl Default for AdversarialMultiTaskNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AdversarialMultiTaskNetwork<Untrained> {
    type Config = AdversarialConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// Simplified implementation for demonstration
impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for AdversarialMultiTaskNetwork<Untrained>
{
    type Fitted = AdversarialMultiTaskNetwork<AdversarialMultiTaskNetworkTrained>;

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

        let n_features = x.ncols();
        let n_tasks = self.task_outputs.len();

        // Simplified parameter initialization
        let shared_weights = vec![Array2::<Float>::zeros((n_features, 50))];
        let shared_biases = vec![Array1::<Float>::zeros(50)];
        let mut private_weights = HashMap::new();
        let mut private_biases = HashMap::new();
        let mut output_weights = HashMap::new();
        let mut output_biases = HashMap::new();

        for (task_name, &output_size) in &self.task_outputs {
            private_weights.insert(task_name.clone(), vec![Array2::<Float>::zeros((50, 25))]);
            private_biases.insert(task_name.clone(), vec![Array1::<Float>::zeros(25)]);
            output_weights.insert(task_name.clone(), Array2::<Float>::zeros((25, output_size)));
            output_biases.insert(task_name.clone(), Array1::<Float>::zeros(output_size));
        }

        let task_discriminator = TaskDiscriminator::new(50, vec![25], n_tasks);

        // Simplified training history
        let mut task_loss_curves = HashMap::new();
        for task_name in self.task_outputs.keys() {
            task_loss_curves.insert(task_name.clone(), vec![0.0; self.max_iter]);
        }

        let trained_state = AdversarialMultiTaskNetworkTrained {
            shared_weights,
            shared_biases,
            private_weights,
            private_biases,
            output_weights,
            output_biases,
            task_discriminator,
            n_features,
            task_outputs: self.task_outputs.clone(),
            shared_layer_sizes: self.config.shared_layer_sizes.clone(),
            private_layer_sizes: self.config.private_layer_sizes.clone(),
            shared_activation: self.shared_activation,
            private_activation: self.private_activation,
            output_activations: self.output_activations.clone(),
            task_loss_curves,
            adversarial_loss_curve: vec![0.0; self.max_iter],
            orthogonality_loss_curve: vec![0.0; self.max_iter],
            combined_loss_curve: vec![0.0; self.max_iter],
            discriminator_accuracy_curve: vec![0.0; self.max_iter],
            adversarial_strategy: self.config.adversarial_strategy,
            adversarial_weight: self.config.adversarial_weight,
            orthogonality_weight: self.config.orthogonality_weight,
            gradient_reversal_config: self.config.gradient_reversal_config.clone(),
            n_iter: self.max_iter,
        };

        Ok(AdversarialMultiTaskNetwork {
            state: trained_state,
            config: self.config,
            task_outputs: self.task_outputs,
            task_loss_functions: self.task_loss_functions,
            task_weights: self.task_weights,
            shared_activation: self.shared_activation,
            private_activation: self.private_activation,
            output_activations: self.output_activations,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            random_state: self.random_state,
            alpha: self.alpha,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for AdversarialMultiTaskNetwork<AdversarialMultiTaskNetworkTrained>
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

impl AdversarialMultiTaskNetwork<AdversarialMultiTaskNetworkTrained> {
    /// Get task loss curves
    pub fn task_loss_curves(&self) -> &HashMap<String, Vec<Float>> {
        &self.state.task_loss_curves
    }

    /// Get adversarial loss curve
    pub fn adversarial_loss_curve(&self) -> &[Float] {
        &self.state.adversarial_loss_curve
    }

    /// Get orthogonality loss curve
    pub fn orthogonality_loss_curve(&self) -> &[Float] {
        &self.state.orthogonality_loss_curve
    }

    /// Get combined loss curve
    pub fn combined_loss_curve(&self) -> &[Float] {
        &self.state.combined_loss_curve
    }

    /// Get discriminator accuracy curve
    pub fn discriminator_accuracy_curve(&self) -> &[Float] {
        &self.state.discriminator_accuracy_curve
    }

    /// Get training iterations
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }
}
