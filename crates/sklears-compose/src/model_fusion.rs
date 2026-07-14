//! Model Fusion for Ensemble Learning
//!
//! This module provides advanced fusion strategies for combining predictions from
//! multiple models. Model fusion goes beyond simple voting by learning optimal
//! combination strategies through various approaches including linear combinations
//! and small neural networks.
//!
//! # Fusion Strategies
//!
//! ## Linear Fusion
//! - **Weighted Linear Combination**: weights are learned by solving the ordinary
//!   least-squares (or ridge-regularized) problem `argmin_w ||P w - y||^2`, where
//!   `P` collects every base model's predictions as one column each.
//!
//! ## Neural Fusion
//! - **Neural Network Fusion**: a small multi-layer perceptron (architecture given
//!   by `hidden_layers`) is trained with real forward/backward-propagation gradient
//!   descent on the base models' predictions to minimize mean squared error against
//!   the true targets.
//!
//! Other strategies declared on [`FusionStrategy`] (gating networks, attention
//! fusion, Bayesian fusion, meta-learning fusion, adversarial fusion, adaptive
//! linear fusion, deep fusion) describe a much larger design space that is not yet
//! implemented with real training logic; attempting to `fit` a [`ModelFusion`] with
//! one of those strategies returns a clear [`SklearsError::NotImplemented`] error
//! rather than silently fabricating a trained model.
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble::{ModelFusion, FusionStrategy};
//!
//! // Linear fusion with learned weights
//! let linear_fusion = ModelFusion::builder()
//!     .base_model("cnn", Box::new(cnn_model))
//!     .base_model("rnn", Box::new(rnn_model))
//!     .fusion_strategy(FusionStrategy::WeightedLinear {
//!         regularization: Some(0.01)
//!     })
//!     .build();
//!
//! // Neural network fusion
//! let nn_fusion = ModelFusion::builder()
//!     .base_model("model1", Box::new(model1))
//!     .base_model("model2", Box::new(model2))
//!     .fusion_strategy(FusionStrategy::NeuralNetwork {
//!         hidden_layers: vec![8, 4],
//!         activation: "tanh".to_string(),
//!         dropout: None,
//!     })
//!     .build();
//! ```

use scirs2_core::linalg::{lstsq_ndarray, solve_ndarray};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, SeedableRng};
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::{Estimator, Fit, Untrained},
    types::Float,
};
use std::collections::HashMap;

use crate::PipelinePredictor;

/// Fusion strategies for combining model predictions
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// Simple weighted linear combination, fitted via (optionally ridge-regularized)
    /// least squares.
    WeightedLinear {
        /// Ridge (L2) regularization strength applied to the normal equations.
        /// `None` or `Some(0.0)` performs plain ordinary least squares.
        regularization: Option<Float>,
    },
    /// Adaptive linear fusion with input-dependent weights
    AdaptiveLinear {
        /// Number of context features for adaptation
        context_features: usize,
        /// Learning rate for adaptation
        learning_rate: Float,
    },
    /// Neural network-based fusion, trained via real gradient descent.
    NeuralNetwork {
        /// Hidden layer sizes
        hidden_layers: Vec<usize>,
        /// Activation function: one of `"relu"`, `"sigmoid"`, `"tanh"`; any other
        /// value falls back to the identity activation.
        activation: String,
        /// Dropout probability (not yet used by the training loop)
        dropout: Option<Float>,
    },
    /// Deep fusion with multiple hidden layers
    DeepFusion {
        /// Network architecture
        architecture: Vec<usize>,
        /// Batch normalization
        batch_norm: bool,
        /// Residual connections
        residual: bool,
    },
    /// Gating network for mixture of experts
    GatingNetwork {
        /// Type of gating mechanism
        gating_type: GatingType,
        /// Temperature for softmax gating
        temperature: Float,
    },
    /// Attention-based fusion
    AttentionFusion {
        /// Attention head count
        num_heads: usize,
        /// Key/query/value dimensions
        head_dim: usize,
        /// Enable multi-head attention
        multi_head: bool,
    },
    /// Bayesian fusion with uncertainty
    BayesianFusion {
        /// Prior distribution parameters
        prior_alpha: Float,
        /// Posterior update rate
        posterior_beta: Float,
    },
    /// Meta-learning fusion
    MetaLearning {
        /// Meta-learner type
        meta_learner: String,
        /// Support set size for few-shot learning
        support_size: usize,
    },
    /// Adversarial robust fusion
    AdversarialFusion {
        /// Adversarial training strength
        adversarial_weight: Float,
        /// Attack method for training
        attack_method: String,
    },
}

/// Types of gating mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum GatingType {
    /// Mixture of experts gating
    MixtureOfExperts,
    /// Hierarchical gating
    Hierarchical,
    /// Input-dependent gating
    InputDependent,
    /// Probabilistic gating with uncertainty
    Probabilistic,
    /// Attention-based gating
    AttentionBased,
}

/// Regularization types for fusion
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso)
    L1,
    /// L2 regularization (Ridge)
    L2,
    /// Elastic net (L1 + L2)
    ElasticNet {
        /// Mixing ratio between L1 and L2 (0 = pure L2, 1 = pure L1).
        l1_ratio: Float,
    },
    /// Group Lasso for structured sparsity
    GroupLasso,
    /// Nuclear norm for low-rank solutions
    Nuclear,
}

/// Model fusion for advanced ensemble combination
///
/// # Type Parameters
///
/// * `S` - State type ([`Untrained`] or [`ModelFusionTrained`])
#[derive(Debug)]
pub struct ModelFusion<S = Untrained> {
    state: S,
    /// Named base models for fusion
    base_models: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Fusion strategy
    fusion_strategy: FusionStrategy,
    /// Regularization type and strength (currently recorded but not consulted by
    /// any implemented strategy; `FusionStrategy::WeightedLinear`'s own
    /// `regularization` field is what actually drives ridge fitting).
    regularization: Option<(RegularizationType, Float)>,
    /// Enable cross-validation for hyperparameter tuning (reserved for future use)
    enable_cv: bool,
    /// Number of CV folds (reserved for future use)
    cv_folds: usize,
    /// Feature scaling for inputs (reserved for future use)
    scale_features: bool,
    /// Temperature for probability calibration (reserved for future use)
    calibration_temperature: Option<Float>,
    /// Enable uncertainty estimation (reserved for future use)
    uncertainty_estimation: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs (reserved for future use)
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
}

/// Trained state for [`ModelFusion`], produced by [`Fit::fit`].
pub struct ModelFusionTrained {
    /// Base models, each genuinely fitted on the training data during `fit`.
    fitted_base_models: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Fusion parameters learned from the base models' predictions.
    fusion_params: FusionParameters,
    /// Number of input features seen during fitting.
    n_features_in: usize,
    /// Feature names, if provided.
    feature_names_in: Option<Vec<String>>,
}

impl std::fmt::Debug for ModelFusionTrained {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelFusionTrained")
            .field("n_base_models", &self.fitted_base_models.len())
            .field("fusion_params", &self.fusion_params)
            .field("n_features_in", &self.n_features_in)
            .field("feature_names_in", &self.feature_names_in)
            .finish()
    }
}

/// Fusion weights and parameters learned during fitting
#[derive(Debug, Clone)]
pub struct FusionParameters {
    /// Linear combination weights (populated by [`FusionStrategy::WeightedLinear`])
    pub linear_weights: Option<Array1<Float>>,
    /// Neural network weight matrices, one per layer (populated by
    /// [`FusionStrategy::NeuralNetwork`])
    pub neural_weights: Option<Vec<Array2<Float>>>,
    /// Neural network biases, one vector per layer
    pub neural_biases: Option<Vec<Array1<Float>>>,
    /// Gating network parameters (reserved; not currently learned)
    pub gating_weights: Option<Array2<Float>>,
    /// Attention weights (reserved; not currently learned)
    pub attention_weights: Option<Array2<Float>>,
    /// Regularization penalty actually applied
    pub regularization_penalty: Float,
}

/// Prediction result with uncertainty information
#[derive(Debug, Clone)]
pub struct FusionPrediction {
    /// Final fused prediction
    pub prediction: Array1<Float>,
    /// Individual model predictions
    pub individual_predictions: Vec<Array1<Float>>,
    /// Fusion weights used (for `NeuralNetwork` fusion this is a real, but
    /// approximate, per-model importance proxy derived from the magnitude of the
    /// first layer's weights, not a literal linear combination weight)
    pub fusion_weights: Array1<Float>,
    /// Prediction uncertainty (if available)
    pub uncertainty: Option<Array1<Float>>,
    /// Confidence scores
    pub confidence: Array1<Float>,
}

/// Performance metrics for fusion evaluation
#[derive(Debug, Clone)]
pub struct FusionMetrics {
    /// Mean squared error
    pub mse: Float,
    /// Mean absolute error
    pub mae: Float,
    /// R² score
    pub r2: Float,
    /// Individual model contributions
    pub model_contributions: HashMap<String, Float>,
    /// Fusion complexity (parameter count)
    pub complexity: usize,
}

impl ModelFusion<Untrained> {
    /// Create a new model fusion builder
    #[must_use]
    pub fn builder() -> ModelFusionBuilder {
        ModelFusionBuilder::new()
    }

    /// Create a new model fusion with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            base_models: Vec::new(),
            fusion_strategy: FusionStrategy::WeightedLinear {
                regularization: None,
            },
            regularization: None,
            enable_cv: false,
            cv_folds: 5,
            scale_features: true,
            calibration_temperature: None,
            uncertainty_estimation: false,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add a base model to the fusion ensemble
    #[must_use]
    pub fn add_base_model(mut self, name: &str, model: Box<dyn PipelinePredictor>) -> Self {
        self.base_models.push((name.to_string(), model));
        self
    }

    /// Set fusion strategy
    #[must_use]
    pub fn set_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set regularization
    #[must_use]
    pub fn set_regularization(mut self, reg_type: RegularizationType, strength: Float) -> Self {
        self.regularization = Some((reg_type, strength));
        self
    }
}

impl Default for ModelFusion<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> ModelFusion<S> {
    /// Get base model names
    #[must_use]
    pub fn base_model_names(&self) -> Vec<&str> {
        self.base_models
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get number of base models
    #[must_use]
    pub fn n_base_models(&self) -> usize {
        self.base_models.len()
    }

    /// Get fusion strategy
    #[must_use]
    pub fn fusion_strategy(&self) -> &FusionStrategy {
        &self.fusion_strategy
    }

    /// Validate configuration
    fn validate_configuration(&self) -> SklResult<()> {
        if self.base_models.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "base_models".to_string(),
                reason: "ModelFusion requires at least one base model".to_string(),
            });
        }

        if self.cv_folds < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "cv_folds".to_string(),
                reason: "CV folds must be at least 2".to_string(),
            });
        }

        match &self.fusion_strategy {
            FusionStrategy::NeuralNetwork { hidden_layers, .. } if hidden_layers.is_empty() => {
                return Err(SklearsError::InvalidParameter {
                    name: "fusion_strategy.hidden_layers".to_string(),
                    reason: "neural network fusion requires at least one hidden layer".to_string(),
                });
            }
            FusionStrategy::AttentionFusion {
                num_heads,
                head_dim,
                ..
            } if *num_heads == 0 || *head_dim == 0 => {
                return Err(SklearsError::InvalidParameter {
                    name: "fusion_strategy.num_heads/head_dim".to_string(),
                    reason: "attention fusion requires positive num_heads and head_dim".to_string(),
                });
            }
            _ => {}
        }

        Ok(())
    }

    /// Initialize fusion parameters for the configured strategy. Exposed mainly
    /// for introspection/testing; `fit` computes the real, data-dependent
    /// parameters via an internal learning routine instead of using this initial
    /// guess.
    #[must_use]
    pub fn initialize_parameters(&self, input_dim: usize) -> FusionParameters {
        compute_initial_parameters(self.base_models.len(), &self.fusion_strategy, input_dim)
    }
}

/// Compute a strategy-appropriate initial (untrained) [`FusionParameters`] value.
fn compute_initial_parameters(
    n_models: usize,
    strategy: &FusionStrategy,
    input_dim: usize,
) -> FusionParameters {
    match strategy {
        FusionStrategy::NeuralNetwork { hidden_layers, .. } if !hidden_layers.is_empty() => {
            let mut weights = Vec::new();
            let mut biases = Vec::new();

            let input_size = n_models;
            weights.push(Array2::zeros((input_size, hidden_layers[0])));
            biases.push(Array1::zeros(hidden_layers[0]));

            for i in 1..hidden_layers.len() {
                weights.push(Array2::zeros((hidden_layers[i - 1], hidden_layers[i])));
                biases.push(Array1::zeros(hidden_layers[i]));
            }

            let last_hidden = hidden_layers.last().copied().unwrap_or(0);
            weights.push(Array2::zeros((last_hidden, 1)));
            biases.push(Array1::zeros(1));

            FusionParameters {
                linear_weights: None,
                neural_weights: Some(weights),
                neural_biases: Some(biases),
                gating_weights: None,
                attention_weights: None,
                regularization_penalty: 0.0,
            }
        }
        FusionStrategy::GatingNetwork { .. } => FusionParameters {
            linear_weights: None,
            neural_weights: None,
            neural_biases: None,
            gating_weights: Some(Array2::zeros((input_dim, n_models))),
            attention_weights: None,
            regularization_penalty: 0.0,
        },
        FusionStrategy::AttentionFusion {
            num_heads,
            head_dim,
            ..
        } => {
            let attention_dim = num_heads * head_dim;
            FusionParameters {
                linear_weights: None,
                neural_weights: None,
                neural_biases: None,
                gating_weights: None,
                attention_weights: Some(Array2::zeros((n_models, attention_dim))),
                regularization_penalty: 0.0,
            }
        }
        _ => {
            let uniform = if n_models > 0 {
                Array1::from_elem(n_models, 1.0 / n_models as Float)
            } else {
                Array1::zeros(0)
            };
            FusionParameters {
                linear_weights: Some(uniform),
                neural_weights: None,
                neural_biases: None,
                gating_weights: None,
                attention_weights: None,
                regularization_penalty: 0.0,
            }
        }
    }
}

impl Estimator for ModelFusion<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for ModelFusion<Untrained> {
    type Fitted = ModelFusion<ModelFusionTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Fitted> {
        self.validate_configuration()?;

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(),
                y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        // Actually train every base model on the training data (this used to be a
        // silent no-op that just forwarded the untrained models unchanged).
        let mut fitted_base_models = Vec::with_capacity(self.base_models.len());
        for (name, mut model) in self.base_models {
            model.fit(x, y)?;
            fitted_base_models.push((name, model));
        }

        // Learn the fusion parameters from the *fitted* models' predictions, not
        // from data collected before training (which previously wasn't collected
        // at all — the learned weights were hard-coded to uniform 1/n_models).
        let predictions = collect_predictions(&fitted_base_models, x)?;
        let fusion_params =
            learn_fusion_parameters(&self.fusion_strategy, &predictions, y, self.random_state)?;

        Ok(ModelFusion {
            state: ModelFusionTrained {
                fitted_base_models,
                fusion_params,
                n_features_in: x.ncols(),
                feature_names_in: None,
            },
            base_models: Vec::new(),
            fusion_strategy: self.fusion_strategy,
            regularization: self.regularization,
            enable_cv: self.enable_cv,
            cv_folds: self.cv_folds,
            scale_features: self.scale_features,
            calibration_temperature: self.calibration_temperature,
            uncertainty_estimation: self.uncertainty_estimation,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
        })
    }
}

/// Collect predictions from a fixed set of (name, model) pairs.
fn collect_predictions(
    models: &[(String, Box<dyn PipelinePredictor>)],
    x: &ArrayView2<'_, Float>,
) -> SklResult<Vec<Array1<Float>>> {
    models.iter().map(|(_, model)| model.predict(x)).collect()
}

/// Stack per-model prediction vectors as columns of a design matrix.
fn build_design_matrix(predictions: &[Array1<Float>], n_samples: usize) -> Array2<Float> {
    let n_models = predictions.len();
    let mut design = Array2::<Float>::zeros((n_samples, n_models));
    for (col, pred) in predictions.iter().enumerate() {
        for (row, &value) in pred.iter().enumerate() {
            design[[row, col]] = value;
        }
    }
    design
}

/// Learn fusion parameters from the real, fitted base models' predictions.
///
/// Only [`FusionStrategy::WeightedLinear`] and [`FusionStrategy::NeuralNetwork`]
/// currently have a genuine training implementation. Every other strategy
/// returns [`SklearsError::NotImplemented`] rather than fabricating a
/// "successfully trained" result.
fn learn_fusion_parameters(
    strategy: &FusionStrategy,
    predictions: &[Array1<Float>],
    y: &ArrayView1<'_, Float>,
    random_state: Option<u64>,
) -> SklResult<FusionParameters> {
    match strategy {
        FusionStrategy::WeightedLinear { regularization } => {
            let weights = learn_linear_weights(predictions, y, *regularization)?;
            Ok(FusionParameters {
                linear_weights: Some(weights),
                neural_weights: None,
                neural_biases: None,
                gating_weights: None,
                attention_weights: None,
                regularization_penalty: regularization.unwrap_or(0.0),
            })
        }
        FusionStrategy::NeuralNetwork {
            hidden_layers,
            activation,
            ..
        } => learn_neural_network(predictions, y, hidden_layers, activation, random_state),
        other => Err(SklearsError::NotImplemented(format!(
            "ModelFusion training for fusion strategy {other:?} is not yet implemented; \
             use FusionStrategy::WeightedLinear or FusionStrategy::NeuralNetwork"
        ))),
    }
}

/// Learn linear combination weights by solving `argmin_w ||P w - y||^2`, where
/// `P`'s columns are the base models' predictions. When `regularization` carries
/// a positive value, solves the ridge-regularized normal equations
/// `(PᵀP + λI) w = Pᵀy` instead.
fn learn_linear_weights(
    predictions: &[Array1<Float>],
    y: &ArrayView1<'_, Float>,
    regularization: Option<Float>,
) -> SklResult<Array1<Float>> {
    let n_models = predictions.len();
    let n_samples = y.len();
    let design = build_design_matrix(predictions, n_samples);
    let target: Array1<Float> = y.to_owned();

    let weights = match regularization {
        Some(lambda) if lambda > 0.0 => {
            let mut gram = design.t().dot(&design);
            for i in 0..n_models {
                gram[[i, i]] += lambda;
            }
            let rhs = design.t().dot(&target);
            solve_ndarray(&gram, &rhs).map_err(|e| {
                SklearsError::NumericalError(format!(
                    "failed to solve ridge-regularized fusion weights: {e}"
                ))
            })?
        }
        _ => lstsq_ndarray(&design, &target).map_err(|e| {
            SklearsError::NumericalError(format!(
                "failed to solve least-squares fusion weights: {e}"
            ))
        })?,
    };

    Ok(weights)
}

/// Activation function used inside the small fusion MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlpActivation {
    Relu,
    Sigmoid,
    Tanh,
    Identity,
}

impl MlpActivation {
    fn parse(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "relu" => Self::Relu,
            "sigmoid" => Self::Sigmoid,
            "tanh" => Self::Tanh,
            _ => Self::Identity,
        }
    }

    fn apply(self, z: Float) -> Float {
        match self {
            Self::Relu => z.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            Self::Tanh => z.tanh(),
            Self::Identity => z,
        }
    }

    /// Derivative expressed in terms of the activation's own output `a = f(z)`.
    fn derivative_from_output(self, activated: Float) -> Float {
        match self {
            Self::Relu => {
                if activated > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sigmoid => activated * (1.0 - activated),
            Self::Tanh => 1.0 - activated * activated,
            Self::Identity => 1.0,
        }
    }
}

/// Forward pass through the MLP. Returns the activation of every layer,
/// `activations[0] == x` and `activations[last]` is the network output. The
/// final layer always uses the identity activation (regression head).
fn mlp_forward(
    x: &Array2<Float>,
    weights: &[Array2<Float>],
    biases: &[Array1<Float>],
    activation: MlpActivation,
) -> Vec<Array2<Float>> {
    let n_layers = weights.len();
    let mut activations: Vec<Array2<Float>> = Vec::with_capacity(n_layers + 1);
    activations.push(x.clone());

    for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
        let mut z = activations[i].dot(w);
        for mut row in z.rows_mut() {
            row += b;
        }
        let is_output_layer = i + 1 == n_layers;
        let a = if is_output_layer {
            z
        } else {
            z.mapv(|v| activation.apply(v))
        };
        activations.push(a);
    }

    activations
}

/// Backpropagate mean-squared-error gradients through the MLP, returning
/// `(grad_weights, grad_biases)` aligned with `weights`/`biases`.
fn mlp_backward(
    activations: &[Array2<Float>],
    weights: &[Array2<Float>],
    y: &Array2<Float>,
    activation: MlpActivation,
) -> (Vec<Array2<Float>>, Vec<Array1<Float>>) {
    let n_layers = weights.len();
    let n_samples = activations[0].nrows() as Float;

    let a_last = &activations[n_layers];
    let mut delta: Array2<Float> = (a_last - y).mapv(|v| v * (2.0 / n_samples));

    let mut rev_grad_w = Vec::with_capacity(n_layers);
    let mut rev_grad_b = Vec::with_capacity(n_layers);

    for i in (0..n_layers).rev() {
        let a_prev = &activations[i];
        rev_grad_w.push(a_prev.t().dot(&delta));
        rev_grad_b.push(delta.sum_axis(Axis(0)));

        if i > 0 {
            let d_prev = delta.dot(&weights[i].t());
            let deriv = activations[i].mapv(|v| activation.derivative_from_output(v));
            delta = d_prev * deriv;
        }
    }

    rev_grad_w.reverse();
    rev_grad_b.reverse();
    (rev_grad_w, rev_grad_b)
}

/// Train a small MLP fusion network via real gradient descent on the base
/// models' predictions, minimizing mean squared error against `y`.
fn learn_neural_network(
    predictions: &[Array1<Float>],
    y: &ArrayView1<'_, Float>,
    hidden_layers: &[usize],
    activation_name: &str,
    random_state: Option<u64>,
) -> SklResult<FusionParameters> {
    if hidden_layers.is_empty() {
        return Err(SklearsError::InvalidParameter {
            name: "hidden_layers".to_string(),
            reason: "neural network fusion requires at least one hidden layer".to_string(),
        });
    }

    let n_models = predictions.len();
    let n_samples = y.len();
    let design = build_design_matrix(predictions, n_samples);

    let mut target = Array2::<Float>::zeros((n_samples, 1));
    for (i, &v) in y.iter().enumerate() {
        target[[i, 0]] = v;
    }

    let mut layer_dims = Vec::with_capacity(hidden_layers.len() + 2);
    layer_dims.push(n_models);
    layer_dims.extend_from_slice(hidden_layers);
    layer_dims.push(1);

    let activation = MlpActivation::parse(activation_name);
    let mut rng = StdRng::seed_from_u64(random_state.unwrap_or(42));

    let mut weights: Vec<Array2<Float>> = Vec::with_capacity(layer_dims.len() - 1);
    let mut biases: Vec<Array1<Float>> = Vec::with_capacity(layer_dims.len() - 1);
    for dims in layer_dims.windows(2) {
        let (fan_in, fan_out) = (dims[0], dims[1]);
        // Xavier/Glorot uniform initialization.
        let limit = (6.0 / (fan_in + fan_out) as Float).sqrt();
        let mut w = Array2::<Float>::zeros((fan_in, fan_out));
        for v in w.iter_mut() {
            *v = rng.random_range(-limit..limit);
        }
        weights.push(w);
        biases.push(Array1::zeros(fan_out));
    }

    const EPOCHS: usize = 500;
    const LEARNING_RATE: Float = 0.02;
    const GRADIENT_CLIP: Float = 10.0;

    for _ in 0..EPOCHS {
        let activations = mlp_forward(&design, &weights, &biases, activation);
        let (grad_w, grad_b) = mlp_backward(&activations, &weights, &target, activation);
        for i in 0..weights.len() {
            let step_w = grad_w[i].mapv(|g| g.clamp(-GRADIENT_CLIP, GRADIENT_CLIP) * LEARNING_RATE);
            let step_b = grad_b[i].mapv(|g| g.clamp(-GRADIENT_CLIP, GRADIENT_CLIP) * LEARNING_RATE);
            weights[i] = &weights[i] - &step_w;
            biases[i] = &biases[i] - &step_b;
        }
    }

    Ok(FusionParameters {
        linear_weights: None,
        neural_weights: Some(weights),
        neural_biases: Some(biases),
        gating_weights: None,
        attention_weights: None,
        regularization_penalty: 0.0,
    })
}

/// Real (non-fabricated) per-model importance proxy for a trained fusion MLP:
/// the L1 norm of each input model's row in the first layer's weight matrix,
/// normalized to sum to one.
fn neural_contribution(first_layer: &Array2<Float>) -> Array1<Float> {
    let mut contribution = Array1::from_shape_fn(first_layer.nrows(), |j| {
        first_layer.row(j).mapv(Float::abs).sum()
    });
    let total: Float = contribution.sum();
    if total > 0.0 {
        contribution.mapv_inplace(|v| v / total);
    }
    contribution
}

impl ModelFusion<ModelFusionTrained> {
    /// Predict using the fitted fusion ensemble.
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }
        Ok(self.predict_with_details(x)?.prediction)
    }

    /// Predict with detailed fusion information
    pub fn predict_with_details(&self, x: &ArrayView2<'_, Float>) -> SklResult<FusionPrediction> {
        let individual_predictions = self.collect_base_predictions(x)?;

        let (fused_prediction, fusion_weights) = match &self.fusion_strategy {
            FusionStrategy::WeightedLinear { .. } => {
                self.apply_linear_fusion(&individual_predictions)?
            }
            FusionStrategy::NeuralNetwork { .. } => {
                self.apply_neural_fusion(&individual_predictions)?
            }
            other => {
                return Err(SklearsError::InvalidState(format!(
                    "trained ModelFusion holds unsupported fusion strategy {other:?}; \
                     this should be unreachable because fit() rejects it"
                )));
            }
        };

        let confidence = calculate_confidence(&individual_predictions, &fused_prediction);

        Ok(FusionPrediction {
            prediction: fused_prediction,
            individual_predictions,
            fusion_weights,
            uncertainty: None,
            confidence,
        })
    }

    /// Apply the learned linear fusion weights.
    fn apply_linear_fusion(
        &self,
        predictions: &[Array1<Float>],
    ) -> SklResult<(Array1<Float>, Array1<Float>)> {
        let weights = self
            .state
            .fusion_params
            .linear_weights
            .as_ref()
            .ok_or_else(|| {
                SklearsError::InvalidState(
                    "linear fusion weights were not learned during fit".to_string(),
                )
            })?;

        let n_samples = predictions.first().map_or(0, Array1::len);
        let mut fused = Array1::<Float>::zeros(n_samples);
        for (model_idx, pred) in predictions.iter().enumerate() {
            let w = weights[model_idx];
            for sample_idx in 0..n_samples {
                fused[sample_idx] += pred[sample_idx] * w;
            }
        }

        Ok((fused, weights.clone()))
    }

    /// Run the trained fusion MLP forward to produce predictions.
    fn apply_neural_fusion(
        &self,
        predictions: &[Array1<Float>],
    ) -> SklResult<(Array1<Float>, Array1<Float>)> {
        let (weights, biases) = match (
            &self.state.fusion_params.neural_weights,
            &self.state.fusion_params.neural_biases,
        ) {
            (Some(w), Some(b)) => (w, b),
            _ => {
                return Err(SklearsError::InvalidState(
                    "neural fusion parameters were not learned during fit".to_string(),
                ))
            }
        };

        let activation = match &self.fusion_strategy {
            FusionStrategy::NeuralNetwork { activation, .. } => MlpActivation::parse(activation),
            _ => MlpActivation::Identity,
        };

        let n_samples = predictions.first().map_or(0, Array1::len);
        let design = build_design_matrix(predictions, n_samples);
        let activations = mlp_forward(&design, weights, biases, activation);
        let output = activations
            .last()
            .ok_or_else(|| SklearsError::InvalidState("empty MLP activations".to_string()))?;
        let fused = output.column(0).to_owned();

        let contribution = weights
            .first()
            .map_or_else(|| Array1::zeros(predictions.len()), neural_contribution);

        Ok((fused, contribution))
    }

    /// Collect predictions from all fitted base models
    fn collect_base_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
        collect_predictions(&self.state.fitted_base_models, x)
    }

    /// Evaluate fusion performance
    pub fn evaluate_fusion(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<FusionMetrics> {
        let predictions = self.predict(x)?;

        let mse = y
            .iter()
            .zip(predictions.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
            .sum::<Float>()
            / y.len() as Float;

        let mae = y
            .iter()
            .zip(predictions.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).abs())
            .sum::<Float>()
            / y.len() as Float;

        let y_mean = y.mean().unwrap_or(0.0);
        let ss_res: Float = y
            .iter()
            .zip(predictions.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
            .sum();
        let ss_tot: Float = y.iter().map(|true_val| (true_val - y_mean).powi(2)).sum();
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        // Real per-model contribution, derived from whatever the fusion layer
        // actually learned (linear weights, or the neural importance proxy),
        // rather than a hard-coded uniform 1/n_models placeholder.
        let contributions: Array1<Float> = match &self.fusion_strategy {
            FusionStrategy::WeightedLinear { .. } => self
                .state
                .fusion_params
                .linear_weights
                .clone()
                .unwrap_or_else(|| Array1::zeros(self.state.fitted_base_models.len())),
            FusionStrategy::NeuralNetwork { .. } => self
                .state
                .fusion_params
                .neural_weights
                .as_ref()
                .and_then(|w| w.first())
                .map_or_else(
                    || Array1::zeros(self.state.fitted_base_models.len()),
                    neural_contribution,
                ),
            _ => Array1::zeros(self.state.fitted_base_models.len()),
        };

        let mut model_contributions = HashMap::new();
        for (idx, (name, _)) in self.state.fitted_base_models.iter().enumerate() {
            let contribution = contributions.get(idx).copied().unwrap_or(0.0);
            model_contributions.insert(name.clone(), contribution);
        }

        Ok(FusionMetrics {
            mse,
            mae,
            r2,
            model_contributions,
            complexity: self.get_parameter_count(),
        })
    }

    /// Get total parameter count for complexity measurement
    fn get_parameter_count(&self) -> usize {
        match &self.fusion_strategy {
            FusionStrategy::NeuralNetwork { hidden_layers, .. } if !hidden_layers.is_empty() => {
                let n_models = self.state.fitted_base_models.len();
                let mut count = n_models * hidden_layers[0];
                for i in 1..hidden_layers.len() {
                    count += hidden_layers[i - 1] * hidden_layers[i];
                }
                count += hidden_layers.last().copied().unwrap_or(0);
                count
            }
            _ => self.state.fitted_base_models.len(),
        }
    }

    /// Get the fitted base models.
    #[must_use]
    pub fn base_models(&self) -> &[(String, Box<dyn PipelinePredictor>)] {
        &self.state.fitted_base_models
    }

    /// Get the learned fusion parameters.
    #[must_use]
    pub fn fusion_params(&self) -> &FusionParameters {
        &self.state.fusion_params
    }

    /// Get the number of input features seen during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.state.n_features_in
    }
}

/// Calculate prediction confidence from ensemble agreement (real, non-fabricated:
/// higher disagreement/variance between base models yields lower confidence).
fn calculate_confidence(
    predictions: &[Array1<Float>],
    fused_prediction: &Array1<Float>,
) -> Array1<Float> {
    let n_samples = fused_prediction.len();
    let mut confidence = Array1::zeros(n_samples);

    if predictions.is_empty() {
        return confidence;
    }

    for sample_idx in 0..n_samples {
        let fused_val = fused_prediction[sample_idx];
        let variance: Float = predictions
            .iter()
            .map(|pred| (pred[sample_idx] - fused_val).powi(2))
            .sum::<Float>()
            / predictions.len() as Float;
        confidence[sample_idx] = 1.0 / (1.0 + variance);
    }

    confidence
}

/// Configuration for ModelFusion
#[derive(Debug, Clone)]
pub struct ModelFusionConfig {
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Regularization settings
    pub regularization: Option<(RegularizationType, Float)>,
    /// Enable cross-validation
    pub enable_cv: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Scale features
    pub scale_features: bool,
    /// Calibration temperature
    pub calibration_temperature: Option<Float>,
    /// Enable uncertainty estimation
    pub uncertainty_estimation: bool,
    /// Random state
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for ModelFusionConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::WeightedLinear {
                regularization: None,
            },
            regularization: None,
            enable_cv: false,
            cv_folds: 5,
            scale_features: true,
            calibration_temperature: None,
            uncertainty_estimation: false,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }
}

/// Builder for ModelFusion
#[derive(Debug)]
pub struct ModelFusionBuilder {
    base_models: Vec<(String, Box<dyn PipelinePredictor>)>,
    fusion_strategy: FusionStrategy,
    regularization: Option<(RegularizationType, Float)>,
    enable_cv: bool,
    cv_folds: usize,
    scale_features: bool,
    calibration_temperature: Option<Float>,
    uncertainty_estimation: bool,
    random_state: Option<u64>,
    n_jobs: Option<i32>,
    verbose: bool,
}

impl ModelFusionBuilder {
    /// Create new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_models: Vec::new(),
            fusion_strategy: FusionStrategy::WeightedLinear {
                regularization: None,
            },
            regularization: None,
            enable_cv: false,
            cv_folds: 5,
            scale_features: true,
            calibration_temperature: None,
            uncertainty_estimation: false,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add a base model
    #[must_use]
    pub fn base_model(mut self, name: &str, model: Box<dyn PipelinePredictor>) -> Self {
        self.base_models.push((name.to_string(), model));
        self
    }

    /// Set fusion strategy
    #[must_use]
    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set regularization
    #[must_use]
    pub fn regularization(mut self, reg_type: RegularizationType, strength: Float) -> Self {
        self.regularization = Some((reg_type, strength));
        self
    }

    /// Enable cross-validation
    #[must_use]
    pub fn enable_cv(mut self, enable: bool) -> Self {
        self.enable_cv = enable;
        self
    }

    /// Set CV folds
    #[must_use]
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Set feature scaling
    #[must_use]
    pub fn scale_features(mut self, scale: bool) -> Self {
        self.scale_features = scale;
        self
    }

    /// Set calibration temperature
    #[must_use]
    pub fn calibration_temperature(mut self, temperature: Float) -> Self {
        self.calibration_temperature = Some(temperature);
        self
    }

    /// Enable uncertainty estimation
    #[must_use]
    pub fn uncertainty_estimation(mut self, enable: bool) -> Self {
        self.uncertainty_estimation = enable;
        self
    }

    /// Set random state
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Set verbose flag
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the ModelFusion
    #[must_use]
    pub fn build(self) -> ModelFusion<Untrained> {
        ModelFusion {
            state: Untrained,
            base_models: self.base_models,
            fusion_strategy: self.fusion_strategy,
            regularization: self.regularization,
            enable_cv: self.enable_cv,
            cv_folds: self.cv_folds,
            scale_features: self.scale_features,
            calibration_temperature: self.calibration_temperature,
            uncertainty_estimation: self.uncertainty_estimation,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
        }
    }
}

impl Default for ModelFusionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockPredictor;
    use scirs2_core::ndarray::array;

    /// Test-only predictor that ignores its inputs at `fit` time and always
    /// returns a fixed, caller-controlled prediction vector (cycled if the
    /// requested number of rows differs from the stored vector's length). Used
    /// to build fusion scenarios with an exactly-known "perfect" model and
    /// "pure noise" models, which a hard-coded uniform-weight fusion could not
    /// pass.
    #[derive(Debug, Clone)]
    struct FixedPredictor {
        values: Array1<Float>,
    }

    impl FixedPredictor {
        fn new(values: Vec<Float>) -> Self {
            Self {
                values: Array1::from_vec(values),
            }
        }
    }

    impl PipelinePredictor for FixedPredictor {
        fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
            let n = x.nrows();
            let vals: Vec<Float> = self.values.iter().copied().cycle().take(n).collect();
            Ok(Array1::from_vec(vals))
        }

        fn fit(&mut self, _x: &ArrayView2<'_, Float>, _y: &ArrayView1<'_, Float>) -> SklResult<()> {
            Ok(())
        }

        fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_model_fusion_builder() {
        let fusion = ModelFusion::builder()
            .fusion_strategy(FusionStrategy::NeuralNetwork {
                hidden_layers: vec![64, 32],
                activation: "relu".to_string(),
                dropout: Some(0.2),
            })
            .enable_cv(true)
            .cv_folds(10)
            .scale_features(false)
            .build();

        match fusion.fusion_strategy {
            FusionStrategy::NeuralNetwork {
                ref hidden_layers, ..
            } => {
                assert_eq!(hidden_layers, &vec![64, 32]);
            }
            _ => panic!("Expected NeuralNetwork fusion strategy"),
        }
        assert!(fusion.enable_cv);
        assert_eq!(fusion.cv_folds, 10);
        assert!(!fusion.scale_features);
    }

    #[test]
    fn test_fusion_strategies() {
        let strategies = vec![
            FusionStrategy::WeightedLinear {
                regularization: Some(0.01),
            },
            FusionStrategy::NeuralNetwork {
                hidden_layers: vec![32],
                activation: "tanh".to_string(),
                dropout: None,
            },
            FusionStrategy::GatingNetwork {
                gating_type: GatingType::MixtureOfExperts,
                temperature: 1.5,
            },
            FusionStrategy::AttentionFusion {
                num_heads: 8,
                head_dim: 64,
                multi_head: true,
            },
        ];

        for strategy in strategies {
            let fusion = ModelFusion::builder()
                .fusion_strategy(strategy.clone())
                .build();
            assert_eq!(fusion.fusion_strategy, strategy);
        }
    }

    #[test]
    fn test_gating_types() {
        let gating_types = vec![
            GatingType::MixtureOfExperts,
            GatingType::Hierarchical,
            GatingType::InputDependent,
            GatingType::Probabilistic,
            GatingType::AttentionBased,
        ];

        for gating_type in gating_types {
            let strategy = FusionStrategy::GatingNetwork {
                gating_type: gating_type.clone(),
                temperature: 1.0,
            };
            let fusion = ModelFusion::builder().fusion_strategy(strategy).build();

            match fusion.fusion_strategy {
                FusionStrategy::GatingNetwork {
                    gating_type: gt, ..
                } => {
                    assert_eq!(gt, gating_type);
                }
                _ => panic!("Expected GatingNetwork fusion strategy"),
            }
        }
    }

    #[test]
    fn test_regularization_types() {
        let reg_types = vec![
            RegularizationType::L1,
            RegularizationType::L2,
            RegularizationType::ElasticNet { l1_ratio: 0.5 },
            RegularizationType::GroupLasso,
            RegularizationType::Nuclear,
        ];

        for reg_type in reg_types {
            let fusion = ModelFusion::builder()
                .regularization(reg_type.clone(), 0.01)
                .build();

            if let Some((rt, strength)) = fusion.regularization {
                assert_eq!(rt, reg_type);
                assert_eq!(strength, 0.01);
            } else {
                panic!("Expected regularization to be set");
            }
        }
    }

    #[test]
    fn test_parameter_initialization() {
        let fusion = ModelFusion::new();
        let params = fusion.initialize_parameters(10);

        assert!(params.linear_weights.is_some());
        assert!(params.neural_weights.is_none());
        assert!(params.gating_weights.is_none());
        assert!(params.attention_weights.is_none());
    }

    #[test]
    fn test_configuration_validation() {
        let fusion = ModelFusion::new();
        assert!(fusion.validate_configuration().is_err());

        let mut fusion = ModelFusion::new();
        fusion.cv_folds = 1;
        assert!(fusion.validate_configuration().is_err());

        let fusion = ModelFusion::builder()
            .fusion_strategy(FusionStrategy::NeuralNetwork {
                hidden_layers: vec![],
                activation: "relu".to_string(),
                dropout: None,
            })
            .build();
        assert!(fusion.validate_configuration().is_err());
    }

    /// Regression test for the original silent-fabrication bug: `fit()` used to
    /// forward the untrained base models unchanged (`// Note: In practice, each
    /// model would be properly trained`). An unfitted `MockPredictor` returns
    /// `Err(NotFitted)` from `predict`, so if `ModelFusion::fit` had merely
    /// forwarded it, extracting the base model and calling `predict` on it
    /// directly would fail here.
    #[test]
    fn test_base_models_are_actually_fitted() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let fusion = ModelFusion::builder()
            .base_model("m1", Box::new(MockPredictor::new()))
            .fusion_strategy(FusionStrategy::WeightedLinear {
                regularization: None,
            })
            .build();

        let fitted = fusion
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");

        let (_, model) = &fitted.base_models()[0];
        let preds = model
            .predict(&x.view())
            .expect("base model should be genuinely fitted by ModelFusion::fit");
        assert_eq!(preds.len(), x.nrows());
    }

    /// The test the original bug would have failed: one base model exactly
    /// matches the target, the others are pure noise linearly independent from
    /// it. A real least-squares solve must concentrate weight on the good
    /// model; the original code always returned uniform `1/n_models` weights
    /// regardless of the data.
    #[test]
    fn test_linear_weights_concentrate_on_good_model() {
        let y_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise_b = vec![10.0, -10.0, 10.0, -10.0, 10.0, -10.0];
        let noise_c = vec![-3.0, 7.0, -3.0, 7.0, -3.0, 7.0];

        let x = Array2::from_shape_vec((6, 1), (0..6).map(|v| v as Float).collect())
            .expect("valid shape");
        let y = Array1::from_vec(y_vals.clone());

        let fusion = ModelFusion::builder()
            .base_model("good", Box::new(FixedPredictor::new(y_vals)))
            .base_model("noise_b", Box::new(FixedPredictor::new(noise_b)))
            .base_model("noise_c", Box::new(FixedPredictor::new(noise_c)))
            .fusion_strategy(FusionStrategy::WeightedLinear {
                regularization: None,
            })
            .build();

        let fitted = fusion
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");

        let weights = fitted
            .fusion_params()
            .linear_weights
            .as_ref()
            .expect("linear weights should have been learned");

        assert_eq!(weights.len(), 3);
        assert!(
            weights[0] > 0.9,
            "good model's weight should dominate, got {weights:?}"
        );
        assert!(
            weights[1].abs() < 0.1,
            "noisy model's weight should be near zero, got {weights:?}"
        );
        assert!(
            weights[2].abs() < 0.1,
            "noisy model's weight should be near zero, got {weights:?}"
        );

        // And the fused prediction should essentially reproduce y exactly.
        let preds = fitted.predict(&x.view()).expect("predict should succeed");
        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1e-6, "expected {t}, got {p}");
        }
    }

    #[test]
    fn test_neural_fusion_trains_real_weights() {
        let x = Array2::from_shape_vec((8, 1), (0..8).map(|v| v as Float).collect())
            .expect("valid shape");
        let y_vals: Vec<Float> = (0..8).map(|v| v as Float * 2.0 + 1.0).collect();
        let y = Array1::from_vec(y_vals.clone());

        let fusion = ModelFusion::builder()
            .base_model("a", Box::new(FixedPredictor::new(y_vals.clone())))
            .base_model(
                "b",
                Box::new(FixedPredictor::new(vec![
                    5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0,
                ])),
            )
            .fusion_strategy(FusionStrategy::NeuralNetwork {
                hidden_layers: vec![4],
                activation: "tanh".to_string(),
                dropout: None,
            })
            .random_state(7)
            .build();

        let fitted = fusion
            .fit(&x.view(), &y.view())
            .expect("neural fusion fit should succeed");

        let params = fitted.fusion_params();
        let weights = params
            .neural_weights
            .as_ref()
            .expect("neural weights should have been learned");
        // At least one weight must have moved away from its zero initialization.
        assert!(weights.iter().any(|w| w.iter().any(|&v| v.abs() > 1e-6)));

        // Training should have reduced the error substantially below what a
        // naive "always predict the mean of y" baseline would achieve.
        let preds = fitted.predict(&x.view()).expect("predict should succeed");
        let mse: Float = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<Float>()
            / y.len() as Float;
        let y_mean = y.mean().unwrap_or(0.0);
        let baseline_mse: Float =
            y.iter().map(|t| (t - y_mean).powi(2)).sum::<Float>() / y.len() as Float;
        assert!(
            mse < baseline_mse,
            "trained neural fusion (mse={mse}) should beat the mean baseline (mse={baseline_mse})"
        );
    }

    #[test]
    fn test_unimplemented_strategy_returns_honest_error() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 2.0, 3.0];

        let fusion = ModelFusion::builder()
            .base_model("m1", Box::new(MockPredictor::new()))
            .fusion_strategy(FusionStrategy::GatingNetwork {
                gating_type: GatingType::MixtureOfExperts,
                temperature: 1.0,
            })
            .build();

        let result = fusion.fit(&x.view(), &y.view());
        assert!(
            result.is_err(),
            "unimplemented fusion strategies must error, not silently succeed"
        );
    }
}
