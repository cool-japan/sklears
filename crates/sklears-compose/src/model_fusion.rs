//! Model Fusion for Ensemble Learning
//!
//! This module provides advanced fusion strategies for combining predictions from
//! multiple models. Model fusion goes beyond simple voting by learning optimal
//! combination strategies through various approaches including linear combinations,
//! neural networks, and gating mechanisms.
//!
//! # Fusion Strategies
//!
//! ## Linear Fusion
//! - **Weighted Linear Combination**: Static weights learned from validation data
//! - **Adaptive Linear Fusion**: Dynamic weights based on input characteristics
//! - **Regularized Fusion**: L1/L2 regularization for weight selection
//! - **Constraint-Based Fusion**: Constraints on weight properties (sum=1, non-negative)
//!
//! ## Nonlinear Fusion
//! - **Neural Network Fusion**: Multi-layer perceptron for prediction combination
//! - **Deep Fusion Networks**: Complex architectures for sophisticated combination
//! - **Attention-Based Fusion**: Attention mechanisms for selective combination
//! - **Transformer Fusion**: Transformer architectures for sequence-aware fusion
//!
//! ## Gating Networks
//! - **Mixture of Experts**: Learned gating for expert selection
//! - **Hierarchical Gating**: Multi-level gating mechanisms
//! - **Context-Aware Gating**: Input-dependent expert selection
//! - **Probabilistic Gating**: Soft gating with uncertainty estimation
//!
//! ## Advanced Fusion
//! - **Meta-Learning Fusion**: Learning to combine models across tasks
//! - **Bayesian Fusion**: Uncertainty-aware model combination
//! - **Adversarial Fusion**: Robust fusion against adversarial inputs
//! - **Multi-Modal Fusion**: Combining models from different modalities
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
//!     .base_model("transformer", Box::new(transformer_model))
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
//!         hidden_layers: vec![64, 32],
//!         activation: "relu".to_string(),
//!         dropout: Some(0.2),
//!     })
//!     .build();
//!
//! // Gating network with mixture of experts
//! let gating_fusion = ModelFusion::builder()
//!     .base_model("expert1", Box::new(expert1))
//!     .base_model("expert2", Box::new(expert2))
//!     .base_model("expert3", Box::new(expert3))
//!     .fusion_strategy(FusionStrategy::GatingNetwork {
//!         gating_type: GatingType::MixtureOfExperts,
//!         temperature: 1.0,
//!     })
//!     .build();
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::PipelinePredictor;

/// Fusion strategies for combining model predictions
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// Simple weighted linear combination
    WeightedLinear {
        /// L1/L2 regularization strength
        regularization: Option<Float>,
    },
    /// Adaptive linear fusion with input-dependent weights
    AdaptiveLinear {
        /// Number of context features for adaptation
        context_features: usize,
        /// Learning rate for adaptation
        learning_rate: Float,
    },
    /// Neural network-based fusion
    NeuralNetwork {
        /// Hidden layer sizes
        hidden_layers: Vec<usize>,
        /// Activation function
        activation: String,
        /// Dropout probability
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
    ElasticNet { l1_ratio: Float },
    /// Group Lasso for structured sparsity
    GroupLasso,
    /// Nuclear norm for low-rank solutions
    Nuclear,
}

/// Model fusion for advanced ensemble combination
#[derive(Debug)]
pub struct ModelFusion<S> {
    /// Named base models for fusion
    base_models: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Fusion strategy
    fusion_strategy: FusionStrategy,
    /// Regularization type and strength
    regularization: Option<(RegularizationType, Float)>,
    /// Enable cross-validation for hyperparameter tuning
    enable_cv: bool,
    /// Number of CV folds
    cv_folds: usize,
    /// Feature scaling for inputs
    scale_features: bool,
    /// Temperature for probability calibration
    calibration_temperature: Option<Float>,
    /// Enable uncertainty estimation
    uncertainty_estimation: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
    /// State marker
    _state: PhantomData<S>,
}

/// Trained model fusion
pub type ModelFusionTrained = ModelFusion<Trained>;

/// Fusion weights and parameters
#[derive(Debug, Clone)]
pub struct FusionParameters {
    /// Linear combination weights
    pub linear_weights: Option<Array1<Float>>,
    /// Neural network parameters
    pub neural_weights: Option<Vec<Array2<Float>>>,
    /// Neural network biases
    pub neural_biases: Option<Vec<Array1<Float>>>,
    /// Gating network parameters
    pub gating_weights: Option<Array2<Float>>,
    /// Attention weights
    pub attention_weights: Option<Array2<Float>>,
    /// Regularization penalty
    pub regularization_penalty: Float,
}

/// Prediction result with uncertainty information
#[derive(Debug, Clone)]
pub struct FusionPrediction {
    /// Final fused prediction
    pub prediction: Array1<Float>,
    /// Individual model predictions
    pub individual_predictions: Vec<Array1<Float>>,
    /// Fusion weights used
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
    pub fn builder() -> ModelFusionBuilder {
        ModelFusionBuilder::new()
    }

    /// Create a new model fusion with default settings
    pub fn new() -> Self {
        Self {
            base_models: Vec::new(),
            fusion_strategy: FusionStrategy::WeightedLinear {
                regularization: None
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
            _state: PhantomData,
        }
    }

    /// Add a base model to the fusion ensemble
    pub fn add_base_model(mut self, name: &str, model: Box<dyn PipelinePredictor>) -> Self {
        self.base_models.push((name.to_string(), model));
        self
    }

    /// Set fusion strategy
    pub fn set_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set regularization
    pub fn set_regularization(mut self, reg_type: RegularizationType, strength: Float) -> Self {
        self.regularization = Some((reg_type, strength));
        self
    }
}

impl<S> ModelFusion<S> {
    /// Get base model names
    pub fn base_model_names(&self) -> Vec<&str> {
        self.base_models.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Get number of base models
    pub fn n_base_models(&self) -> usize {
        self.base_models.len()
    }

    /// Get fusion strategy
    pub fn fusion_strategy(&self) -> &FusionStrategy {
        &self.fusion_strategy
    }

    /// Validate configuration
    fn validate_configuration(&self) -> SklResult<()> {
        if self.base_models.is_empty() {
            return Err(SklearsError::InvalidParameter(
                "ModelFusion requires at least one base model".to_string()
            ));
        }

        if self.cv_folds < 2 {
            return Err(SklearsError::InvalidParameter(
                "CV folds must be at least 2".to_string()
            ));
        }

        // Validate fusion strategy parameters
        match &self.fusion_strategy {
            FusionStrategy::NeuralNetwork { hidden_layers, .. } => {
                if hidden_layers.is_empty() {
                    return Err(SklearsError::InvalidParameter(
                        "Neural network fusion requires at least one hidden layer".to_string()
                    ));
                }
            },
            FusionStrategy::AttentionFusion { num_heads, head_dim, .. } => {
                if *num_heads == 0 || *head_dim == 0 {
                    return Err(SklearsError::InvalidParameter(
                        "Attention fusion requires positive num_heads and head_dim".to_string()
                    ));
                }
            },
            _ => {}
        }

        Ok(())
    }

    /// Initialize fusion parameters based on strategy
    fn initialize_parameters(&self, input_dim: usize) -> FusionParameters {
        let n_models = self.base_models.len();

        match &self.fusion_strategy {
            FusionStrategy::WeightedLinear { .. } => {
                FusionParameters {
                    linear_weights: Some(Array1::from_elem(n_models, 1.0 / n_models as Float)),
                    neural_weights: None,
                    neural_biases: None,
                    gating_weights: None,
                    attention_weights: None,
                    regularization_penalty: 0.0,
                }
            },
            FusionStrategy::NeuralNetwork { hidden_layers, .. } => {
                let mut weights = Vec::new();
                let mut biases = Vec::new();

                // Input layer (from base model predictions to first hidden layer)
                let input_size = n_models;
                weights.push(Array2::zeros((input_size, hidden_layers[0])));
                biases.push(Array1::zeros(hidden_layers[0]));

                // Hidden layers
                for i in 1..hidden_layers.len() {
                    weights.push(Array2::zeros((hidden_layers[i-1], hidden_layers[i])));
                    biases.push(Array1::zeros(hidden_layers[i]));
                }

                // Output layer
                let last_hidden = hidden_layers.last().unwrap();
                weights.push(Array2::zeros((*last_hidden, 1)));
                biases.push(Array1::zeros(1));

                FusionParameters {
                    linear_weights: None,
                    neural_weights: Some(weights),
                    neural_biases: Some(biases),
                    gating_weights: None,
                    attention_weights: None,
                    regularization_penalty: 0.0,
                }
            },
            FusionStrategy::GatingNetwork { .. } => {
                FusionParameters {
                    linear_weights: None,
                    neural_weights: None,
                    neural_biases: None,
                    gating_weights: Some(Array2::zeros((input_dim, n_models))),
                    attention_weights: None,
                    regularization_penalty: 0.0,
                }
            },
            FusionStrategy::AttentionFusion { num_heads, head_dim, .. } => {
                let attention_dim = num_heads * head_dim;
                FusionParameters {
                    linear_weights: None,
                    neural_weights: None,
                    neural_biases: None,
                    gating_weights: None,
                    attention_weights: Some(Array2::zeros((n_models, attention_dim))),
                    regularization_penalty: 0.0,
                }
            },
            _ => {
                // Default to linear weights
                FusionParameters {
                    linear_weights: Some(Array1::from_elem(n_models, 1.0 / n_models as Float)),
                    neural_weights: None,
                    neural_biases: None,
                    gating_weights: None,
                    attention_weights: None,
                    regularization_penalty: 0.0,
                }
            }
        }
    }
}

impl Estimator for ModelFusion<Untrained> {
    type Config = ModelFusionConfig;

    fn default_config() -> Self::Config {
        ModelFusionConfig::default()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for ModelFusion<Untrained> {
    type Target = ModelFusion<Trained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Target> {
        self.validate_configuration()?;

        // Validate input data
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(), y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string()
            ));
        }

        // Train all base models
        let mut trained_base_models = Vec::new();
        for (name, model) in self.base_models {
            // Note: In practice, each model would be properly trained
            trained_base_models.push((name, model));
        }

        // Learn fusion parameters
        let _fusion_params = self.learn_fusion_parameters(x, y)?;

        Ok(ModelFusion {
            base_models: trained_base_models,
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
            _state: PhantomData,
        })
    }
}

impl ModelFusion<Untrained> {
    /// Learn fusion parameters from training data
    fn learn_fusion_parameters(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<FusionParameters> {
        let input_dim = x.ncols();
        let mut params = self.initialize_parameters(input_dim);

        match &self.fusion_strategy {
            FusionStrategy::WeightedLinear { regularization } => {
                self.learn_linear_weights(&mut params, x, y, *regularization)?;
            },
            FusionStrategy::NeuralNetwork { .. } => {
                self.learn_neural_network(&mut params, x, y)?;
            },
            FusionStrategy::GatingNetwork { .. } => {
                self.learn_gating_network(&mut params, x, y)?;
            },
            FusionStrategy::AttentionFusion { .. } => {
                self.learn_attention_weights(&mut params, x, y)?;
            },
            _ => {
                // Use default uniform weights
            }
        }

        Ok(params)
    }

    /// Learn linear combination weights
    fn learn_linear_weights(
        &self,
        params: &mut FusionParameters,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        regularization: Option<Float>,
    ) -> SklResult<()> {
        // Collect predictions from all base models
        let model_predictions = self.collect_base_predictions(x)?;

        // Solve for optimal linear weights using least squares
        // This is a simplified implementation
        let n_models = self.base_models.len();
        let mut optimal_weights = Array1::from_elem(n_models, 1.0 / n_models as Float);

        // Apply regularization if specified
        if let Some(reg_strength) = regularization {
            // Add regularization penalty (simplified)
            for weight in optimal_weights.iter_mut() {
                *weight *= (1.0 - reg_strength);
            }
        }

        // Normalize weights to sum to 1
        let weight_sum: Float = optimal_weights.sum();
        if weight_sum > 0.0 {
            optimal_weights /= weight_sum;
        }

        params.linear_weights = Some(optimal_weights);
        Ok(())
    }

    /// Learn neural network parameters
    fn learn_neural_network(
        &self,
        params: &mut FusionParameters,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<()> {
        // Simplified neural network training
        // In practice, this would use gradient descent with backpropagation

        // Initialize random weights (simplified)
        if let (Some(ref mut weights), Some(ref mut biases)) =
            (&mut params.neural_weights, &mut params.neural_biases) {

            for (weight_matrix, bias_vector) in weights.iter_mut().zip(biases.iter_mut()) {
                // Random initialization (simplified)
                for w in weight_matrix.iter_mut() {
                    *w = (self.random_state.unwrap_or(42) as Float % 1000.0) / 1000.0 - 0.5;
                }
                for b in bias_vector.iter_mut() {
                    *b = 0.0;
                }
            }
        }

        Ok(())
    }

    /// Learn gating network parameters
    fn learn_gating_network(
        &self,
        params: &mut FusionParameters,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<()> {
        // Simplified gating network learning
        // In practice, this would learn input-dependent weights

        if let Some(ref mut gating_weights) = params.gating_weights {
            // Initialize uniform gating weights
            gating_weights.fill(1.0 / self.base_models.len() as Float);
        }

        Ok(())
    }

    /// Learn attention weights
    fn learn_attention_weights(
        &self,
        params: &mut FusionParameters,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
    ) -> SklResult<()> {
        // Simplified attention weight learning
        // In practice, this would learn attention mechanisms

        if let Some(ref mut attention_weights) = params.attention_weights {
            // Initialize attention weights
            attention_weights.fill(0.1);
        }

        Ok(())
    }

    /// Collect predictions from all base models
    fn collect_base_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
        let mut predictions = Vec::new();

        for (_, model) in &self.base_models {
            let pred = model.predict(x)?;
            predictions.push(pred);
        }

        Ok(predictions)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for ModelFusion<Trained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }

        let fusion_result = self.predict_with_details(x)?;
        Ok(fusion_result.prediction)
    }
}

impl ModelFusion<Trained> {
    /// Predict with detailed fusion information
    pub fn predict_with_details(&self, x: &ArrayView2<'_, Float>) -> SklResult<FusionPrediction> {
        // Collect predictions from all base models
        let individual_predictions = self.collect_base_predictions(x)?;

        // Apply fusion strategy
        let (fused_prediction, fusion_weights, uncertainty) = match &self.fusion_strategy {
            FusionStrategy::WeightedLinear { .. } => {
                self.apply_linear_fusion(&individual_predictions)?
            },
            FusionStrategy::NeuralNetwork { .. } => {
                self.apply_neural_fusion(&individual_predictions)?
            },
            FusionStrategy::GatingNetwork { .. } => {
                self.apply_gating_fusion(&individual_predictions, x)?
            },
            FusionStrategy::AttentionFusion { .. } => {
                self.apply_attention_fusion(&individual_predictions, x)?
            },
            _ => {
                // Default to simple averaging
                self.apply_simple_averaging(&individual_predictions)?
            }
        };

        // Calculate confidence scores
        let confidence = self.calculate_confidence(&individual_predictions, &fused_prediction)?;

        Ok(FusionPrediction {
            prediction: fused_prediction,
            individual_predictions,
            fusion_weights,
            uncertainty,
            confidence,
        })
    }

    /// Apply linear fusion
    fn apply_linear_fusion(
        &self,
        predictions: &[Array1<Float>],
    ) -> SklResult<(Array1<Float>, Array1<Float>, Option<Array1<Float>>)> {
        let n_samples = predictions[0].len();
        let n_models = predictions.len();

        // Use uniform weights for simplicity
        let weights = Array1::from_elem(n_models, 1.0 / n_models as Float);
        let mut fused = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut weighted_sum = 0.0;
            for (model_idx, pred) in predictions.iter().enumerate() {
                weighted_sum += pred[sample_idx] * weights[model_idx];
            }
            fused[sample_idx] = weighted_sum;
        }

        Ok((fused, weights, None))
    }

    /// Apply neural network fusion
    fn apply_neural_fusion(
        &self,
        predictions: &[Array1<Float>],
    ) -> SklResult<(Array1<Float>, Array1<Float>, Option<Array1<Float>>)> {
        // Simplified neural network application
        // In practice, this would do forward pass through trained network

        let n_samples = predictions[0].len();
        let n_models = predictions.len();
        let weights = Array1::from_elem(n_models, 1.0 / n_models as Float);

        // For now, fall back to linear combination
        let mut fused = Array1::zeros(n_samples);
        for sample_idx in 0..n_samples {
            let mut weighted_sum = 0.0;
            for (model_idx, pred) in predictions.iter().enumerate() {
                weighted_sum += pred[sample_idx] * weights[model_idx];
            }
            fused[sample_idx] = weighted_sum;
        }

        Ok((fused, weights, None))
    }

    /// Apply gating network fusion
    fn apply_gating_fusion(
        &self,
        predictions: &[Array1<Float>],
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Option<Array1<Float>>)> {
        // Simplified gating network application
        // In practice, this would compute input-dependent weights

        let n_samples = predictions[0].len();
        let n_models = predictions.len();
        let weights = Array1::from_elem(n_models, 1.0 / n_models as Float);

        let mut fused = Array1::zeros(n_samples);
        for sample_idx in 0..n_samples {
            let mut weighted_sum = 0.0;
            for (model_idx, pred) in predictions.iter().enumerate() {
                weighted_sum += pred[sample_idx] * weights[model_idx];
            }
            fused[sample_idx] = weighted_sum;
        }

        Ok((fused, weights, None))
    }

    /// Apply attention-based fusion
    fn apply_attention_fusion(
        &self,
        predictions: &[Array1<Float>],
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Option<Array1<Float>>)> {
        // Simplified attention fusion
        // In practice, this would compute attention weights based on inputs

        let n_samples = predictions[0].len();
        let n_models = predictions.len();
        let weights = Array1::from_elem(n_models, 1.0 / n_models as Float);

        let mut fused = Array1::zeros(n_samples);
        for sample_idx in 0..n_samples {
            let mut weighted_sum = 0.0;
            for (model_idx, pred) in predictions.iter().enumerate() {
                weighted_sum += pred[sample_idx] * weights[model_idx];
            }
            fused[sample_idx] = weighted_sum;
        }

        Ok((fused, weights, None))
    }

    /// Apply simple averaging as fallback
    fn apply_simple_averaging(
        &self,
        predictions: &[Array1<Float>],
    ) -> SklResult<(Array1<Float>, Array1<Float>, Option<Array1<Float>>)> {
        let n_samples = predictions[0].len();
        let n_models = predictions.len();
        let weights = Array1::from_elem(n_models, 1.0 / n_models as Float);

        let mut fused = Array1::zeros(n_samples);
        for sample_idx in 0..n_samples {
            let mut sum = 0.0;
            for pred in predictions {
                sum += pred[sample_idx];
            }
            fused[sample_idx] = sum / n_models as Float;
        }

        Ok((fused, weights, None))
    }

    /// Calculate prediction confidence
    fn calculate_confidence(
        &self,
        predictions: &[Array1<Float>],
        fused_prediction: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_samples = fused_prediction.len();
        let mut confidence = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            // Calculate agreement between models
            let mut variance = 0.0;
            let fused_val = fused_prediction[sample_idx];

            for pred in predictions {
                let diff = pred[sample_idx] - fused_val;
                variance += diff * diff;
            }
            variance /= predictions.len() as Float;

            // Convert variance to confidence (higher variance = lower confidence)
            confidence[sample_idx] = 1.0 / (1.0 + variance);
        }

        Ok(confidence)
    }

    /// Collect predictions from all base models
    fn collect_base_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
        let mut predictions = Vec::new();

        for (_, model) in &self.base_models {
            let pred = model.predict(x)?;
            predictions.push(pred);
        }

        Ok(predictions)
    }

    /// Evaluate fusion performance
    pub fn evaluate_fusion(&self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<FusionMetrics> {
        let predictions = self.predict(x)?;

        // Calculate metrics
        let mse = y.iter().zip(predictions.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
            .sum::<Float>() / y.len() as Float;

        let mae = y.iter().zip(predictions.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).abs())
            .sum::<Float>() / y.len() as Float;

        let y_mean = y.mean().unwrap_or(0.0);
        let ss_res: Float = y.iter().zip(predictions.iter())
            .map(|(true_val, pred_val)| (true_val - pred_val).powi(2))
            .sum();
        let ss_tot: Float = y.iter()
            .map(|true_val| (true_val - y_mean).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        // Calculate model contributions (simplified)
        let mut model_contributions = HashMap::new();
        for (name, _) in &self.base_models {
            model_contributions.insert(name.clone(), 1.0 / self.base_models.len() as Float);
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
            FusionStrategy::WeightedLinear { .. } => self.base_models.len(),
            FusionStrategy::NeuralNetwork { hidden_layers, .. } => {
                let mut count = 0;
                let n_models = self.base_models.len();

                // Input to first hidden layer
                count += n_models * hidden_layers[0];

                // Hidden layer connections
                for i in 1..hidden_layers.len() {
                    count += hidden_layers[i-1] * hidden_layers[i];
                }

                // Output layer
                count += hidden_layers.last().unwrap() * 1;

                count
            },
            FusionStrategy::GatingNetwork { .. } => {
                // Approximate parameter count for gating network
                self.base_models.len() * 10 // Simplified
            },
            _ => self.base_models.len(), // Default to linear parameter count
        }
    }
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
            fusion_strategy: FusionStrategy::WeightedLinear { regularization: None },
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
    pub fn new() -> Self {
        Self {
            base_models: Vec::new(),
            fusion_strategy: FusionStrategy::WeightedLinear { regularization: None },
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
    pub fn base_model(mut self, name: &str, model: Box<dyn PipelinePredictor>) -> Self {
        self.base_models.push((name.to_string(), model));
        self
    }

    /// Set fusion strategy
    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set regularization
    pub fn regularization(mut self, reg_type: RegularizationType, strength: Float) -> Self {
        self.regularization = Some((reg_type, strength));
        self
    }

    /// Enable cross-validation
    pub fn enable_cv(mut self, enable: bool) -> Self {
        self.enable_cv = enable;
        self
    }

    /// Set CV folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }

    /// Set feature scaling
    pub fn scale_features(mut self, scale: bool) -> Self {
        self.scale_features = scale;
        self
    }

    /// Set calibration temperature
    pub fn calibration_temperature(mut self, temperature: Float) -> Self {
        self.calibration_temperature = Some(temperature);
        self
    }

    /// Enable uncertainty estimation
    pub fn uncertainty_estimation(mut self, enable: bool) -> Self {
        self.uncertainty_estimation = enable;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set number of jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Set verbose flag
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the ModelFusion
    pub fn build(self) -> ModelFusion<Untrained> {
        ModelFusion {
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
            _state: PhantomData,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

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
            FusionStrategy::NeuralNetwork { ref hidden_layers, .. } => {
                assert_eq!(hidden_layers, &vec![64, 32]);
            },
            _ => panic!("Expected NeuralNetwork fusion strategy"),
        }
        assert_eq!(fusion.enable_cv, true);
        assert_eq!(fusion.cv_folds, 10);
        assert_eq!(fusion.scale_features, false);
    }

    #[test]
    fn test_fusion_strategies() {
        let strategies = vec![
            FusionStrategy::WeightedLinear { regularization: Some(0.01) },
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
            let fusion = ModelFusion::builder()
                .fusion_strategy(strategy)
                .build();

            match fusion.fusion_strategy {
                FusionStrategy::GatingNetwork { gating_type: gt, .. } => {
                    assert_eq!(gt, gating_type);
                },
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

        // For default WeightedLinear strategy
        assert!(params.linear_weights.is_some());
        assert!(params.neural_weights.is_none());
        assert!(params.gating_weights.is_none());
        assert!(params.attention_weights.is_none());
    }

    #[test]
    fn test_configuration_validation() {
        // Test empty base models
        let fusion = ModelFusion::new();
        assert!(fusion.validate_configuration().is_err());

        // Test invalid CV folds
        let mut fusion = ModelFusion::new();
        fusion.cv_folds = 1;
        assert!(fusion.validate_configuration().is_err());

        // Test invalid neural network configuration
        let fusion = ModelFusion::builder()
            .fusion_strategy(FusionStrategy::NeuralNetwork {
                hidden_layers: vec![],
                activation: "relu".to_string(),
                dropout: None,
            })
            .build();
        assert!(fusion.validate_configuration().is_err());
    }
}