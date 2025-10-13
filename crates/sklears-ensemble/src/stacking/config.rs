//! Configuration types for stacking ensemble methods
//!
//! This module provides configuration structures and enums for all stacking
//! ensemble variants including simple stacking, blending, and multi-layer stacking.

use sklears_core::types::Float;

/// Stacking Classifier Configuration
#[derive(Debug, Clone)]
pub struct StackingConfig {
    /// Number of folds for cross-validation
    pub cv: usize,
    /// Whether to use cross-validation predictions as features
    pub use_probabilities: bool,
    /// Random seed for cross-validation
    pub random_state: Option<u64>,
    /// Whether to fit the base estimators on the full dataset
    pub passthrough: bool,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            cv: 5,
            use_probabilities: false,
            random_state: None,
            passthrough: false,
        }
    }
}

/// Meta-learning strategies for combining predictions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetaLearningStrategy {
    /// Simple linear regression (default)
    LinearRegression,
    /// Ridge regression with L2 regularization
    Ridge(Float),
    /// Lasso regression with L1 regularization
    Lasso(Float),
    /// Elastic net with combined L1/L2 regularization
    ElasticNet(Float, Float),
    /// Logistic regression for classification
    LogisticRegression,
    /// Support Vector Machine
    SVM,
    /// Neural network meta-learner
    NeuralNetwork,
    /// Bayesian averaging
    BayesianAveraging,
}

/// Advanced meta-feature engineering strategies
#[derive(Debug, Clone, PartialEq)]
pub enum MetaFeatureStrategy {
    /// Raw predictions only (default)
    Raw,
    /// Include statistical transformations (mean, std, skew, etc.)
    Statistical,
    /// Include pairwise interactions between predictions
    Interactions,
    /// Include confidence-based features
    ConfidenceBased,
    /// Include diversity measures between base learners
    DiversityBased,
    /// Comprehensive feature engineering (all of the above)
    Comprehensive,
    /// Temporal features for time-series data
    Temporal,
    /// Spatial features for geographic data
    Spatial,
    /// Spectral features using FFT and wavelets
    Spectral,
    /// Information-theoretic features (mutual information, entropy)
    InformationTheoretic,
    /// Neural embedding features
    NeuralEmbedding,
    /// Kernel-based features
    KernelBased,
    /// Advanced polynomial and basis expansions
    BasisExpansion,
    /// Meta-learning features
    MetaLearning,
}

impl Default for MetaFeatureStrategy {
    fn default() -> Self {
        Self::Raw
    }
}

/// Configuration for a single stacking layer
#[derive(Debug, Clone)]
pub struct StackingLayerConfig {
    /// Number of base estimators in this layer
    pub n_estimators: usize,
    /// Meta-learning strategy for this layer
    pub meta_strategy: MetaLearningStrategy,
    /// Whether to use probability outputs instead of predictions
    pub use_probabilities: bool,
    /// Cross-validation folds for this layer
    pub cv_folds: usize,
    /// Whether to include original features in meta-features
    pub passthrough: bool,
    /// L2 regularization strength for meta-learner
    pub meta_regularization: Float,
    /// Meta-feature engineering strategy
    pub meta_feature_strategy: MetaFeatureStrategy,
    /// Enable polynomial feature generation
    pub polynomial_features: bool,
    /// Polynomial degree for feature generation
    pub polynomial_degree: usize,
}

impl Default for StackingLayerConfig {
    fn default() -> Self {
        Self {
            n_estimators: 5,
            meta_strategy: MetaLearningStrategy::LinearRegression,
            use_probabilities: false,
            cv_folds: 5,
            passthrough: false,
            meta_regularization: 0.1,
            meta_feature_strategy: MetaFeatureStrategy::Raw,
            polynomial_features: false,
            polynomial_degree: 2,
        }
    }
}

/// Multi-layer stacking configuration
#[derive(Debug, Clone)]
pub struct MultiLayerStackingConfig {
    /// Configuration for each stacking layer
    pub layers: Vec<StackingLayerConfig>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Final meta-learner strategy
    pub final_meta_strategy: MetaLearningStrategy,
    /// Whether to use ensemble pruning
    pub enable_pruning: bool,
    /// Diversity threshold for ensemble pruning
    pub diversity_threshold: Float,
    /// Confidence-based weighting
    pub confidence_weighting: bool,
}

impl MultiLayerStackingConfig {
    pub fn new() -> Self {
        Self {
            layers: vec![StackingLayerConfig::default()],
            random_state: None,
            final_meta_strategy: MetaLearningStrategy::LogisticRegression,
            enable_pruning: false,
            diversity_threshold: 0.1,
            confidence_weighting: false,
        }
    }

    /// Add a layer to the configuration
    pub fn add_layer(mut self, layer_config: StackingLayerConfig) -> Self {
        self.layers.push(layer_config);
        self
    }

    /// Set the final meta-learning strategy
    pub fn final_meta_strategy(mut self, strategy: MetaLearningStrategy) -> Self {
        self.final_meta_strategy = strategy;
        self
    }

    /// Enable ensemble pruning
    pub fn enable_pruning(mut self, enable: bool) -> Self {
        self.enable_pruning = enable;
        self
    }

    /// Set diversity threshold for pruning
    pub fn diversity_threshold(mut self, threshold: Float) -> Self {
        self.diversity_threshold = threshold;
        self
    }

    /// Enable confidence-based weighting
    pub fn confidence_weighting(mut self, enable: bool) -> Self {
        self.confidence_weighting = enable;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Create a deep stacking configuration with multiple layers
    pub fn deep_stacking(n_layers: usize, estimators_per_layer: usize) -> Self {
        let mut config = Self::new();
        config.layers.clear();

        for i in 0..n_layers {
            let layer_config = StackingLayerConfig {
                n_estimators: estimators_per_layer,
                meta_strategy: if i == 0 {
                    MetaLearningStrategy::Ridge(0.1)
                } else {
                    MetaLearningStrategy::ElasticNet(0.1, 0.1)
                },
                use_probabilities: i > 0, // Use probabilities in higher layers
                cv_folds: 5,
                passthrough: i == 0, // Only pass through features in first layer
                meta_regularization: 0.1 * (i + 1) as Float, // Increase regularization in higher layers
                meta_feature_strategy: if i == 0 {
                    MetaFeatureStrategy::Statistical
                } else {
                    MetaFeatureStrategy::Comprehensive
                },
                polynomial_features: i > 0, // Enable polynomial features in higher layers
                polynomial_degree: 2,
            };
            config.layers.push(layer_config);
        }

        config.final_meta_strategy = MetaLearningStrategy::BayesianAveraging;
        config.enable_pruning = true;
        config.diversity_threshold = 0.15;
        config.confidence_weighting = true;
        config
    }

    /// Create a configuration with advanced meta-feature engineering
    pub fn with_meta_feature_engineering() -> Self {
        let mut config = Self::new();
        config.layers[0].meta_feature_strategy = MetaFeatureStrategy::Comprehensive;
        config.layers[0].polynomial_features = true;
        config.layers[0].polynomial_degree = 2;
        config
    }

    /// Create a configuration optimized for statistical meta-features
    pub fn with_statistical_features() -> Self {
        let mut config = Self::new();
        config.layers[0].meta_feature_strategy = MetaFeatureStrategy::Statistical;
        config.confidence_weighting = true;
        config
    }

    /// Create a configuration with interaction-based meta-features
    pub fn with_interaction_features() -> Self {
        let mut config = Self::new();
        config.layers[0].meta_feature_strategy = MetaFeatureStrategy::Interactions;
        config.layers[0].polynomial_features = true;
        config.layers[0].polynomial_degree = 3;
        config
    }

    /// Create a configuration with diversity-based meta-features
    pub fn with_diversity_features() -> Self {
        let mut config = Self::new();
        config.layers[0].meta_feature_strategy = MetaFeatureStrategy::DiversityBased;
        config.enable_pruning = true;
        config.diversity_threshold = 0.1;
        config
    }
}

impl Default for MultiLayerStackingConfig {
    fn default() -> Self {
        Self::new()
    }
}

// Type aliases for the trait bounds (placeholder for now)
pub trait BaseEstimator {}
pub trait MetaEstimator {}
