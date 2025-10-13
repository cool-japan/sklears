//! Configuration types and enums for voting ensemble methods

use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

/// Voting strategy for ensemble
#[derive(Debug, Clone, PartialEq)]
pub enum VotingStrategy {
    /// Hard voting (majority vote)
    Hard,
    /// Soft voting (average of probabilities)
    Soft,
    /// Weighted voting based on estimator weights
    Weighted,
    /// Confidence-weighted voting based on prediction confidence
    ConfidenceWeighted,
    /// Bayesian model averaging
    BayesianAveraging,
    /// Rank-based voting using ordinal rankings
    RankBased,
    /// Meta-learning voting with learned combination weights
    MetaVoting,
    /// Dynamic weight adjustment based on recent performance
    DynamicWeightAdjustment,
    /// Uncertainty-aware voting considering prediction uncertainty
    UncertaintyAware,
    /// Consensus-based voting requiring minimum agreement
    ConsensusBased,
    /// Entropy-weighted voting favoring low-entropy predictions
    EntropyWeighted,
    /// Variance-weighted voting favoring low-variance predictions
    VarianceWeighted,
    /// Bootstrap aggregation (Bagging)
    BootstrapAggregation,
    /// Temperature-scaled soft voting
    TemperatureScaled,
    /// Adaptive ensemble voting
    AdaptiveEnsemble,
}

/// Configuration for Voting Classifier
#[derive(Debug, Clone)]
pub struct VotingClassifierConfig {
    pub voting: VotingStrategy,
    pub weights: Option<Vec<Float>>,
    pub confidence_weighting: bool,
    pub confidence_threshold: Float,
    pub min_confidence_weight: Float,
    pub enable_uncertainty: bool,
    pub temperature: Float,
    pub meta_regularization: Float,
    pub n_bootstrap_samples: usize,
    pub uncertainty_method: UncertaintyMethod,
    pub consensus_threshold: Float,
    pub entropy_weight_factor: Float,
    pub variance_weight_factor: Float,
    pub weight_adjustment_rate: Float,
}

impl Default for VotingClassifierConfig {
    fn default() -> Self {
        Self {
            voting: VotingStrategy::Hard,
            weights: None,
            confidence_weighting: false,
            confidence_threshold: 0.5,
            min_confidence_weight: 0.1,
            enable_uncertainty: false,
            temperature: 1.0,
            meta_regularization: 0.0,
            n_bootstrap_samples: 100,
            uncertainty_method: UncertaintyMethod::Variance,
            consensus_threshold: 0.7,
            entropy_weight_factor: 1.0,
            variance_weight_factor: 1.0,
            weight_adjustment_rate: 0.1,
        }
    }
}

/// Uncertainty estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyMethod {
    /// Use prediction variance as uncertainty measure
    Variance,
    /// Use prediction entropy as uncertainty measure
    Entropy,
    /// Use ensemble disagreement as uncertainty measure
    EnsembleDisagreement,
    /// Use prediction confidence as uncertainty measure
    Confidence,
    /// Combined uncertainty using multiple methods
    Combined,
    /// Bootstrap-based uncertainty estimation
    Bootstrap,
    /// Monte Carlo dropout uncertainty (if applicable)
    MCDropout,
    /// Bayesian uncertainty estimation
    Bayesian,
    /// Temperature-scaled uncertainty
    TemperatureScaled,
    /// Quantile-based uncertainty
    Quantile,
}

/// Ensemble size recommendations
#[derive(Debug, Clone)]
pub struct EnsembleSizeRecommendations {
    pub min_size: usize,
    pub max_size: usize,
    pub sweet_spot: usize,
    pub diminishing_returns_threshold: usize,
}

/// Ensemble size analysis results
#[derive(Debug, Clone)]
pub struct EnsembleSizeAnalysis {
    pub performance_curve: Array1<Float>,
    pub diversity_curve: Array1<Float>,
    pub optimal_size: usize,
    pub performance_plateau_size: usize,
    pub diversity_saturation_size: usize,
}
