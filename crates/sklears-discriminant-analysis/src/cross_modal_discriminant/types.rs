//! Types and configuration for Cross-Modal Discriminant Learning

use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

/// Configuration for Cross-Modal Discriminant Learning
#[derive(Debug, Clone)]
pub struct CrossModalDiscriminantLearningConfig {
    /// Fusion strategy for combining modalities
    pub fusion_strategy: FusionStrategy,
    /// Regularization parameter for cross-modal consistency
    pub cross_modal_regularization: Float,
    /// Weights for different modalities
    pub modality_weights: Option<Array1<Float>>,
    /// Whether to perform modality alignment
    pub align_modalities: bool,
    /// Dimensionality for common subspace
    pub common_subspace_dim: Option<usize>,
    /// Maximum correlation threshold for modality pruning
    pub correlation_threshold: Float,
    /// Learning rate for iterative optimization
    pub learning_rate: Float,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Domain adaptation strategy
    pub domain_adaptation_strategy: DomainAdaptationStrategy,
    /// Domain adaptation regularization weight
    pub domain_adaptation_weight: Float,
    /// Whether to perform domain invariant feature learning
    pub domain_invariant_learning: bool,
}

impl Default for CrossModalDiscriminantLearningConfig {
    fn default() -> Self {
        Self {
            fusion_strategy: FusionStrategy::ConcatenationFusion,
            cross_modal_regularization: 0.1,
            modality_weights: None,
            align_modalities: true,
            common_subspace_dim: None,
            correlation_threshold: 0.9,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-6,
            random_state: None,
            domain_adaptation_strategy: DomainAdaptationStrategy::None,
            domain_adaptation_weight: 0.1,
            domain_invariant_learning: false,
        }
    }
}

/// Fusion strategies for cross-modal learning
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Early fusion: concatenate all modalities
    ConcatenationFusion,
    /// Late fusion: separate models then ensemble
    LateFusion,
    /// Canonical correlation fusion
    CanonicalCorrelationFusion,
    /// Multi-modal deep fusion
    DeepFusion { hidden_dims: Vec<usize> },
    /// Cross-modal attention fusion
    AttentionFusion { num_heads: usize },
}

/// Domain adaptation strategies
#[derive(Debug, Clone)]
pub enum DomainAdaptationStrategy {
    /// No domain adaptation
    None,
    /// Maximum Mean Discrepancy (MMD)
    MMD,
    /// Correlation Alignment (CORAL)
    CORAL,
    /// Domain Adversarial Training
    DomainAdversarial,
    /// Deep Domain Confusion
    DeepDomainConfusion,
}

/// Modality alignment methods
#[derive(Debug, Clone)]
pub enum ModalityAlignment {
    /// No alignment
    None,
    /// Canonical Correlation Analysis
    CCA,
    /// Partial Least Squares
    PLS,
    /// Deep Canonical Correlation Analysis
    DeepCCA { hidden_dims: Vec<usize> },
    /// Generalized Canonical Correlation Analysis
    GeneralizedCCA,
}

/// Information about a modality (e.g., text, image, audio)
#[derive(Debug, Clone)]
pub struct ModalityInfo {
    /// Name of the modality
    pub name: String,
    /// Dimensionality of the modality
    pub dimension: usize,
    /// Weight/importance of this modality
    pub weight: Float,
    /// Whether this modality should be normalized
    pub normalize: bool,
}

/// Information about a domain (e.g., source/target domain)
#[derive(Debug, Clone)]
pub struct DomainInfo {
    /// Name of the domain
    pub name: String,
    /// Number of samples in this domain
    pub sample_count: usize,
    /// Domain-specific regularization weight
    pub regularization_weight: Float,
}
