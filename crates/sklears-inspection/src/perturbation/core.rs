//! Core types and configuration for perturbation analysis
//!
//! This module contains the fundamental types and configurations used
//! throughout the perturbation analysis system.

use crate::Float;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for perturbation analysis
#[derive(Debug, Clone)]
pub struct PerturbationConfig {
    /// Perturbation strategy to use
    pub strategy: PerturbationStrategy,
    /// Magnitude of perturbation
    pub magnitude: Float,
    /// Number of perturbation samples to generate
    pub n_samples: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Distribution parameters for noise-based perturbation
    pub noise_distribution: NoiseDistribution,
    /// Step size for adversarial perturbation
    pub adversarial_step_size: Float,
    /// Number of adversarial steps
    pub adversarial_steps: usize,
    /// Target class for adversarial attacks (None for regression)
    pub target_class: Option<usize>,
    /// Preserve feature correlations
    pub preserve_correlations: bool,
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            strategy: PerturbationStrategy::Gaussian,
            magnitude: 0.1,
            n_samples: 100,
            random_state: Some(42),
            noise_distribution: NoiseDistribution::Gaussian { std: 0.1 },
            adversarial_step_size: 0.01,
            adversarial_steps: 10,
            target_class: None,
            preserve_correlations: false,
        }
    }
}

/// Perturbation strategies
#[derive(Debug, Clone, Copy)]
pub enum PerturbationStrategy {
    /// Gaussian noise perturbation
    Gaussian,
    /// Uniform noise perturbation
    Uniform,
    /// Adversarial perturbation (gradient-based)
    Adversarial,
    /// Synthetic data generation
    Synthetic,
    /// Distribution-preserving perturbation
    DistributionPreserving,
    /// Structured perturbation (feature groups)
    Structured,
    /// Salt-and-pepper noise
    SaltPepper,
    /// Dropout-style perturbation
    Dropout,
}

/// Noise distribution parameters
#[derive(Debug, Clone)]
pub enum NoiseDistribution {
    /// Gaussian distribution with specified standard deviation
    Gaussian { std: Float },
    /// Uniform distribution with specified range
    Uniform { min: Float, max: Float },
    /// Laplace distribution with specified scale
    Laplace { scale: Float },
    /// Exponential distribution with specified rate
    Exponential { rate: Float },
}

/// Result of perturbation analysis
#[derive(Debug, Clone)]
pub struct PerturbationResult {
    /// Original data
    pub original_data: Array2<Float>,
    /// Perturbed data samples
    pub perturbed_data: Vec<Array2<Float>>,
    /// Original predictions
    pub original_predictions: Vec<Float>,
    /// Perturbed predictions
    pub perturbed_predictions: Vec<Vec<Float>>,
    /// Perturbation statistics
    pub perturbation_stats: PerturbationStats,
    /// Robustness metrics
    pub robustness_metrics: RobustnessMetrics,
}

/// Statistics about perturbations
#[derive(Debug, Clone)]
pub struct PerturbationStats {
    /// Mean perturbation magnitude per feature
    pub mean_magnitude: Array1<Float>,
    /// Standard deviation of perturbations per feature
    pub std_magnitude: Array1<Float>,
    /// Maximum perturbation per feature
    pub max_magnitude: Array1<Float>,
    /// Correlation between original and perturbed features
    pub correlation_matrix: Array2<Float>,
}

/// Robustness metrics
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    /// Mean prediction stability (lower is more robust)
    pub prediction_stability: Float,
    /// Standard deviation of prediction changes
    pub prediction_variance: Float,
    /// Maximum prediction change
    pub max_prediction_change: Float,
    /// Fraction of predictions that changed significantly
    pub significant_change_fraction: Float,
    /// Local Lipschitz estimate
    pub local_lipschitz: Float,
}

/// Configuration for perturbation pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Execution mode (sequential, parallel, conditional, branching)
    pub execution_mode: ExecutionMode,
    /// Maximum number of parallel stages
    pub max_parallel_stages: usize,
    /// Enable stage result caching
    pub enable_caching: bool,
    /// Memory limit for pipeline execution (in MB)
    pub memory_limit_mb: usize,
    /// Timeout for pipeline execution (in seconds)
    pub timeout_seconds: u64,
    /// Enable progress tracking
    pub enable_progress_tracking: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Sequential,
            max_parallel_stages: 4,
            enable_caching: true,
            memory_limit_mb: 512,
            timeout_seconds: 300,
            enable_progress_tracking: true,
        }
    }
}

/// Execution mode for perturbation pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Execute stages sequentially
    Sequential,
    /// Execute stages in parallel
    Parallel,
    /// Execute stages conditionally based on data characteristics
    Conditional,
    /// Execute stages in branching pattern
    Branching,
}

/// Individual stage in the perturbation pipeline
#[derive(Debug, Clone)]
pub struct PerturbationStage {
    /// Stage identifier
    pub id: String,
    /// Stage name
    pub name: String,
    /// Perturbation configuration for this stage
    pub config: PerturbationConfig,
    /// Execution condition (optional)
    pub condition: Option<ExecutionCondition>,
    /// Stage dependencies (stage IDs that must complete first)
    pub dependencies: Vec<String>,
    /// Whether this stage is enabled
    pub enabled: bool,
    /// Stage priority (higher values execute first in parallel mode)
    pub priority: u32,
    /// Maximum retry attempts
    pub max_retries: u32,
}

/// Condition for conditional execution
#[derive(Debug, Clone)]
pub enum ExecutionCondition {
    /// Execute if data has specific characteristics
    DataCharacteristics {
        /// Minimum number of samples
        min_samples: Option<usize>,
        /// Maximum number of samples
        max_samples: Option<usize>,
        /// Minimum number of features
        min_features: Option<usize>,
        /// Maximum number of features
        max_features: Option<usize>,
        /// Required data sparsity threshold
        sparsity_threshold: Option<Float>,
    },
    /// Execute if previous stage meets criteria
    PreviousStageResult {
        /// Stage ID to check
        stage_id: String,
        /// Required success status
        success_required: bool,
        /// Minimum quality threshold
        quality_threshold: Option<Float>,
    },
    /// Execute based on custom function
    Custom {
        /// Custom condition function
        condition_fn:
            fn(&scirs2_core::ndarray::ArrayView2<Float>, &HashMap<String, StageResult>) -> bool,
    },
}

/// Result of a pipeline stage execution
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage ID
    pub stage_id: String,
    /// Execution success status
    pub success: bool,
    /// Perturbed data from this stage
    pub perturbed_data: Option<Vec<Array2<Float>>>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Quality metrics
    pub quality_metrics: StageQualityMetrics,
    /// Error message if stage failed
    pub error_message: Option<String>,
    /// Number of retry attempts
    pub retry_count: u32,
}

/// Quality metrics for a pipeline stage
#[derive(Debug, Clone)]
pub struct StageQualityMetrics {
    /// Perturbation magnitude statistics
    pub perturbation_magnitude: Float,
    /// Data diversity score
    pub diversity_score: Float,
    /// Robustness score
    pub robustness_score: Float,
    /// Coverage score (how well the perturbations cover the input space)
    pub coverage_score: Float,
}

/// Pipeline execution metadata
#[derive(Debug, Clone)]
pub struct PipelineMetadata {
    /// Total execution time
    pub total_execution_time_ms: u64,
    /// Peak memory usage
    pub peak_memory_usage_bytes: usize,
    /// Number of stages executed
    pub stages_executed: usize,
    /// Number of stages skipped
    pub stages_skipped: usize,
    /// Number of stages failed
    pub stages_failed: usize,
    /// Overall success rate
    pub success_rate: Float,
}

/// Result of pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Pipeline metadata
    pub metadata: PipelineMetadata,
    /// Results from each stage
    pub stage_results: HashMap<String, StageResult>,
    /// Final combined perturbed data
    pub final_perturbed_data: Vec<Array2<Float>>,
    /// Pipeline execution graph
    pub execution_graph: ExecutionGraph,
}

/// Graph representing pipeline execution flow
#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    /// Nodes in the execution graph
    pub nodes: Vec<ExecutionNode>,
    /// Edges representing dependencies
    pub edges: Vec<ExecutionEdge>,
}

/// Node in the execution graph
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Stage ID
    pub stage_id: String,
    /// Execution status
    pub status: ExecutionStatus,
    /// Start time
    pub start_time: std::time::Instant,
    /// End time
    pub end_time: Option<std::time::Instant>,
}

/// Edge in the execution graph
#[derive(Debug, Clone)]
pub struct ExecutionEdge {
    /// Source stage ID
    pub from_stage: String,
    /// Target stage ID
    pub to_stage: String,
    /// Dependency type
    pub dependency_type: DependencyType,
}

/// Execution status of a stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    /// Stage is pending execution
    Pending,
    /// Stage is currently running
    Running,
    /// Stage completed successfully
    Completed,
    /// Stage failed
    Failed,
    /// Stage was skipped
    Skipped,
}

/// Type of dependency between stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyType {
    /// Hard dependency - must complete successfully
    Hard,
    /// Soft dependency - can fail but affects execution
    Soft,
    /// Data dependency - output is used as input
    Data,
}
