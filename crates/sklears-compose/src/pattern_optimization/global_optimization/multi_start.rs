//! Multi-start optimization methods for global search.
//!
//! This module implements multi-start optimization algorithms that perform multiple
//! local optimizations from different starting points to increase the probability
//! of finding the global optimum. Multi-start methods are particularly effective
//! when the optimization landscape has multiple local optima.
//!
//! # Algorithm Overview
//!
//! Multi-start optimization works by:
//! 1. Generating diverse starting points across the search space
//! 2. Performing local optimization from each starting point
//! 3. Collecting and analyzing all local optima found
//! 4. Determining the best solution among all local optima
//! 5. Providing coverage analysis and convergence statistics
//!
//! # Key Features
//!
//! - **Smart starting point generation**: Various strategies for initial point selection
//! - **Coverage analysis**: Comprehensive analysis of search space exploration
//! - **Clustering analysis**: Identification of solution clusters and basins
//! - **Adaptive strategies**: Dynamic adjustment of starting points based on results
//! - **Parallel execution**: Efficient parallel processing of multiple starts
//! - **Statistical analysis**: Detailed convergence and performance statistics
//!
//! # Examples
//!
//! ```rust
//! use sklears_compose::pattern_optimization::global_optimization::multi_start::*;
//! use scirs2_core::ndarray::Array1;
//!
//! // Create multi-start parameters
//! let params = MultiStartParameters::builder()
//!     .num_starts(100)
//!     .starting_point_strategy(StartingPointStrategy::LatinHypercube)
//!     .clustering_enabled(true)
//!     .parallel_execution(true)
//!     .build();
//!
//! // The multi-start method would be implemented by specific algorithms
//! // that use these parameters and follow the MultiStartMethod trait
//! ```
//!
//! ## Starting Point Strategies
//!
//! Various strategies for generating initial starting points:
//! - **Random Uniform**: Uniform random distribution in bounds
//! - **Latin Hypercube**: Space-filling Latin hypercube sampling
//! - **Sobol Sequence**: Low-discrepancy Sobol sequence
//! - **Adaptive**: Adaptive sampling based on previous results
//! - **Grid-based**: Regular grid sampling
//! - **Gradient-informed**: Using gradient information for placement
//!
//! ## Coverage Analysis
//!
//! Comprehensive analysis includes:
//! - Search space coverage percentage
//! - Identification of uncovered regions
//! - Coverage efficiency metrics
//! - Recommendation for additional starting points

use std::collections::{HashMap, HashSet};
use std::time::Duration;

// SciRS2 imports following compliance requirements
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::error::CoreError;
use scirs2_core::random::{Random, rng};

// Local imports
use sklears_core::error::Result as SklResult;
use crate::enhanced_errors::PipelineError;
use super::basin_hopping::{Solution, OptimizationProblem, ConvergenceStatus};

/// Multi-start optimization method for global search.
///
/// Implements various multi-start strategies that generate multiple starting points
/// and perform local optimization from each point to find global optima.
pub trait MultiStartMethod: Send + Sync {
    /// Generate starting points for multi-start optimization.
    ///
    /// Creates a set of diverse starting points across the search space using
    /// various strategies including random sampling, space-filling designs,
    /// and adaptive methods.
    ///
    /// # Arguments
    /// * `problem` - Optimization problem definition with bounds
    /// * `num_starts` - Number of starting points to generate
    ///
    /// # Returns
    /// Vector of starting points
    fn generate_starting_points(&self, problem: &OptimizationProblem, num_starts: usize) -> SklResult<Vec<Array1<f64>>>;

    /// Perform local optimization from multiple starting points.
    ///
    /// Executes local optimization algorithms from each starting point,
    /// potentially in parallel, and collects all resulting local optima.
    ///
    /// # Arguments
    /// * `starting_points` - Vector of starting points
    /// * `problem` - Optimization problem definition
    ///
    /// # Returns
    /// Vector of solutions from all starting points
    fn optimize_from_points(&self, starting_points: &[Array1<f64>], problem: &OptimizationProblem) -> SklResult<Vec<Solution>>;

    /// Cluster solutions to identify distinct local optima.
    ///
    /// Groups similar solutions together to identify unique local optima
    /// and estimate the number of basins in the optimization landscape.
    ///
    /// # Arguments
    /// * `solutions` - All solutions found from multiple starts
    ///
    /// # Returns
    /// Clustered solutions with analysis
    fn cluster_solutions(&self, solutions: &[Solution]) -> SklResult<SolutionClusters>;

    /// Analyze coverage of the search space.
    ///
    /// Evaluates how well the starting points and solutions cover the
    /// search space and identifies regions that may need more exploration.
    ///
    /// # Arguments
    /// * `starting_points` - All starting points used
    /// * `solutions` - All solutions found
    /// * `problem` - Optimization problem with bounds
    ///
    /// # Returns
    /// Coverage analysis results
    fn analyze_coverage(&self, starting_points: &[Array1<f64>], solutions: &[Solution], problem: &OptimizationProblem) -> SklResult<CoverageAnalysis>;

    /// Get convergence statistics for multi-start optimization.
    ///
    /// Computes detailed statistics about convergence behavior across
    /// all starting points including success rates and consistency measures.
    ///
    /// # Arguments
    /// * `solutions` - All solutions from multi-start optimization
    ///
    /// # Returns
    /// Convergence statistics and analysis
    fn get_convergence_statistics(&self, solutions: &[Solution]) -> SklResult<MultiStartConvergenceStats>;

    /// Generate recommendations for improving multi-start performance.
    ///
    /// Analyzes results and provides recommendations for better starting
    /// point strategies, number of starts, and parameter adjustments.
    ///
    /// # Arguments
    /// * `analysis_results` - Results from coverage and convergence analysis
    ///
    /// # Returns
    /// Recommendations for improvement
    fn generate_recommendations(&self, analysis_results: &MultiStartAnalysis) -> SklResult<MultiStartRecommendations>;

    /// Adaptive starting point generation based on previous results.
    ///
    /// Uses information from previous optimization runs to generate
    /// more effective starting points for subsequent runs.
    ///
    /// # Arguments
    /// * `previous_solutions` - Solutions from previous optimization runs
    /// * `problem` - Optimization problem definition
    /// * `num_additional_starts` - Number of additional points to generate
    ///
    /// # Returns
    /// Adaptively generated starting points
    fn adaptive_starting_points(&self, previous_solutions: &[Solution], problem: &OptimizationProblem, num_additional_starts: usize) -> SklResult<Vec<Array1<f64>>>;
}

/// Configuration parameters for multi-start optimization.
#[derive(Debug, Clone)]
pub struct MultiStartParameters {
    /// Number of starting points to generate
    pub num_starts: usize,
    /// Strategy for generating starting points
    pub starting_point_strategy: StartingPointStrategy,
    /// Whether to enable solution clustering
    pub clustering_enabled: bool,
    /// Clustering tolerance for identifying distinct solutions
    pub clustering_tolerance: f64,
    /// Whether to enable parallel execution
    pub parallel_execution: bool,
    /// Maximum number of parallel threads
    pub max_parallel_threads: Option<usize>,
    /// Local optimization parameters
    pub local_optimizer_config: LocalOptimizerConfig,
    /// Coverage analysis parameters
    pub coverage_analysis_config: CoverageAnalysisConfig,
    /// Adaptive strategy parameters
    pub adaptive_strategy_config: Option<AdaptiveStrategyConfig>,
}

impl Default for MultiStartParameters {
    fn default() -> Self {
        Self {
            num_starts: 100,
            starting_point_strategy: StartingPointStrategy::LatinHypercube,
            clustering_enabled: true,
            clustering_tolerance: 1e-6,
            parallel_execution: true,
            max_parallel_threads: None,
            local_optimizer_config: LocalOptimizerConfig::default(),
            coverage_analysis_config: CoverageAnalysisConfig::default(),
            adaptive_strategy_config: None,
        }
    }
}

impl MultiStartParameters {
    /// Create a new builder for MultiStartParameters.
    pub fn builder() -> MultiStartParametersBuilder {
        MultiStartParametersBuilder::new()
    }

    /// Validate the multi-start parameters.
    pub fn validate(&self) -> SklResult<()> {
        if self.num_starts == 0 {
            return Err(CoreError::InvalidInput("Number of starts must be positive".to_string()).into());
        }
        if self.clustering_tolerance <= 0.0 {
            return Err(CoreError::InvalidInput("Clustering tolerance must be positive".to_string()).into());
        }
        if let Some(threads) = self.max_parallel_threads {
            if threads == 0 {
                return Err(CoreError::InvalidInput("Number of parallel threads must be positive".to_string()).into());
            }
        }
        Ok(())
    }
}

/// Builder for MultiStartParameters.
#[derive(Debug)]
pub struct MultiStartParametersBuilder {
    num_starts: usize,
    starting_point_strategy: StartingPointStrategy,
    clustering_enabled: bool,
    clustering_tolerance: f64,
    parallel_execution: bool,
    max_parallel_threads: Option<usize>,
    local_optimizer_config: LocalOptimizerConfig,
    coverage_analysis_config: CoverageAnalysisConfig,
    adaptive_strategy_config: Option<AdaptiveStrategyConfig>,
}

impl MultiStartParametersBuilder {
    pub fn new() -> Self {
        Self {
            num_starts: 100,
            starting_point_strategy: StartingPointStrategy::LatinHypercube,
            clustering_enabled: true,
            clustering_tolerance: 1e-6,
            parallel_execution: true,
            max_parallel_threads: None,
            local_optimizer_config: LocalOptimizerConfig::default(),
            coverage_analysis_config: CoverageAnalysisConfig::default(),
            adaptive_strategy_config: None,
        }
    }

    /// Set the number of starting points.
    pub fn num_starts(mut self, num_starts: usize) -> Self {
        self.num_starts = num_starts;
        self
    }

    /// Set the starting point generation strategy.
    pub fn starting_point_strategy(mut self, strategy: StartingPointStrategy) -> Self {
        self.starting_point_strategy = strategy;
        self
    }

    /// Enable or disable solution clustering.
    pub fn clustering_enabled(mut self, enabled: bool) -> Self {
        self.clustering_enabled = enabled;
        self
    }

    /// Set the clustering tolerance.
    pub fn clustering_tolerance(mut self, tolerance: f64) -> Self {
        self.clustering_tolerance = tolerance;
        self
    }

    /// Enable or disable parallel execution.
    pub fn parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }

    /// Set the maximum number of parallel threads.
    pub fn max_parallel_threads(mut self, threads: usize) -> Self {
        self.max_parallel_threads = Some(threads);
        self
    }

    /// Set the local optimizer configuration.
    pub fn local_optimizer_config(mut self, config: LocalOptimizerConfig) -> Self {
        self.local_optimizer_config = config;
        self
    }

    /// Set the coverage analysis configuration.
    pub fn coverage_analysis_config(mut self, config: CoverageAnalysisConfig) -> Self {
        self.coverage_analysis_config = config;
        self
    }

    /// Set the adaptive strategy configuration.
    pub fn adaptive_strategy_config(mut self, config: AdaptiveStrategyConfig) -> Self {
        self.adaptive_strategy_config = Some(config);
        self
    }

    /// Build the MultiStartParameters.
    pub fn build(self) -> SklResult<MultiStartParameters> {
        let params = MultiStartParameters {
            num_starts: self.num_starts,
            starting_point_strategy: self.starting_point_strategy,
            clustering_enabled: self.clustering_enabled,
            clustering_tolerance: self.clustering_tolerance,
            parallel_execution: self.parallel_execution,
            max_parallel_threads: self.max_parallel_threads,
            local_optimizer_config: self.local_optimizer_config,
            coverage_analysis_config: self.coverage_analysis_config,
            adaptive_strategy_config: self.adaptive_strategy_config,
        };
        params.validate()?;
        Ok(params)
    }
}

/// Strategies for generating starting points in multi-start optimization.
#[derive(Debug, Clone)]
pub enum StartingPointStrategy {
    /// Uniform random sampling within bounds
    UniformRandom,
    /// Latin hypercube sampling for space-filling design
    LatinHypercube,
    /// Sobol sequence for low-discrepancy sampling
    SobolSequence,
    /// Regular grid sampling
    Grid { points_per_dimension: usize },
    /// Adaptive sampling based on function landscape
    Adaptive { adaptation_factor: f64 },
    /// Gradient-informed starting point placement
    GradientInformed { gradient_weight: f64 },
    /// Clustering-based sampling to avoid redundancy
    ClusteringBased { min_distance: f64 },
    /// Custom starting point generation function
    Custom { name: String },
}

/// Local optimizer configuration for multi-start methods.
#[derive(Debug, Clone)]
pub struct LocalOptimizerConfig {
    /// Optimization method to use for local search
    pub method: LocalOptimizationMethod,
    /// Maximum iterations for each local optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum function evaluations per local optimization
    pub max_function_evaluations: usize,
    /// Timeout for each local optimization
    pub timeout: Option<Duration>,
}

impl Default for LocalOptimizerConfig {
    fn default() -> Self {
        Self {
            method: LocalOptimizationMethod::BFGS,
            max_iterations: 1000,
            tolerance: 1e-6,
            max_function_evaluations: 10000,
            timeout: Some(Duration::from_secs(60)),
        }
    }
}

/// Local optimization methods available for multi-start.
#[derive(Debug, Clone)]
pub enum LocalOptimizationMethod {
    /// Broyden-Fletcher-Goldfarb-Shanno
    BFGS,
    /// Limited-memory BFGS
    LBFGS,
    /// L-BFGS-B with bounds
    LBFGSB,
    /// Conjugate gradient
    ConjugateGradient,
    /// Nelder-Mead simplex
    NelderMead,
    /// Trust region methods
    TrustRegion,
    /// Newton's method
    Newton,
}

/// Configuration for coverage analysis.
#[derive(Debug, Clone)]
pub struct CoverageAnalysisConfig {
    /// Grid resolution for coverage computation
    pub grid_resolution: usize,
    /// Minimum coverage threshold
    pub min_coverage_threshold: f64,
    /// Whether to identify uncovered regions
    pub identify_uncovered_regions: bool,
    /// Maximum number of uncovered regions to report
    pub max_uncovered_regions: usize,
}

impl Default for CoverageAnalysisConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 50,
            min_coverage_threshold: 0.8,
            identify_uncovered_regions: true,
            max_uncovered_regions: 10,
        }
    }
}

/// Configuration for adaptive strategies.
#[derive(Debug, Clone)]
pub struct AdaptiveStrategyConfig {
    /// Learning rate for adaptive adjustments
    pub learning_rate: f64,
    /// Memory length for historical information
    pub memory_length: usize,
    /// Minimum improvement threshold for adaptation
    pub improvement_threshold: f64,
    /// Maximum adaptation cycles
    pub max_adaptation_cycles: usize,
}

impl Default for AdaptiveStrategyConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            memory_length: 10,
            improvement_threshold: 0.01,
            max_adaptation_cycles: 5,
        }
    }
}

/// Solution clusters from multi-start optimization.
#[derive(Debug)]
pub struct SolutionClusters {
    /// Individual clusters of solutions
    pub clusters: Vec<SolutionCluster>,
    /// Number of unique local optima found
    pub num_unique_optima: usize,
    /// Clustering statistics
    pub clustering_stats: ClusteringStatistics,
}

/// Individual cluster of similar solutions.
#[derive(Debug)]
pub struct SolutionCluster {
    /// Solutions in this cluster
    pub solutions: Vec<Solution>,
    /// Cluster centroid
    pub centroid: Array1<f64>,
    /// Cluster radius (maximum distance from centroid)
    pub radius: f64,
    /// Best solution in cluster
    pub best_solution: Solution,
    /// Cluster quality metrics
    pub quality_metrics: ClusterQualityMetrics,
}

/// Statistics about the clustering process.
#[derive(Debug)]
pub struct ClusteringStatistics {
    /// Total number of solutions clustered
    pub total_solutions: usize,
    /// Number of clusters found
    pub num_clusters: usize,
    /// Average cluster size
    pub avg_cluster_size: f64,
    /// Silhouette score for clustering quality
    pub silhouette_score: f64,
    /// Dunn index for cluster separation
    pub dunn_index: f64,
}

/// Quality metrics for individual clusters.
#[derive(Debug)]
pub struct ClusterQualityMetrics {
    /// Intra-cluster variance
    pub intra_cluster_variance: f64,
    /// Cluster compactness measure
    pub compactness: f64,
    /// Cluster isolation measure
    pub isolation: f64,
    /// Confidence in cluster validity
    pub confidence: f64,
}

/// Coverage analysis results for multi-start methods.
#[derive(Debug)]
pub struct CoverageAnalysis {
    /// Search space coverage percentage
    pub coverage_percentage: f64,
    /// Uncovered regions identified
    pub uncovered_regions: Vec<UncoveredRegion>,
    /// Coverage efficiency score
    pub coverage_efficiency: f64,
    /// Distribution uniformity measure
    pub distribution_uniformity: f64,
    /// Coverage quality metrics
    pub coverage_metrics: CoverageMetrics,
}

/// Uncovered region in search space.
#[derive(Debug)]
pub struct UncoveredRegion {
    /// Region bounds (lower and upper bounds)
    pub bounds: (Array1<f64>, Array1<f64>),
    /// Estimated importance of this region
    pub importance_score: f64,
    /// Recommended number of additional starting points
    pub recommended_starts: usize,
    /// Region characteristics
    pub characteristics: RegionCharacteristics,
}

/// Characteristics of uncovered regions.
#[derive(Debug)]
pub struct RegionCharacteristics {
    /// Region volume
    pub volume: f64,
    /// Distance to nearest starting point
    pub min_distance_to_start: f64,
    /// Estimated function landscape ruggedness
    pub estimated_ruggedness: f64,
    /// Priority for exploration
    pub exploration_priority: ExplorationPriority,
}

/// Priority levels for exploring uncovered regions.
#[derive(Debug, Clone)]
pub enum ExplorationPriority {
    /// High priority - likely contains important optima
    High,
    /// Medium priority - moderate importance
    Medium,
    /// Low priority - low expected value
    Low,
}

/// Coverage quality metrics.
#[derive(Debug)]
pub struct CoverageMetrics {
    /// Dispersion measure (larger is better)
    pub dispersion: f64,
    /// Discrepancy measure (smaller is better)
    pub discrepancy: f64,
    /// Fill distance
    pub fill_distance: f64,
    /// Separation distance
    pub separation_distance: f64,
}

/// Multi-start convergence statistics.
#[derive(Debug)]
pub struct MultiStartConvergenceStats {
    /// Overall success rate across all starts
    pub overall_success_rate: f64,
    /// Average convergence time per start
    pub avg_convergence_time: Duration,
    /// Distribution of convergence times
    pub convergence_time_distribution: Vec<Duration>,
    /// Consistency measure across starts
    pub consistency_measure: f64,
    /// Success rate by starting point strategy
    pub success_by_strategy: HashMap<String, f64>,
    /// Function evaluation statistics
    pub evaluation_statistics: EvaluationStatistics,
}

/// Function evaluation statistics.
#[derive(Debug)]
pub struct EvaluationStatistics {
    /// Total function evaluations across all starts
    pub total_evaluations: u64,
    /// Average evaluations per successful start
    pub avg_evaluations_per_success: f64,
    /// Evaluation efficiency (success rate per evaluation)
    pub evaluation_efficiency: f64,
    /// Peak evaluation rate
    pub peak_evaluation_rate: f64,
}

/// Complete multi-start analysis results.
#[derive(Debug)]
pub struct MultiStartAnalysis {
    /// Convergence statistics
    pub convergence_stats: MultiStartConvergenceStats,
    /// Coverage analysis
    pub coverage_analysis: CoverageAnalysis,
    /// Solution clustering results
    pub solution_clusters: SolutionClusters,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
}

/// Performance summary for multi-start optimization.
#[derive(Debug)]
pub struct PerformanceSummary {
    /// Best solution found
    pub best_solution: Solution,
    /// Number of local optima found
    pub num_local_optima: usize,
    /// Total optimization time
    pub total_time: Duration,
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Recommendation confidence
    pub recommendation_confidence: f64,
}

/// Recommendations for improving multi-start performance.
#[derive(Debug)]
pub struct MultiStartRecommendations {
    /// Optimal number of starting points
    pub optimal_num_starts: usize,
    /// Recommended starting point strategy
    pub recommended_strategy: StartingPointStrategy,
    /// Parameter adjustments
    pub parameter_adjustments: HashMap<String, f64>,
    /// Additional regions to explore
    pub additional_exploration_regions: Vec<UncoveredRegion>,
    /// Confidence in recommendations
    pub confidence: f64,
    /// Expected improvement estimate
    pub expected_improvement: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_start_parameters_default() {
        let params = MultiStartParameters::default();
        assert!(params.num_starts > 0);
        assert!(params.clustering_tolerance > 0.0);
        assert!(params.clustering_enabled);
        assert!(params.parallel_execution);
    }

    #[test]
    fn test_multi_start_parameters_builder() {
        let params = MultiStartParameters::builder()
            .num_starts(50)
            .clustering_enabled(false)
            .parallel_execution(false)
            .build()
            .unwrap();

        assert_eq!(params.num_starts, 50);
        assert!(!params.clustering_enabled);
        assert!(!params.parallel_execution);
    }

    #[test]
    fn test_multi_start_parameters_validation() {
        // Invalid number of starts
        let result = MultiStartParameters::builder()
            .num_starts(0)
            .build();
        assert!(result.is_err());

        // Invalid clustering tolerance
        let result = MultiStartParameters::builder()
            .clustering_tolerance(-1.0)
            .build();
        assert!(result.is_err());

        // Invalid number of threads
        let result = MultiStartParameters::builder()
            .max_parallel_threads(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_starting_point_strategy() {
        let strategies = vec![
            StartingPointStrategy::UniformRandom,
            StartingPointStrategy::LatinHypercube,
            StartingPointStrategy::SobolSequence,
            StartingPointStrategy::Grid { points_per_dimension: 10 },
        ];

        for strategy in strategies {
            // Test that strategies can be created and cloned
            let _cloned = strategy.clone();
        }
    }

    #[test]
    fn test_local_optimizer_config_default() {
        let config = LocalOptimizerConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.tolerance > 0.0);
        assert!(config.max_function_evaluations > 0);
        matches!(config.method, LocalOptimizationMethod::BFGS);
    }

    #[test]
    fn test_coverage_analysis_config_default() {
        let config = CoverageAnalysisConfig::default();
        assert!(config.grid_resolution > 0);
        assert!(config.min_coverage_threshold > 0.0 && config.min_coverage_threshold <= 1.0);
        assert!(config.identify_uncovered_regions);
        assert!(config.max_uncovered_regions > 0);
    }

    #[test]
    fn test_adaptive_strategy_config_default() {
        let config = AdaptiveStrategyConfig::default();
        assert!(config.learning_rate > 0.0 && config.learning_rate <= 1.0);
        assert!(config.memory_length > 0);
        assert!(config.improvement_threshold >= 0.0);
        assert!(config.max_adaptation_cycles > 0);
    }

    #[test]
    fn test_exploration_priority() {
        let priorities = vec![
            ExplorationPriority::High,
            ExplorationPriority::Medium,
            ExplorationPriority::Low,
        ];

        for priority in priorities {
            let _cloned = priority.clone();
        }
    }

    #[test]
    fn test_solution_cluster_creation() {
        use scirs2_core::ndarray::Array1;
        use std::time::Duration;
        use super::super::basin_hopping::{Solution, SolutionMetadata, ConvergenceStatus};

        let metadata = SolutionMetadata {
            function_evaluations: 100,
            computation_time: Duration::from_millis(50),
            convergence_status: ConvergenceStatus::Converged,
            optimality_certificate: None,
        };

        let solution = Solution {
            variables: Array1::from_vec(vec![1.0, 2.0]),
            objective_value: 5.0,
            constraint_violation: 0.0,
            gradient: None,
            hessian: None,
            metadata,
        };

        let quality_metrics = ClusterQualityMetrics {
            intra_cluster_variance: 0.1,
            compactness: 0.9,
            isolation: 0.8,
            confidence: 0.95,
        };

        let cluster = SolutionCluster {
            solutions: vec![solution.clone()],
            centroid: Array1::from_vec(vec![1.0, 2.0]),
            radius: 0.1,
            best_solution: solution,
            quality_metrics,
        };

        assert_eq!(cluster.solutions.len(), 1);
        assert_eq!(cluster.centroid.len(), 2);
        assert!(cluster.radius > 0.0);
    }
}