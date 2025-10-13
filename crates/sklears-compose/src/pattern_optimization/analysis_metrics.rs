//! Analysis and metrics for optimization performance
//!
//! This module provides comprehensive analysis tools including:
//! - Convergence analysis and detection algorithms
//! - Sensitivity analysis for parameter robustness
//! - Performance metrics and benchmarking
//! - Statistical analysis of optimization results
//! - Bottleneck identification and scalability assessment

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use scirs2_core::ndarray::{Array1, Array2};
use crate::core::SklResult;
use super::optimization_core::Solution;

/// Optimization metrics collector and analyzer
#[derive(Debug)]
pub struct OptimizationMetrics {
    /// Metrics identifier
    pub metrics_id: String,
    /// Performance measurements
    pub performance_metrics: PerformanceMetrics,
    /// Convergence tracking
    pub convergence_metrics: ConvergenceMetrics,
    /// Resource usage tracking
    pub resource_metrics: ResourceMetrics,
    /// Quality measurements
    pub quality_metrics: QualityMetrics,
    /// Statistical summaries
    pub statistical_metrics: StatisticalMetrics,
}

/// Performance metrics for optimization algorithms
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total execution time
    pub execution_time: Duration,
    /// Iterations completed
    pub iterations_completed: u64,
    /// Function evaluations performed
    pub function_evaluations: u64,
    /// Gradient evaluations performed
    pub gradient_evaluations: u64,
    /// Hessian evaluations performed
    pub hessian_evaluations: u64,
    /// Solutions generated
    pub solutions_generated: u64,
    /// Average time per iteration
    pub average_iteration_time: Duration,
    /// Throughput (solutions/second)
    pub throughput: f64,
    /// Efficiency ratio
    pub efficiency_ratio: f64,
}

/// Convergence analysis and detection
#[derive(Debug)]
pub struct ConvergenceAnalyzer {
    /// Analyzer identifier
    pub analyzer_id: String,
    /// Convergence detection algorithms
    pub detection_algorithms: Vec<ConvergenceDetector>,
    /// Historical convergence data
    pub convergence_history: Vec<ConvergencePoint>,
    /// Current convergence state
    pub current_state: ConvergenceState,
    /// Convergence parameters
    pub parameters: ConvergenceParameters,
}

/// Convergence metrics structure
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Whether algorithm has converged
    pub converged: bool,
    /// Convergence rate estimate
    pub convergence_rate: f64,
    /// Stagnation period length
    pub stagnation_period: u64,
    /// Oscillation measure
    pub oscillation_measure: f64,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Convergence confidence
    pub confidence: f64,
    /// Estimated iterations to convergence
    pub estimated_remaining_iterations: Option<u64>,
}

/// Trend direction enumeration
#[derive(Debug, Clone)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Declining trend
    Declining,
    /// Stable/plateaued
    Stable,
    /// Oscillating
    Oscillating,
    /// Unknown/insufficient data
    Unknown,
}

/// Individual convergence detection point
#[derive(Debug, Clone)]
pub struct ConvergencePoint {
    /// Iteration number
    pub iteration: u64,
    /// Objective value
    pub objective_value: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Step size
    pub step_size: f64,
    /// Constraint violation
    pub constraint_violation: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Convergence state tracking
#[derive(Debug, Clone)]
pub struct ConvergenceState {
    /// Current convergence status
    pub status: ConvergenceStatus,
    /// Convergence criterion values
    pub criterion_values: HashMap<String, f64>,
    /// Time since last improvement
    pub time_since_improvement: Duration,
    /// Iterations since last improvement
    pub iterations_since_improvement: u64,
}

/// Convergence status enumeration
#[derive(Debug, Clone)]
pub enum ConvergenceStatus {
    /// Not yet converged
    NotConverged,
    /// Slow convergence detected
    SlowConvergence,
    /// Fast convergence detected
    FastConvergence,
    /// Converged to solution
    Converged,
    /// Stagnated without improvement
    Stagnated,
    /// Diverging from solution
    Diverged,
}

/// Convergence detection parameters
#[derive(Debug, Clone)]
pub struct ConvergenceParameters {
    /// Objective tolerance
    pub objective_tolerance: f64,
    /// Gradient tolerance
    pub gradient_tolerance: f64,
    /// Step tolerance
    pub step_tolerance: f64,
    /// Maximum stagnation iterations
    pub max_stagnation_iterations: u64,
    /// Relative tolerance enabled
    pub relative_tolerance: bool,
    /// Window size for trend analysis
    pub trend_window_size: usize,
}

/// Convergence detector trait
pub trait ConvergenceDetector: Send + Sync {
    /// Check convergence based on current state
    fn check_convergence(&self, points: &[ConvergencePoint]) -> SklResult<bool>;

    /// Estimate convergence rate
    fn estimate_convergence_rate(&self, points: &[ConvergencePoint]) -> f64;

    /// Get detector name
    fn get_name(&self) -> &str;

    /// Get detector parameters
    fn get_parameters(&self) -> HashMap<String, f64>;
}

/// Sensitivity analysis for parameter robustness
#[derive(Debug)]
pub struct SensitivityAnalyzer {
    /// Analyzer identifier
    pub analyzer_id: String,
    /// Sensitivity computation methods
    pub computation_methods: Vec<SensitivityMethod>,
    /// Parameter sensitivity results
    pub sensitivity_results: HashMap<String, SensitivityResult>,
    /// Robustness assessments
    pub robustness_assessments: Vec<RobustnessAssessment>,
}

/// Sensitivity analysis method
pub trait SensitivityMethod: Send + Sync {
    /// Compute sensitivity for parameter
    fn compute_sensitivity(
        &self,
        parameter_name: &str,
        base_solution: &Solution,
        perturbation_size: f64,
    ) -> SklResult<SensitivityResult>;

    /// Get method name
    fn get_method_name(&self) -> &str;
}

/// Sensitivity analysis result
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Parameter name
    pub parameter_name: String,
    /// Sensitivity coefficient
    pub sensitivity_coefficient: f64,
    /// Gradient sensitivity
    pub gradient_sensitivity: Array1<f64>,
    /// Hessian sensitivity (if available)
    pub hessian_sensitivity: Option<Array2<f64>>,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Statistical significance
    pub p_value: f64,
}

/// Robustness assessment
#[derive(Debug, Clone)]
pub struct RobustnessAssessment {
    /// Assessment identifier
    pub assessment_id: String,
    /// Robustness score (0-1)
    pub robustness_score: f64,
    /// Critical parameters
    pub critical_parameters: Vec<String>,
    /// Vulnerability analysis
    pub vulnerabilities: Vec<Vulnerability>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Parameter vulnerability
#[derive(Debug, Clone)]
pub struct Vulnerability {
    /// Parameter name
    pub parameter_name: String,
    /// Vulnerability severity
    pub severity: VulnerabilitySeverity,
    /// Impact description
    pub impact: String,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Vulnerability severity levels
#[derive(Debug, Clone)]
pub enum VulnerabilitySeverity {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// I/O operations count
    pub io_operations: u64,
    /// Network operations count
    pub network_operations: u64,
    /// Temporary storage used
    pub temp_storage: usize,
}

/// Quality metrics for solutions
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Solution accuracy
    pub accuracy: f64,
    /// Solution precision
    pub precision: f64,
    /// Solution robustness
    pub robustness: f64,
    /// Numerical stability
    pub numerical_stability: f64,
    /// Constraint satisfaction level
    pub constraint_satisfaction: f64,
    /// Diversity score
    pub diversity_score: f64,
}

/// Statistical metrics
#[derive(Debug, Clone)]
pub struct StatisticalMetrics {
    /// Mean objective value
    pub mean_objective: f64,
    /// Standard deviation
    pub std_deviation: f64,
    /// Variance
    pub variance: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

impl Default for OptimizationMetrics {
    fn default() -> Self {
        Self {
            metrics_id: format!("metrics_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis()),
            performance_metrics: PerformanceMetrics::default(),
            convergence_metrics: ConvergenceMetrics::default(),
            resource_metrics: ResourceMetrics::default(),
            quality_metrics: QualityMetrics::default(),
            statistical_metrics: StatisticalMetrics::default(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_secs(0),
            iterations_completed: 0,
            function_evaluations: 0,
            gradient_evaluations: 0,
            hessian_evaluations: 0,
            solutions_generated: 0,
            average_iteration_time: Duration::from_secs(0),
            throughput: 0.0,
            efficiency_ratio: 0.0,
        }
    }
}

impl Default for ConvergenceAnalyzer {
    fn default() -> Self {
        Self {
            analyzer_id: format!("conv_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis()),
            detection_algorithms: Vec::new(),
            convergence_history: Vec::new(),
            current_state: ConvergenceState::default(),
            parameters: ConvergenceParameters::default(),
        }
    }
}

impl Default for ConvergenceMetrics {
    fn default() -> Self {
        Self {
            converged: false,
            convergence_rate: 0.0,
            stagnation_period: 0,
            oscillation_measure: 0.0,
            trend_direction: TrendDirection::Unknown,
            confidence: 0.0,
            estimated_remaining_iterations: None,
        }
    }
}

impl Default for ConvergenceState {
    fn default() -> Self {
        Self {
            status: ConvergenceStatus::NotConverged,
            criterion_values: HashMap::new(),
            time_since_improvement: Duration::from_secs(0),
            iterations_since_improvement: 0,
        }
    }
}

impl Default for ConvergenceParameters {
    fn default() -> Self {
        Self {
            objective_tolerance: 1e-6,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-10,
            max_stagnation_iterations: 100,
            relative_tolerance: true,
            trend_window_size: 10,
        }
    }
}

impl Default for SensitivityAnalyzer {
    fn default() -> Self {
        Self {
            analyzer_id: format!("sens_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis()),
            computation_methods: Vec::new(),
            sensitivity_results: HashMap::new(),
            robustness_assessments: Vec::new(),
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            memory_usage: 0,
            peak_memory_usage: 0,
            cpu_usage: 0.0,
            io_operations: 0,
            network_operations: 0,
            temp_storage: 0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            robustness: 0.0,
            numerical_stability: 0.0,
            constraint_satisfaction: 0.0,
            diversity_score: 0.0,
        }
    }
}

impl Default for StatisticalMetrics {
    fn default() -> Self {
        Self {
            mean_objective: 0.0,
            std_deviation: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            confidence_intervals: HashMap::new(),
        }
    }
}

impl ConvergenceAnalyzer {
    /// Create new convergence analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Add convergence point
    pub fn add_point(&mut self, point: ConvergencePoint) {
        self.convergence_history.push(point);
        self.update_convergence_state();
    }

    /// Update convergence state based on history
    fn update_convergence_state(&mut self) {
        if self.convergence_history.len() < 2 {
            return;
        }

        let recent_points = &self.convergence_history[self.convergence_history.len().saturating_sub(self.parameters.trend_window_size)..];

        // Analyze trend
        self.current_state.status = self.determine_convergence_status(recent_points);

        // Update criterion values
        if let Some(latest) = self.convergence_history.last() {
            self.current_state.criterion_values.insert("gradient_norm".to_string(), latest.gradient_norm);
            self.current_state.criterion_values.insert("objective_value".to_string(), latest.objective_value);
        }
    }

    /// Determine convergence status from recent points
    fn determine_convergence_status(&self, points: &[ConvergencePoint]) -> ConvergenceStatus {
        if points.len() < 2 {
            return ConvergenceStatus::NotConverged;
        }

        let latest = &points[points.len() - 1];

        // Check gradient-based convergence
        if latest.gradient_norm < self.parameters.gradient_tolerance {
            return ConvergenceStatus::Converged;
        }

        // Check for stagnation
        let improvement_count = points.windows(2)
            .filter(|w| w[1].objective_value < w[0].objective_value)
            .count();

        if improvement_count == 0 && points.len() >= self.parameters.max_stagnation_iterations as usize {
            return ConvergenceStatus::Stagnated;
        }

        // Check convergence rate
        if points.len() >= 3 {
            let recent_improvements: Vec<f64> = points.windows(2)
                .map(|w| w[0].objective_value - w[1].objective_value)
                .collect();

            let avg_improvement: f64 = recent_improvements.iter().sum::<f64>() / recent_improvements.len() as f64;

            if avg_improvement > 0.01 {
                ConvergenceStatus::FastConvergence
            } else if avg_improvement > 0.0001 {
                ConvergenceStatus::SlowConvergence
            } else {
                ConvergenceStatus::NotConverged
            }
        } else {
            ConvergenceStatus::NotConverged
        }
    }

    /// Get convergence metrics
    pub fn get_metrics(&self) -> ConvergenceMetrics {
        let mut metrics = ConvergenceMetrics::default();

        metrics.converged = matches!(self.current_state.status, ConvergenceStatus::Converged);
        metrics.convergence_rate = self.estimate_convergence_rate();
        metrics.trend_direction = self.analyze_trend();

        metrics
    }

    /// Estimate convergence rate
    fn estimate_convergence_rate(&self) -> f64 {
        if self.convergence_history.len() < 3 {
            return 0.0;
        }

        let recent_points = &self.convergence_history[self.convergence_history.len().saturating_sub(5)..];
        let improvements: Vec<f64> = recent_points.windows(2)
            .map(|w| (w[0].objective_value - w[1].objective_value) / w[0].objective_value.abs())
            .collect();

        improvements.iter().sum::<f64>() / improvements.len() as f64
    }

    /// Analyze trend direction
    fn analyze_trend(&self) -> TrendDirection {
        if self.convergence_history.len() < 3 {
            return TrendDirection::Unknown;
        }

        let recent_objectives: Vec<f64> = self.convergence_history
            .iter()
            .rev()
            .take(5)
            .map(|p| p.objective_value)
            .collect();

        if recent_objectives.len() < 2 {
            return TrendDirection::Unknown;
        }

        let improvements = recent_objectives.windows(2)
            .filter(|w| w[1] < w[0])
            .count();

        let degradations = recent_objectives.windows(2)
            .filter(|w| w[1] > w[0])
            .count();

        if improvements > degradations + 1 {
            TrendDirection::Improving
        } else if degradations > improvements + 1 {
            TrendDirection::Declining
        } else if improvements == degradations {
            TrendDirection::Oscillating
        } else {
            TrendDirection::Stable
        }
    }
}