//! Advanced testing and validation for kernel approximations
//!
//! This module provides comprehensive testing frameworks for convergence analysis,
//! approximation error bounds testing, and quality assessment of kernel approximation methods.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Convergence rate analysis for kernel approximation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
/// ConvergenceAnalyzer
pub struct ConvergenceAnalyzer {
    /// max_components
    pub max_components: usize,
    /// component_steps
    pub component_steps: Vec<usize>,
    /// n_trials
    pub n_trials: usize,
    /// convergence_tolerance
    pub convergence_tolerance: f64,
    /// reference_method
    pub reference_method: ReferenceMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// ReferenceMethod
pub enum ReferenceMethod {
    /// ExactKernel
    ExactKernel,
    /// HighPrecisionApproximation
    HighPrecisionApproximation { n_components: usize },
    /// MonteCarloEstimate
    MonteCarloEstimate { n_samples: usize },
}

impl ConvergenceAnalyzer {
    pub fn new(max_components: usize) -> Self {
        let component_steps = (1..=10)
            .map(|i| (max_components * i) / 10)
            .filter(|&x| x > 0)
            .collect();

        Self {
            max_components,
            component_steps,
            n_trials: 10,
            convergence_tolerance: 1e-6,
            reference_method: ReferenceMethod::ExactKernel,
        }
    }

    pub fn component_steps(mut self, steps: Vec<usize>) -> Self {
        self.component_steps = steps;
        self
    }

    pub fn n_trials(mut self, n_trials: usize) -> Self {
        self.n_trials = n_trials;
        self
    }

    pub fn convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.convergence_tolerance = tolerance;
        self
    }

    pub fn reference_method(mut self, method: ReferenceMethod) -> Self {
        self.reference_method = method;
        self
    }

    pub fn analyze_rbf_convergence(
        &self,
        x: &Array2<f64>,
        gamma: f64,
    ) -> Result<ConvergenceResult> {
        let mut convergence_rates = Vec::new();
        let mut approximation_errors = Vec::new();

        // Compute reference kernel matrix
        let reference_kernel = self.compute_reference_kernel(x, x, gamma)?;

        for &n_components in &self.component_steps {
            let mut trial_errors = Vec::new();

            for _ in 0..self.n_trials {
                // Generate RBF approximation
                let approximated_kernel =
                    self.compute_rbf_approximation(x, x, gamma, n_components)?;

                // Compute approximation error
                let error =
                    self.compute_approximation_error(&reference_kernel, &approximated_kernel)?;
                trial_errors.push(error);
            }

            let mean_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
            let std_error = {
                let variance = trial_errors
                    .iter()
                    .map(|&e| (e - mean_error).powi(2))
                    .sum::<f64>()
                    / trial_errors.len() as f64;
                variance.sqrt()
            };

            convergence_rates.push(n_components);
            approximation_errors.push(ApproximationError {
                mean: mean_error,
                std: std_error,
                min: trial_errors.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                max: trial_errors
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            });
        }

        // Compute convergence rate
        let convergence_rate =
            self.estimate_convergence_rate(&convergence_rates, &approximation_errors)?;
        let is_converged = approximation_errors.last().unwrap().mean < self.convergence_tolerance;

        Ok(ConvergenceResult {
            component_counts: convergence_rates,
            approximation_errors,
            convergence_rate,
            is_converged,
        })
    }

    fn compute_reference_kernel(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        gamma: f64,
    ) -> Result<Array2<f64>> {
        match &self.reference_method {
            ReferenceMethod::ExactKernel => self.compute_exact_rbf_kernel(x, y, gamma),
            ReferenceMethod::HighPrecisionApproximation { n_components } => {
                self.compute_rbf_approximation(x, y, gamma, *n_components)
            }
            ReferenceMethod::MonteCarloEstimate { n_samples } => {
                self.compute_monte_carlo_estimate(x, y, gamma, *n_samples)
            }
        }
    }

    fn compute_exact_rbf_kernel(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        gamma: f64,
    ) -> Result<Array2<f64>> {
        let n_x = x.nrows();
        let n_y = y.nrows();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                let diff = &x.row(i) - &y.row(j);
                let squared_norm = diff.dot(&diff);
                kernel_matrix[[i, j]] = (-gamma * squared_norm).exp();
            }
        }

        Ok(kernel_matrix)
    }

    fn compute_rbf_approximation(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        gamma: f64,
        n_components: usize,
    ) -> Result<Array2<f64>> {
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, (2.0 * gamma).sqrt()).unwrap();
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();

        let input_dim = x.ncols();

        // Generate random weights and biases
        let mut weights = Array2::zeros((n_components, input_dim));
        let mut biases = Array1::zeros(n_components);

        for i in 0..n_components {
            for j in 0..input_dim {
                weights[[i, j]] = rng.sample(normal);
            }
            biases[i] = rng.sample(uniform);
        }

        // Compute features for x and y
        let features_x = self.compute_rff_features(x, &weights, &biases, n_components)?;
        let features_y = self.compute_rff_features(y, &weights, &biases, n_components)?;

        // Approximate kernel matrix
        let n_x = x.nrows();
        let n_y = y.nrows();
        let mut kernel_matrix = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                kernel_matrix[[i, j]] = features_x.row(i).dot(&features_y.row(j));
            }
        }

        Ok(kernel_matrix)
    }

    fn compute_rff_features(
        &self,
        x: &Array2<f64>,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut features = Array2::zeros((n_samples, n_components));
        let scaling = (2.0 / n_components as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..n_components {
                let mut dot_product = 0.0;
                for k in 0..x.ncols() {
                    dot_product += x[[i, k]] * weights[[j, k]];
                }
                dot_product += biases[j];
                features[[i, j]] = scaling * dot_product.cos();
            }
        }

        Ok(features)
    }

    fn compute_monte_carlo_estimate(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        gamma: f64,
        n_samples: usize,
    ) -> Result<Array2<f64>> {
        // Monte Carlo estimation using multiple RBF approximations
        let mut kernel_sum = Array2::zeros((x.nrows(), y.nrows()));

        for _ in 0..n_samples {
            let approx = self.compute_rbf_approximation(x, y, gamma, 1000)?;
            kernel_sum = kernel_sum + approx;
        }

        Ok(kernel_sum / n_samples as f64)
    }

    fn compute_approximation_error(
        &self,
        reference: &Array2<f64>,
        approximation: &Array2<f64>,
    ) -> Result<f64> {
        if reference.shape() != approximation.shape() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions don't match".to_string(),
            ));
        }

        // Frobenius norm of the difference
        let diff = reference - approximation;
        let frobenius_norm = diff.mapv(|x| x * x).sum().sqrt();

        // Normalized by reference norm
        let reference_norm = reference.mapv(|x| x * x).sum().sqrt();

        if reference_norm > 0.0 {
            Ok(frobenius_norm / reference_norm)
        } else {
            Ok(frobenius_norm)
        }
    }

    fn estimate_convergence_rate(
        &self,
        components: &[usize],
        errors: &[ApproximationError],
    ) -> Result<f64> {
        if components.len() < 2 || errors.len() < 2 {
            return Ok(0.0);
        }

        // Fit power law: error ~ n^(-rate)
        let mut log_components = Vec::new();
        let mut log_errors = Vec::new();

        for (i, error) in errors.iter().enumerate() {
            if error.mean > 0.0 {
                log_components.push((components[i] as f64).ln());
                log_errors.push(error.mean.ln());
            }
        }

        if log_components.len() < 2 {
            return Ok(0.0);
        }

        // Simple linear regression in log space
        let n = log_components.len() as f64;
        let sum_x = log_components.iter().sum::<f64>();
        let sum_y = log_errors.iter().sum::<f64>();
        let sum_xy = log_components
            .iter()
            .zip(log_errors.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = log_components.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        Ok(-slope) // Negative because we expect error to decrease
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// ApproximationError
pub struct ApproximationError {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// ConvergenceResult
pub struct ConvergenceResult {
    /// component_counts
    pub component_counts: Vec<usize>,
    /// approximation_errors
    pub approximation_errors: Vec<ApproximationError>,
    /// convergence_rate
    pub convergence_rate: f64,
    /// is_converged
    pub is_converged: bool,
}

/// Error bounds testing framework
#[derive(Debug, Clone, Serialize, Deserialize)]
/// ErrorBoundsValidator
pub struct ErrorBoundsValidator {
    /// confidence_level
    pub confidence_level: f64,
    /// n_bootstrap_samples
    pub n_bootstrap_samples: usize,
    /// bound_types
    pub bound_types: Vec<BoundType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
/// BoundType
pub enum BoundType {
    /// Hoeffding
    Hoeffding,
    /// McDiarmid
    McDiarmid,
    /// Azuma
    Azuma,
    /// Bernstein
    Bernstein,
    /// Empirical
    Empirical,
}

impl ErrorBoundsValidator {
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
            n_bootstrap_samples: 1000,
            bound_types: vec![
                BoundType::Hoeffding,
                BoundType::McDiarmid,
                BoundType::Bernstein,
                BoundType::Empirical,
            ],
        }
    }

    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    pub fn n_bootstrap_samples(mut self, n_samples: usize) -> Self {
        self.n_bootstrap_samples = n_samples;
        self
    }

    pub fn bound_types(mut self, bounds: Vec<BoundType>) -> Self {
        self.bound_types = bounds;
        self
    }

    pub fn validate_rff_bounds(
        &self,
        x: &Array2<f64>,
        gamma: f64,
        n_components: usize,
    ) -> Result<ErrorBoundsResult> {
        let mut bound_results = HashMap::new();

        // Compute exact kernel for comparison
        let exact_kernel = self.compute_exact_kernel(x, gamma)?;

        // Generate multiple approximations for empirical distribution
        let mut approximation_errors = Vec::new();

        for _ in 0..self.n_bootstrap_samples {
            let approx_kernel = self.compute_rff_approximation(x, gamma, n_components)?;
            let error = self.compute_relative_error(&exact_kernel, &approx_kernel)?;
            approximation_errors.push(error);
        }

        approximation_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let empirical_mean =
            approximation_errors.iter().sum::<f64>() / approximation_errors.len() as f64;
        let empirical_variance = approximation_errors
            .iter()
            .map(|&e| (e - empirical_mean).powi(2))
            .sum::<f64>()
            / approximation_errors.len() as f64;

        // Compute theoretical bounds
        for bound_type in &self.bound_types {
            let bound = self.compute_theoretical_bound(
                bound_type,
                n_components,
                x.ncols(),
                self.confidence_level,
            )?;
            let empirical_quantile_idx =
                ((1.0 - self.confidence_level) * approximation_errors.len() as f64) as usize;
            let empirical_bound =
                approximation_errors[empirical_quantile_idx.min(approximation_errors.len() - 1)];

            bound_results.insert(
                bound_type.clone(),
                BoundValidation {
                    theoretical_bound: bound,
                    empirical_bound,
                    is_valid: bound >= empirical_bound,
                    tightness_ratio: if bound > 0.0 {
                        empirical_bound / bound
                    } else {
                        0.0
                    },
                },
            );
        }

        Ok(ErrorBoundsResult {
            empirical_mean,
            empirical_variance,
            empirical_quantiles: self.compute_quantiles(&approximation_errors),
            bound_validations: bound_results,
        })
    }

    fn compute_exact_kernel(&self, x: &Array2<f64>, gamma: f64) -> Result<Array2<f64>> {
        let n = x.nrows();
        let mut kernel = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let diff = &x.row(i) - &x.row(j);
                let squared_norm = diff.dot(&diff);
                kernel[[i, j]] = (-gamma * squared_norm).exp();
            }
        }

        Ok(kernel)
    }

    fn compute_rff_approximation(
        &self,
        x: &Array2<f64>,
        gamma: f64,
        n_components: usize,
    ) -> Result<Array2<f64>> {
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, (2.0 * gamma).sqrt()).unwrap();
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();

        let input_dim = x.ncols();
        let n_samples = x.nrows();

        // Generate random features
        let mut weights = Array2::zeros((n_components, input_dim));
        let mut biases = Array1::zeros(n_components);

        for i in 0..n_components {
            for j in 0..input_dim {
                weights[[i, j]] = rng.sample(normal);
            }
            biases[i] = rng.sample(uniform);
        }

        // Compute features
        let mut features = Array2::zeros((n_samples, n_components));
        let scaling = (2.0 / n_components as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..n_components {
                let mut dot_product = 0.0;
                for k in 0..input_dim {
                    dot_product += x[[i, k]] * weights[[j, k]];
                }
                dot_product += biases[j];
                features[[i, j]] = scaling * dot_product.cos();
            }
        }

        // Compute approximate kernel
        let mut kernel = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                kernel[[i, j]] = features.row(i).dot(&features.row(j));
            }
        }

        Ok(kernel)
    }

    fn compute_relative_error(&self, exact: &Array2<f64>, approx: &Array2<f64>) -> Result<f64> {
        let diff = exact - approx;
        let frobenius_error = diff.mapv(|x| x * x).sum().sqrt();
        let exact_norm = exact.mapv(|x| x * x).sum().sqrt();

        if exact_norm > 0.0 {
            Ok(frobenius_error / exact_norm)
        } else {
            Ok(frobenius_error)
        }
    }

    fn compute_theoretical_bound(
        &self,
        bound_type: &BoundType,
        n_components: usize,
        input_dim: usize,
        confidence: f64,
    ) -> Result<f64> {
        let delta = 1.0 - confidence;

        match bound_type {
            BoundType::Hoeffding => {
                // Hoeffding bound for RFF approximation
                let c = 2.0; // Bounded by assumption
                Ok(c * (2.0 * delta.ln().abs() / n_components as f64).sqrt())
            }
            BoundType::McDiarmid => {
                // McDiarmid bound with bounded differences
                let c = 4.0 / (n_components as f64).sqrt();
                Ok(c * (2.0 * delta.ln().abs() / n_components as f64).sqrt())
            }
            BoundType::Bernstein => {
                // Bernstein bound (simplified)
                let variance_bound = 1.0 / n_components as f64;
                let range_bound = 2.0 / (n_components as f64).sqrt();
                let term1 = (2.0 * variance_bound * delta.ln().abs() / n_components as f64).sqrt();
                let term2 = 2.0 * range_bound * delta.ln().abs() / (3.0 * n_components as f64);
                Ok(term1 + term2)
            }
            BoundType::Azuma => {
                // Azuma-Hoeffding for martingale differences
                let c = 1.0 / (n_components as f64).sqrt();
                Ok(c * (2.0 * delta.ln().abs()).sqrt())
            }
            BoundType::Empirical => {
                // Empirical bound based on Rademacher complexity
                let rademacher_complexity = (input_dim as f64 / n_components as f64).sqrt();
                Ok(2.0 * rademacher_complexity
                    + (delta.ln().abs() / (2.0 * n_components as f64)).sqrt())
            }
        }
    }

    fn compute_quantiles(&self, data: &[f64]) -> HashMap<String, f64> {
        let mut quantiles = HashMap::new();
        let n = data.len();

        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95, 0.99] {
            let idx = ((p * n as f64) as usize).min(n - 1);
            quantiles.insert(format!("q{}", (p * 100.0) as u8), data[idx]);
        }

        quantiles
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// BoundValidation
pub struct BoundValidation {
    /// theoretical_bound
    pub theoretical_bound: f64,
    /// empirical_bound
    pub empirical_bound: f64,
    /// is_valid
    pub is_valid: bool,
    /// tightness_ratio
    pub tightness_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// ErrorBoundsResult
pub struct ErrorBoundsResult {
    /// empirical_mean
    pub empirical_mean: f64,
    /// empirical_variance
    pub empirical_variance: f64,
    /// empirical_quantiles
    pub empirical_quantiles: HashMap<String, f64>,
    /// bound_validations
    pub bound_validations: HashMap<BoundType, BoundValidation>,
}

/// Quality assessment framework for kernel approximations
#[derive(Debug, Clone, Serialize, Deserialize)]
/// QualityAssessment
pub struct QualityAssessment {
    /// metrics
    pub metrics: Vec<QualityMetric>,
    /// baseline_methods
    pub baseline_methods: Vec<BaselineMethod>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
/// QualityMetric
pub enum QualityMetric {
    /// KernelAlignment
    KernelAlignment,
    /// SpectralError
    SpectralError,
    /// FrobeniusError
    FrobeniusError,
    /// NuclearNormError
    NuclearNormError,
    /// OperatorNormError
    OperatorNormError,
    /// RelativeError
    RelativeError,
    /// EffectiveRank
    EffectiveRank,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// BaselineMethod
pub enum BaselineMethod {
    /// RandomSampling
    RandomSampling,
    /// UniformSampling
    UniformSampling,
    /// ExactMethod
    ExactMethod,
    /// PreviousBest
    PreviousBest { method_name: String },
}

impl QualityAssessment {
    pub fn new() -> Self {
        Self {
            metrics: vec![
                QualityMetric::KernelAlignment,
                QualityMetric::SpectralError,
                QualityMetric::FrobeniusError,
                QualityMetric::RelativeError,
            ],
            baseline_methods: vec![BaselineMethod::RandomSampling, BaselineMethod::ExactMethod],
        }
    }

    pub fn assess_approximation(
        &self,
        exact_kernel: &Array2<f64>,
        approx_kernel: &Array2<f64>,
    ) -> Result<QualityResult> {
        let mut metric_scores = HashMap::new();

        for metric in &self.metrics {
            let score = self.compute_metric(metric, exact_kernel, approx_kernel)?;
            metric_scores.insert(metric.clone(), score);
        }

        let overall_score = self.compute_overall_score(&metric_scores);
        Ok(QualityResult {
            metric_scores: metric_scores.clone(),
            overall_score,
        })
    }

    fn compute_metric(
        &self,
        metric: &QualityMetric,
        exact: &Array2<f64>,
        approx: &Array2<f64>,
    ) -> Result<f64> {
        match metric {
            QualityMetric::KernelAlignment => {
                let exact_centered = self.center_kernel(exact)?;
                let approx_centered = self.center_kernel(approx)?;

                let numerator = self.frobenius_inner_product(&exact_centered, &approx_centered)?;
                let exact_norm = self.frobenius_norm(&exact_centered)?;
                let approx_norm = self.frobenius_norm(&approx_centered)?;

                if exact_norm > 0.0 && approx_norm > 0.0 {
                    Ok(numerator / (exact_norm * approx_norm))
                } else {
                    Ok(0.0)
                }
            }
            QualityMetric::FrobeniusError => {
                let diff = exact - approx;
                Ok(self.frobenius_norm(&diff)?)
            }
            QualityMetric::RelativeError => {
                let diff = exact - approx;
                let error_norm = self.frobenius_norm(&diff)?;
                let exact_norm = self.frobenius_norm(exact)?;

                if exact_norm > 0.0 {
                    Ok(error_norm / exact_norm)
                } else {
                    Ok(error_norm)
                }
            }
            QualityMetric::SpectralError => {
                // Simplified spectral error (largest singular value of difference)
                let diff = exact - approx;
                self.largest_singular_value(&diff)
            }
            QualityMetric::NuclearNormError => {
                let diff = exact - approx;
                self.nuclear_norm(&diff)
            }
            QualityMetric::OperatorNormError => {
                let diff = exact - approx;
                self.operator_norm(&diff)
            }
            QualityMetric::EffectiveRank => self.effective_rank(approx),
        }
    }

    fn center_kernel(&self, kernel: &Array2<f64>) -> Result<Array2<f64>> {
        let n = kernel.nrows();
        let row_means = kernel.mean_axis(Axis(1)).unwrap();
        let col_means = kernel.mean_axis(Axis(0)).unwrap();
        let overall_mean = kernel.mean().unwrap();

        let mut centered = kernel.clone();

        for i in 0..n {
            for j in 0..n {
                centered[[i, j]] = kernel[[i, j]] - row_means[i] - col_means[j] + overall_mean;
            }
        }

        Ok(centered)
    }

    fn frobenius_inner_product(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<f64> {
        if a.shape() != b.shape() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions don't match".to_string(),
            ));
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    fn frobenius_norm(&self, matrix: &Array2<f64>) -> Result<f64> {
        Ok(matrix.mapv(|x| x * x).sum().sqrt())
    }

    fn largest_singular_value(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Simplified using power iteration
        let n = matrix.nrows();
        let m = matrix.ncols();

        if n == 0 || m == 0 {
            return Ok(0.0);
        }

        let mut v = Array1::from_vec(vec![1.0; m]);
        v /= (v.dot(&v) as f64).sqrt();

        for _ in 0..50 {
            let mut av: Array1<f64> = Array1::zeros(n);
            for i in 0..n {
                for j in 0..m {
                    av[i] += matrix[[i, j]] * v[j];
                }
            }

            let mut ata_v: Array1<f64> = Array1::zeros(m);
            for j in 0..m {
                for i in 0..n {
                    ata_v[j] += matrix[[i, j]] * av[i];
                }
            }

            let norm = ata_v.dot(&ata_v).sqrt();
            if norm > 1e-12 {
                v = ata_v / norm;
            } else {
                break;
            }
        }

        let mut av: Array1<f64> = Array1::zeros(n);
        for i in 0..n {
            for j in 0..m {
                av[i] += matrix[[i, j]] * v[j];
            }
        }

        Ok(av.dot(&av).sqrt())
    }

    fn nuclear_norm(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Simplified nuclear norm estimation
        // In practice, this would use SVD
        let trace = (0..matrix.nrows().min(matrix.ncols()))
            .map(|i| matrix[[i, i]].abs())
            .sum::<f64>();
        Ok(trace)
    }

    fn operator_norm(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Operator norm is the largest singular value
        self.largest_singular_value(matrix)
    }

    fn effective_rank(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Simplified effective rank using trace and Frobenius norm
        let trace = (0..matrix.nrows().min(matrix.ncols()))
            .map(|i| matrix[[i, i]])
            .sum::<f64>();
        let frobenius_squared = matrix.mapv(|x| x * x).sum();

        if frobenius_squared > 0.0 {
            Ok(trace * trace / frobenius_squared)
        } else {
            Ok(0.0)
        }
    }

    fn compute_overall_score(&self, metric_scores: &HashMap<QualityMetric, f64>) -> f64 {
        // Simple weighted average (in practice, use domain-specific weights)
        let weights = [
            (QualityMetric::KernelAlignment, 0.4),
            (QualityMetric::RelativeError, 0.3),
            (QualityMetric::FrobeniusError, 0.2),
            (QualityMetric::EffectiveRank, 0.1),
        ];

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (metric, weight) in &weights {
            if let Some(&score) = metric_scores.get(metric) {
                let normalized_score = match metric {
                    QualityMetric::KernelAlignment => score, // Higher is better
                    QualityMetric::EffectiveRank => score / matrix_size_as_float(metric_scores),
                    _ => 1.0 / (1.0 + score), // Lower is better, transform to 0-1 scale
                };
                weighted_sum += weight * normalized_score;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
}

fn matrix_size_as_float(_metric_scores: &HashMap<QualityMetric, f64>) -> f64 {
    // Placeholder for matrix size normalization
    100.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// QualityResult
pub struct QualityResult {
    /// metric_scores
    pub metric_scores: HashMap<QualityMetric, f64>,
    /// overall_score
    pub overall_score: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;

    use scirs2_core::ndarray::{Array, Array2};
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_convergence_analyzer() {
        let x: Array2<f64> = Array::from_shape_fn((20, 5), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let analyzer = ConvergenceAnalyzer::new(50)
            .component_steps(vec![10, 20, 30, 40, 50])
            .n_trials(3);

        let result = analyzer.analyze_rbf_convergence(&x, 1.0).unwrap();

        assert_eq!(result.component_counts.len(), 5);
        assert!(result.convergence_rate >= 0.0);
    }

    #[test]
    fn test_error_bounds_validator() {
        let x: Array2<f64> = Array::from_shape_fn((15, 4), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let validator = ErrorBoundsValidator::new()
            .confidence_level(0.9)
            .n_bootstrap_samples(50);

        let result = validator.validate_rff_bounds(&x, 1.0, 20).unwrap();

        assert!(result.empirical_mean >= 0.0);
        assert!(result.empirical_variance >= 0.0);
        assert!(!result.bound_validations.is_empty());
    }

    #[test]
    fn test_quality_assessment() {
        let exact = Array::eye(10);
        let mut approx = Array::eye(10);
        approx[[0, 0]] = 0.9; // Small perturbation

        let assessment = QualityAssessment::new();
        let result = assessment.assess_approximation(&exact, &approx).unwrap();

        assert!(result.overall_score > 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(!result.metric_scores.is_empty());
    }
}
